"""The decision API: one shared state, many typed questions, one prefill, typed answers.

A trained policy answers a question with one forward pass and a softmax over the letter
rows at the last prompt token. The prompt is render_prefix(state) + render_suffix(question),
and attention is causal, so the prefix's hidden states do not depend on the suffix. ask()
therefore runs the model body once over the prefix, keeps its key/value cache, and runs one
batched forward over the padded suffixes with positions continuing from the prefix length.
Every question sees the state and nothing else, and the logits equal the single-pass ones.

No text is ever generated: the only readout is the 26 letter rows of the output embedding.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, DynamicLayer

from rlcd.metrics import entropy_confidence
from rlcd.policies import HFDecoderPolicy, load_policy
from rlcd.schema import Question, render_prefix, render_suffix

PREFIX_HEADER = "User: Context:\n"
PREFIX_TAIL = "\n\n"
NOUL_OPTIONS = ("true", "false")

# Our own confidence definitions; the metrics module computes them for evaluation.
CONFIDENCE_DEFINITIONS = {
    "confidence": "the largest probability over the declared options (the probability of value)",
    "entropy_confidence": "1 - H(p) / ln(K) with H the entropy of p in nats over the K declared options; "
                          "1 for a point mass, 0 for uniform",
    "score": "for a score question with K ordered levels, sum_i p_i * i / (K - 1): the probability-"
             "weighted mean level index scaled to [0, 1], with 0 the lowest level and 1 the highest",
    "p_true": "for a noul question, the probability of the option true",
    "noul confidence": "max(p_true, 1 - p_true)",
}


@dataclass
class ChoiceQ:
    """An unordered choice among 2..26 options."""
    question: str
    options: list[str]


@dataclass
class ScoreQ:
    """An ordered choice among 2..26 levels, listed from low to high."""
    question: str
    levels: list[str]


@dataclass
class NoulQ:
    """A yes/no question, rendered verbatim with the options true and false, exactly as a
    dataset noul row is rendered (for example "Should an on-call engineer be paged
    immediately?")."""
    question: str


Primitive = ChoiceQ | ScoreQ | NoulQ


def to_question(state: str, q: Primitive) -> Question:
    """The validated Question a primitive stands for over the given state."""
    if isinstance(q, ChoiceQ):
        return Question("choice", state, q.question, list(q.options)).validate()
    if isinstance(q, ScoreQ):
        return Question("score", state, q.question, list(q.levels), ordered=True).validate()
    if isinstance(q, NoulQ):
        return Question("noul", state, q.question, list(NOUL_OPTIONS)).validate()
    raise TypeError(f"expected a ChoiceQ, ScoreQ or NoulQ, got {type(q).__name__}")


# ---------- the tokenization split ----------

_SPLIT_PROBES = ["plain words", "ends with a digit 42", "ends with punctuation.", "ends with a bang!",
                 "ends with a paren (x)", "ends with a colon:", "trailing newline\n", "caf\u00e9",
                 "two lines\nsecond line", "\u4e2d\u6587"]


def check_split(tok) -> None:
    """Raise unless tokenizing the prefix and the suffix separately gives the tokens of the
    whole prompt, for prompts whose state ends in each of several ways. The split point is
    the blank line before Question:, which this tokenizer must not merge across."""
    for state in _SPLIT_PROBES:
        q = Question("choice", state, "which one?", ["a", "b"])
        whole = tok.encode(render_prefix(q.context) + render_suffix(q), add_special_tokens=False)
        split = (tok.encode(render_prefix(q.context), add_special_tokens=False)
                 + tok.encode(render_suffix(q), add_special_tokens=False))
        if whole != split:
            raise ValueError(f"this tokenizer merges tokens across the prefix split for state {state!r}: "
                             f"{whole} != {split}; the cached-prefix path would not match the single pass")


# ---------- the cache ----------

class _PrefixLayer(DynamicLayer):
    """One layer of a prefix computed once at batch size 1. update() returns that prefix
    expanded (as a view) across the incoming batch, followed by the new states, and stores
    nothing, so the prefix stays intact and reusable and the expanded tensors live only for
    the attention call that needs them."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        super().__init__()
        if keys.shape[0] != 1:
            raise ValueError(f"the prefix cache must hold one sequence, got batch {keys.shape[0]}")
        self.keys, self.values = keys, values
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        m = key_states.shape[0]
        keys = self.keys.to(key_states.dtype).expand(m, -1, -1, -1)
        values = self.values.to(value_states.dtype).expand(m, -1, -1, -1)
        return torch.cat([keys, key_states], dim=-2), torch.cat([values, value_states], dim=-2)


class SharedPrefixCache(DynamicCache):
    """A DynamicCache over a batch-1 prefix that serves any batch of suffixes. The prefix is
    computed once; each suffix batch reads it through expanded views. The equivalent of
    batch_repeat_interleave(M) followed by the suffix update, without M copies of the prefix
    and without keeping the concatenated states after the forward."""

    def __init__(self, prefix: DynamicCache):
        Cache.__init__(self, layers=[_PrefixLayer(layer.keys, layer.values) for layer in prefix.layers])


# ---------- the decider ----------

class Decider:
    def __init__(self, policy: HFDecoderPolicy, device: str | None = None, fast: bool = False):
        """policy: an hf-decoder policy, put in eval mode. device: where its model lives
        (defaulting to the device of its parameters). fast: run the body under bf16 autocast
        by default. Off, the body runs in fp32 and ask agrees with the training-time path to
        about 1e-6; on (CUDA only), it is faster at large question counts and agrees to about
        1e-2 at the worst probability, the bf16 noise of the training-time path itself. Every
        ask call can override it. The decision head always runs in fp32."""
        if not isinstance(policy, HFDecoderPolicy):
            raise TypeError(f"Decider needs an hf-decoder policy, got {type(policy).__name__}")
        self.policy = policy.eval()
        self.tok = policy.tok
        self.device = device or str(next(policy.model.parameters()).device)
        self.device_type = torch.device(self.device).type
        self.fast = fast
        check_split(self.tok)

    @classmethod
    def load(cls, path_or_id: str, device: str = "cuda", fast: bool = False) -> "Decider":
        """A decider over a checkpoint written by Policy.save, or over a decision-only export
        (a directory with decision.json)."""
        path_or_id = os.fspath(path_or_id)
        if os.path.isfile(os.path.join(path_or_id, "decision.json")):
            from rlcd.export import load_decision_only
            return cls(load_decision_only(path_or_id, device), device, fast)
        policy = load_policy(path_or_id, device=device)
        if not isinstance(policy, HFDecoderPolicy):
            raise TypeError(f"{path_or_id} is a {policy.name} policy; Decider supports hf-decoder only")
        return cls(policy, device, fast)

    # ----- rendering and truncation -----

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _truncate_state(self, state: str, max_state_tokens: int) -> str:
        """The state text with at most max_state_tokens tokens, cut from the left. The header
        around it is rendered afterwards, so it always survives."""
        if max_state_tokens < 1:
            raise ValueError(f"max_state_tokens must be positive, got {max_state_tokens}")
        state = state.strip()
        ids = self._encode(state)
        if len(ids) <= max_state_tokens:
            return state
        return self.tok.decode(ids[-max_state_tokens:]).strip()

    def _truncate_question(self, q: Question, max_question_tokens: int) -> Question:
        """The question with its text cut from the left until the whole suffix fits in
        max_question_tokens. The options block and the footer are never touched."""
        if max_question_tokens < 1:
            raise ValueError(f"max_question_tokens must be positive, got {max_question_tokens}")
        text = q.question.strip()
        # Cutting the question to its token budget and re-rendering can still run over by a
        # token or two where pieces merge at the seams, so the cut repeats until it fits.
        for _ in range(8):
            if len(self._encode(render_suffix(Question(q.primitive, q.context, text, q.choices, q.ordered)))) \
                    <= max_question_tokens:
                break
            fixed = len(self._encode(render_suffix(Question(q.primitive, q.context, "", q.choices, q.ordered))))
            budget = max_question_tokens - fixed
            ids = self._encode(text)
            if budget < 1 or not ids:
                raise ValueError(f"the options block of {q.question[:40]!r} alone takes {fixed} tokens, "
                                 f"which does not fit max_question_tokens={max_question_tokens}")
            keep = min(budget, len(ids) - 1)
            text = self.tok.decode(ids[-keep:]).strip()
        else:
            raise ValueError(f"could not fit {q.question[:40]!r} into max_question_tokens={max_question_tokens}")
        if text == q.question.strip():
            return q
        return Question(q.primitive, q.context, text, q.choices, q.ordered)

    def render(self, state: str, questions: list[Primitive], max_state_tokens: int,
               max_question_tokens: int) -> tuple[str, list[Question]]:
        """The prefix string of the (possibly truncated) state and the validated, (possibly
        truncated) Questions over it. Both ask paths render from exactly these."""
        state = self._truncate_state(state, max_state_tokens)
        qs = [self._truncate_question(to_question(state, q), max_question_tokens) for q in questions]
        return render_prefix(state), qs

    # ----- the cached-prefix path -----

    def _fast(self, fast: bool | None) -> bool:
        fast = self.fast if fast is None else fast
        if fast and self.device_type != "cuda":
            raise ValueError("fast (bf16 autocast) needs a CUDA device")
        return fast

    def _autocast(self, fast: bool):
        return torch.autocast(self.device_type, dtype=torch.bfloat16, enabled=fast)

    def _score_dtype(self, fast: bool) -> torch.dtype:
        """The dtype of the attention scores: bf16 under autocast, else the weights' dtype."""
        return torch.bfloat16 if fast else next(self.policy.model.parameters()).dtype

    def prefill(self, prefix: str, fast: bool | None = None) -> DynamicCache:
        """The body's key/value cache over the prefix, at batch size 1. The causal mask is
        passed explicitly, as an additive 4D mask in the dtype of the attention scores: sdpa
        adds a float mask of the query's dtype to the scores and eager attention adds it as
        well, so both implementations read it as causal (a boolean mask would be added as
        0/1 by eager, and a mask of another dtype is refused by sdpa). It is passed at all
        because with no mask the sdpa integration of transformers 4.57 hands grouped-query
        attention to torch with enable_gqa, which falls back to the unfused math kernel on
        builds without flash attention (this one); with a mask it repeats the key/value heads
        and the fused kernel runs, several times faster on a long state."""
        fast = self._fast(fast)
        ids = ([self.tok.bos_token_id] if self.policy.prepend_bos else []) + self._encode(prefix)
        ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        n = ids.shape[1]
        dtype = self._score_dtype(fast)
        blocked = ~torch.ones((1, 1, n, n), dtype=torch.bool, device=self.device).tril()
        causal = torch.zeros((1, 1, n, n), dtype=dtype, device=self.device).masked_fill(blocked, torch.finfo(dtype).min)
        with torch.no_grad(), self._autocast(fast):
            out = self.policy._body()(input_ids=ids, attention_mask=causal, use_cache=True)
        return out.past_key_values

    def _suffix_batch(self, qs: list[Question]):
        """Right-padded suffix ids, their mask, and each row's last real position."""
        seqs = [self._encode(render_suffix(q)) for q in qs]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.policy.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        last = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
            last[i] = len(s) - 1
        return ids.to(self.device), mask.to(self.device), last.to(self.device)

    def logits_from_prefix(self, prefix: DynamicCache, qs: list[Question], batch_size: int = 32,
                           fast: bool | None = None) -> torch.Tensor:
        """fp32 decision logits (M, 26), masked beyond each k, for the suffixes of qs over a
        prefilled prefix, in chunks of batch_size rows. The prefix cache is read, never changed."""
        fast = self._fast(fast)
        prefix_len = prefix.get_seq_length()
        body = self.policy._body()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(qs), batch_size):
                chunk = qs[start:start + batch_size]
                ids, mask, last = self._suffix_batch(chunk)
                m, width = ids.shape
                full_mask = torch.cat([torch.ones((m, prefix_len), dtype=mask.dtype, device=mask.device), mask], 1)
                cache_position = torch.arange(prefix_len, prefix_len + width, device=self.device)
                with self._autocast(fast):
                    hidden = body(input_ids=ids, attention_mask=full_mask, position_ids=cache_position[None],
                                  past_key_values=SharedPrefixCache(prefix), use_cache=True,
                                  cache_position=cache_position).last_hidden_state
                rows = hidden[torch.arange(m, device=hidden.device), last]
                chunks.append(self.policy.logits_from_hidden(rows, [q.k for q in chunk]))
        return torch.cat(chunks) if chunks else torch.zeros((0, 26), dtype=torch.float32)

    def probs_from_prefix(self, prefix: DynamicCache, qs: list[Question], batch_size: int = 32,
                          fast: bool | None = None) -> torch.Tensor:
        return torch.softmax(self.logits_from_prefix(prefix, qs, batch_size, fast), -1)

    def logits(self, state: str, questions: list[Primitive], max_state_tokens: int = 1536,
               max_question_tokens: int = 448, batch_size: int = 32, fast: bool | None = None) -> torch.Tensor:
        """The cached-prefix path: prefill once, one batched forward over the suffixes."""
        prefix, qs = self.render(state, questions, max_state_tokens, max_question_tokens)
        if not qs:
            return torch.zeros((0, 26), dtype=torch.float32)
        return self.logits_from_prefix(self.prefill(prefix, fast), qs, batch_size, fast)

    def probs(self, state, questions, max_state_tokens: int = 1536, max_question_tokens: int = 448,
              batch_size: int = 32, fast: bool | None = None) -> torch.Tensor:
        """fp32 probabilities (M, 26) over the letters; positions beyond k hold exactly 0."""
        return torch.softmax(self.logits(state, questions, max_state_tokens, max_question_tokens, batch_size, fast), -1)

    def ask(self, state: str, questions: list[Primitive], max_state_tokens: int = 1536,
            max_question_tokens: int = 448, batch_size: int = 32, fast: bool | None = None) -> list[dict]:
        """One typed answer per question, in order. See answers() for the fields. fast=True
        runs the body under bf16 autocast (see __init__); the default is the decider's."""
        probs = self.probs(state, questions, max_state_tokens, max_question_tokens, batch_size, fast)
        return answers(questions, probs)

    # ----- the single-pass path -----

    def logits_sequential(self, state: str, questions: list[Primitive], max_state_tokens: int = 1536,
                          max_question_tokens: int = 448, batch_size: int = 32,
                          fast: bool | None = None) -> torch.Tensor:
        """The training-time path: every full prompt rendered and run through
        Policy.decision_logits, in right-padded batches. For the equivalence proof and the
        latency comparison; it computes the prefix once per question."""
        _, qs = self.render(state, questions, max_state_tokens, max_question_tokens)
        if not qs:
            return torch.zeros((0, 26), dtype=torch.float32)
        prefix_len = len(self._encode(render_prefix(qs[0].context)))
        max_len = prefix_len + max_question_tokens + int(self.policy.prepend_bos)
        chunks = []
        with torch.no_grad(), self._autocast(self._fast(fast)):
            for start in range(0, len(qs), batch_size):
                chunk = qs[start:start + batch_size]
                logits, _ = self.policy.decision_logits(chunk, max_len, self.device)
                chunks.append(logits)
        return torch.cat(chunks)

    def ask_sequential(self, state, questions, max_state_tokens: int = 1536, max_question_tokens: int = 448,
                       batch_size: int = 32, fast: bool | None = None) -> list[dict]:
        logits = self.logits_sequential(state, questions, max_state_tokens, max_question_tokens, batch_size, fast)
        return answers(questions, torch.softmax(logits, -1))


# ---------- typed answers ----------

def answers(questions: list[Primitive], probs: torch.Tensor) -> list[dict]:
    """The typed answer dicts for probabilities (M, 26) over the letters.

    choice: kind, value (the most probable option), probs (option -> probability), confidence
    (the max probability), entropy_confidence (1 - H(p) / ln K).
    score: the choice fields plus score, the probability-weighted mean level index over K - 1.
    noul: kind, p_true, confidence = max(p_true, 1 - p_true).
    """
    out = []
    for q, row in zip(questions, probs.detach().float().cpu().numpy()):
        if isinstance(q, NoulQ):
            p_true = float(row[0])
            out.append({"kind": "noul", "p_true": p_true, "confidence": max(p_true, 1.0 - p_true)})
            continue
        labels = q.options if isinstance(q, ChoiceQ) else q.levels
        k = len(labels)
        p = row[:k].astype(np.float64)
        p = p / p.sum()
        best = int(p.argmax())
        answer = {"kind": "choice" if isinstance(q, ChoiceQ) else "score", "value": labels[best],
                  "probs": {label: float(x) for label, x in zip(labels, p)}, "confidence": float(p[best]),
                  "entropy_confidence": float(entropy_confidence(p[None], np.array([k]))[0])}
        if isinstance(q, ScoreQ):
            answer["score"] = float((p * np.arange(k)).sum() / (k - 1))
        out.append(answer)
    return out
