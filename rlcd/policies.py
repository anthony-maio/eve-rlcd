"""Backend-agnostic decision policies: one forward pass in, 26 letter logits out.

Three backends share one interface. "eve" wraps the functions in rlcd.policy unchanged, so
Eve results stay reproducible. "hf-decoder" drives any Hugging Face causal LM and reads the
letters at the last prompt token. "hf-mlm" drives a masked LM and reads them at a mask token
placed right after the prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Protocol

import torch

from rlcd.loop import save_checkpoint
from rlcd.policy import EVE_ID, decision_logits, load_eve, questions_to_batch
from rlcd.schema import MAX_CHOICES, NEG, Question, letter_token_ids, render_prompt

BACKENDS = ("eve", "hf-decoder", "hf-mlm")
POLICY_FILE = "policy.json"
LORA = dict(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear")

# Hub repos whose custom modeling code has been read and judged safe, each pinned to the exact
# commit that was read. trust_remote_code is never enabled for anything that is not listed here.
PINNED_REMOTE_CODE: dict[str, str] = {}


class Policy(Protocol):
    name: str
    model: torch.nn.Module
    tok: Any
    letters: list[int]

    def decision_logits(self, questions: list[Question], max_len: int,
                        device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 logits (B, 26), positions >= k filled with NEG, plus a scalar aux loss tensor."""

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]: ...

    def aux_coef(self) -> float: ...

    def save(self, out_dir: str, meta: dict) -> None: ...

    def train(self, mode: bool = True) -> "Policy": ...

    def eval(self) -> "Policy": ...


def remote_code_kwargs(path_or_id: str, base: str | None = None) -> dict:
    """from_pretrained kwargs that enable custom code, only for a pinned repo. A local checkpoint
    of a pinned base carries its own copy of the reviewed files, so it gets no revision."""
    path_or_id = os.fspath(path_or_id)
    if path_or_id in PINNED_REMOTE_CODE:
        return {"trust_remote_code": True, "revision": PINNED_REMOTE_CODE[path_or_id]}
    if base in PINNED_REMOTE_CODE and os.path.isdir(path_or_id):
        return {"trust_remote_code": True}
    return {}


def _mask_beyond_k(logits: torch.Tensor, questions: list[Question]) -> torch.Tensor:
    k = torch.tensor([q.k for q in questions], dtype=torch.long, device=logits.device)
    positions = torch.arange(MAX_CHOICES, device=logits.device)[None, :]
    return logits.masked_fill(positions >= k[:, None], NEG)


class _PolicyBase:
    name = ""
    model: torch.nn.Module

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    @property
    def training(self) -> bool:
        return self.model.training

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def aux_coef(self) -> float:
        return 0.0

    def _write_record(self, out_dir: str, record: dict) -> None:
        (Path(out_dir) / POLICY_FILE).write_text(json.dumps(record, indent=2))


class EvePolicy(_PolicyBase):
    name = "eve"

    def __init__(self, model, tok, letters: list[int] | None = None):
        self.model = model
        self.tok = tok
        self.letters = letter_token_ids(tok) if letters is None else list(letters)

    def decision_logits(self, questions, max_len, device):
        ids, last, k = questions_to_batch(self.tok, questions, max_len, device)
        return decision_logits(self.model, ids, last, self.letters, k)

    def aux_coef(self) -> float:
        return self.model.config.router_aux_loss_coef

    def save(self, out_dir: str, meta: dict) -> None:
        save_checkpoint(self.model, self.tok, out_dir, meta)
        self._write_record(out_dir, {"backend": self.name})


def _adds_bos(tok) -> bool:
    """True when the tokenizer prepends its BOS token by default. Such models never saw a
    sequence without it, so the policy keeps it in front of the (possibly truncated) prompt."""
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return False
    ids = tok.encode("x", add_special_tokens=True)
    return len(ids) > 0 and ids[0] == bos


class _HFPolicy(_PolicyBase):
    """Shared by the two Hugging Face backends: right padding with an explicit attention mask,
    left truncation that keeps the tail, and an fp32 readout at one position per row."""

    def __init__(self, model, tok, letters: list[int] | None = None, base_id: str | None = None,
                 lora: bool = False):
        self.model = model
        self.tok = tok
        self.letters = letter_token_ids(tok) if letters is None else list(letters)
        self.base_id = base_id
        self.lora = lora
        self.prepend_bos = _adds_bos(tok)
        pad = getattr(tok, "pad_token_id", None)
        self.pad_id = pad if pad is not None else getattr(tok, "eos_token_id", None)
        if self.pad_id is None:
            raise ValueError("the tokenizer has neither a pad token nor an eos token to pad with")

    def _suffix(self) -> list[int]:
        return []

    def _hf_model(self):
        """The transformers model, from under the peft wrapper when there is one."""
        return self.model.get_base_model() if self.lora else self.model

    def encode(self, questions: list[Question], max_len: int, device: str):
        """Right-padded ids, the attention mask, and each row's readout position: the last real
        token. The prompt is truncated from the left; BOS and the suffix always survive."""
        prefix = [self.tok.bos_token_id] if self.prepend_bos else []
        suffix = self._suffix()
        room = max_len - len(prefix) - len(suffix)
        if room <= 0:
            raise ValueError(f"max_len {max_len} leaves no room for the prompt")
        seqs = [prefix + self.tok.encode(render_prompt(q), add_special_tokens=False)[-room:] + suffix
                for q in questions]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        last = torch.zeros(len(seqs), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
            last[i] = len(s) - 1
        return ids.to(device), mask.to(device), last.to(device)

    def _readout_rows(self, questions, max_len, device) -> torch.Tensor:
        """The body's final hidden state at each row's readout position, (B, d)."""
        ids, mask, last = self.encode(questions, max_len, device)
        hidden = self._body()(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        return hidden[torch.arange(hidden.size(0), device=hidden.device), last]

    def _body(self):
        raise NotImplementedError

    def save(self, out_dir: str, meta: dict) -> None:
        """Full fine-tune: the whole model. LoRA: the adapter only, with policy.json naming the
        base it belongs to. The tokenizer goes along either way."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out, safe_serialization=True)
        self.tok.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        self._write_record(out_dir, {"backend": self.name, "lora": self.lora, "base": self.base_id,
                                     "prepend_bos": self.prepend_bos})


class HFDecoderPolicy(_HFPolicy):
    name = "hf-decoder"

    def _body(self):
        model = self._hf_model()
        get = getattr(model, "get_decoder", None)
        body = get() if callable(get) else None
        if body is None or body is model:
            body = getattr(model, "model", None)
        if body is None or body is model:
            body = model.base_model
        if body is model:
            raise ValueError(f"cannot find the decoder body of {type(model).__name__}")
        return body

    def decision_logits(self, questions, max_len, device):
        """Runs the body only and multiplies the decision rows by the 26 letter rows of the
        output embedding, so full-vocabulary logits are never materialized."""
        rows = self._readout_rows(questions, max_len, device)
        head = self._hf_model().get_output_embeddings()
        # Autocast off: it would downcast the matmul and quantize the decision logits.
        with torch.autocast(device_type=rows.device.type, enabled=False):
            index = torch.as_tensor(self.letters, device=rows.device)
            logits = rows.float() @ head.weight[index].float().t()
            if getattr(head, "bias", None) is not None:
                logits = logits + head.bias[index].float()
            aux = torch.zeros((), device=rows.device, dtype=torch.float32)
            return _mask_beyond_k(logits, questions), aux


def _mlm_head(model):
    """The module chain that maps hidden states to vocabulary logits in a masked LM."""
    for names in (("cls",), ("lm_head",), ("head", "decoder")):
        if all(isinstance(getattr(model, n, None), torch.nn.Module) for n in names):
            return [getattr(model, n) for n in names]
    raise ValueError(f"cannot find the masked-LM head of {type(model).__name__}")


class HFMaskedLMPolicy(_HFPolicy):
    """The same prompt followed by one mask token; the letters are read at the mask."""
    name = "hf-mlm"

    def __init__(self, model, tok, letters=None, base_id=None, lora=False):
        if getattr(tok, "mask_token_id", None) is None:
            raise ValueError("the hf-mlm backend needs a tokenizer with a mask token")
        super().__init__(model, tok, letters, base_id, lora)

    def _suffix(self) -> list[int]:
        return [self.tok.mask_token_id]

    def _body(self):
        return self._hf_model().base_model

    def decision_logits(self, questions, max_len, device):
        """A masked-LM head is more than a matmul (dense, activation, norm, then the decoder),
        so it runs whole, but only on the B mask rows, in fp32."""
        rows = self._readout_rows(questions, max_len, device)
        with torch.autocast(device_type=rows.device.type, enabled=False):
            x = rows.float()[:, None, :]
            for module in _mlm_head(self._hf_model()):
                x = module(x)
            index = torch.as_tensor(self.letters, device=rows.device)
            logits = x[:, 0, :].float()[:, index]
            aux = torch.zeros((), device=rows.device, dtype=torch.float32)
            return _mask_beyond_k(logits, questions), aux


def _read_record(path_or_id: str) -> dict:
    path = os.path.join(path_or_id, POLICY_FILE)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def resolve_backend(path_or_id: str, backend: str = "auto") -> str:
    """auto: what a local policy.json records, else eve for the Eve hub id or an Eve
    config.json, else hf-decoder."""
    path_or_id = os.fspath(path_or_id)
    if backend != "auto":
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected auto or one of {BACKENDS}")
        return backend
    record = _read_record(path_or_id)
    if record:
        return resolve_backend(path_or_id, record["backend"])
    if path_or_id == EVE_ID:
        return "eve"
    config = os.path.join(path_or_id, "config.json")
    if os.path.isfile(config):
        with open(config, encoding="utf-8") as f:
            if json.load(f).get("model_type") == "eve-moe":
                return "eve"
    return "hf-decoder"


def _load_tokenizer(path_or_id: str, **kwargs):
    """AutoTokenizer, falling back to PreTrainedTokenizerFast for repos saved by transformers 5,
    whose tokenizer_config.json names a class ("TokenizersBackend") that 4.x does not have.
    Such a repo ships a self-contained tokenizer.json, which the fast class reads as is."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    try:
        return AutoTokenizer.from_pretrained(path_or_id, **kwargs)
    except ValueError as e:
        if "Tokenizer class" not in str(e):
            raise
        kwargs.pop("trust_remote_code", None)
        print(f"note: {e} Falling back to PreTrainedTokenizerFast on tokenizer.json.")
        return PreTrainedTokenizerFast.from_pretrained(path_or_id, **kwargs)


def _check_rope(config) -> None:
    """A config.json written by transformers 5 keeps rope_theta inside rope_parameters, where
    4.x does not look. Refuse to run on a silently defaulted value."""
    theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
    if theta is not None and getattr(config, "rope_theta", theta) != theta:
        raise RuntimeError(f"config.rope_theta {config.rope_theta} differs from "
                           f"rope_parameters.rope_theta {theta}")


def _load_hf(policy_cls, auto_cls, path_or_id: str, device: str, lora: bool, grad_checkpointing: bool):
    record = _read_record(path_or_id)
    saved_adapter = bool(record.get("lora"))
    base_id = record.get("base") or path_or_id
    weights = base_id if saved_adapter else path_or_id
    tok = _load_tokenizer(path_or_id, **remote_code_kwargs(path_or_id, base_id))
    model = auto_cls.from_pretrained(weights, torch_dtype=torch.float32,
                                     **remote_code_kwargs(weights, base_id))
    _check_rope(model.config)
    if grad_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    if saved_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path_or_id, is_trainable=True)
    elif lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(**LORA))
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("master weights must stay fp32")
    model.to(device)
    return policy_cls(model, tok, base_id=base_id, lora=saved_adapter or lora)


def load_policy(path_or_id: str, device: str = "cuda", backend: str = "auto", lora: bool = False,
                grad_checkpointing: bool = False) -> Policy:
    """A policy from a hub id or a checkpoint directory written by Policy.save, fp32 on device
    and in train mode; call .eval() for inference."""
    path_or_id = os.fspath(path_or_id)
    backend = resolve_backend(path_or_id, backend)
    if backend == "eve":
        if lora or grad_checkpointing:
            raise ValueError("the eve backend supports neither LoRA nor gradient checkpointing")
        model, tok = load_eve(path_or_id, device=device)
        return EvePolicy(model, tok).train()
    if backend == "hf-decoder":
        from transformers import AutoModelForCausalLM
        return _load_hf(HFDecoderPolicy, AutoModelForCausalLM, path_or_id, device, lora,
                        grad_checkpointing).train()
    from transformers import AutoModelForMaskedLM
    return _load_hf(HFMaskedLMPolicy, AutoModelForMaskedLM, path_or_id, device, lora,
                    grad_checkpointing).train()
