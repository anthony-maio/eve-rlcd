# eve-rlcd Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train Eve-2-MoE-IT-272M into a decision-only model with a faithful RLCD loop (bandit feedback, reward `c - p_a`), and show with an RLVR and supervised-oracle ablation that the calibration reward produces calibrated probabilities.

**Architecture:** One forward pass per question, logits at the final position sliced to the 26 letter tokens, softmax over the declared choices. That categorical is the policy during RL and the API output at inference. Plain PyTorch training loops (no TRL), a bandit environment that reveals only the sampled action's correctness, and an eval harness that produces reliability diagrams and coverage-error curves for each training arm. A final step replaces the LM head with the 26 decision rows so the saved model cannot emit text.

**Tech Stack:** Python 3.12 via uv, PyTorch (CUDA 12.8 wheels), transformers 4.x, datasets, safetensors, numpy, matplotlib, pytest. Windows, single RTX 4080 16 GB.

**Spec:** `docs/plans/2026-09-17-eve-rlcd-design.md`

## Global Constraints

- Base model is `anthonym21/Eve-2-MoE-IT-272M`. Its code is vendored under `rlcd/eve/` (MIT, the author's own). Never load it with `trust_remote_code`; always use the vendored class so the policy adapter can walk `model.transformer.h`.
- Every Eve config is built or loaded through `rlcd.compat.eve_config(**kw)` or `rlcd.compat.load_eve_config(path_or_id)`. Never call `eve_config(...)` or `EveConfig.from_pretrained` directly, and never call `EveMoEForCausalLM.from_pretrained` without `config=load_eve_config(...)`. Reason: EveConfig's MoE routing field `top_k` collides with a legacy generation default in transformers 4.x, which silently overwrites it with 50.
- The HF wrapper does not inherit `GenerationMixin` on transformers >= 4.50, so `model.generate()` is unavailable. Nothing in this project needs it; use an explicit greedy loop where text generation is wanted for a sanity check.
- The ASCII rule exempts the vendored files under `rlcd/eve/`, which stay byte-identical to the hub.
- Prompt format is exactly the one in the spec, ending in `Assistant: The answer is`. The decision token is the next token, one of `" A"` .. `" Z"`.
- Cardinality is 2 to 26 inclusive. `noul` questions have choices exactly `["true", "false"]`. `score` questions must have `ordered: true`.
- Batching uses right padding with pad id 50256. Logits are gathered at each row's last real token. Never left-pad (Eve ignores attention masks and has no position ids).
- Masked (out-of-range) letter logits use the finite constant `NEG = -1e4`, never `-inf`, so KL and log-prob math never produce NaN.
- Training keeps fp32 master weights and runs the forward under `torch.autocast("cuda", dtype=torch.bfloat16)`. No LoRA. Never call `.to(torch.bfloat16)` or `.half()` on an Eve model: its RoPE buffer `freqs_cis` is complex64 and the cast corrupts it. Eve's router auxiliary loss is added with `config.router_aux_loss_coef` (0.01) to every training objective.
- RL feedback is bandit-only: the training loop for the `rlvr` and `rlcd` arms may call only `BanditEnv.step`. Only the `oracle` arm may call `BanditEnv.reveal`.
- The RLCD reward is `outcomes - p_a` where `p_a` is the detached probability of the sampled action. The RLVR reward is `outcomes`.
- No em dashes, no emojis, no non-ASCII punctuation in any file. Arrows are `->`.
- Commit messages never carry co-author or AI attribution lines.
- Generated data goes in `data/`, checkpoints and plots in `runs/`. Both are gitignored.
- Run everything through `uv run` so the project venv is used.

---

## File Structure

```
eve-rlcd/
  pyproject.toml              project metadata, deps, pytorch cu128 index, pytest markers
  README.md                   what this is, how to reproduce, results table (Task 11)
  rlcd/__init__.py
  rlcd/eve/__init__.py
  rlcd/eve/configuration_eve.py   vendored from the HF repo, unchanged
  rlcd/eve/modeling_eve.py        vendored from the HF repo, unchanged
  rlcd/schema.py              Question dataclass, validation, prompt rendering, letter token ids, NOTA constant
  rlcd/policy.py              load_eve, encode_batch, decision_logits (the sliced-softmax policy)
  rlcd/env.py                 BanditEnv
  rlcd/rewards.py             reward functions, policy gradient loss, KL, supervised loss
  rlcd/metrics.py             brier, ece, reliability bins, coverage-error curve, nota rate
  rlcd/data.py                dataset converters, choice subsetting, NOTA injection, synthetic triage, build CLI
  rlcd/loop.py                shared training utilities: cosine lr, batch iterator, checkpoint save/load, jsonl logger
  rlcd/train_sft.py           warmup SFT CLI
  rlcd/train_rl.py            RL CLI with arms rlvr | rlcd | oracle
  rlcd/eval.py                predictions, metrics, plots, compare CLI
  rlcd/calibrate.py           temperature fitting on validation logits
  rlcd/amputate.py            build the decision-only checkpoint
  rlcd/infer.py               DecisionModel with choice / score / noul / ask
  tests/test_schema.py
  tests/test_policy.py
  tests/test_rewards.py
  tests/test_metrics.py
  tests/test_data.py
  tests/test_loop.py
  tests/test_infer.py
  scripts/smoke_eve.py        GPU sanity check: real model loads, tie is correct, zero-shot letters work
```

---

### Task 1: Project scaffold and environment

**Files:**
- Create: `pyproject.toml`
- Create: `rlcd/__init__.py`, `rlcd/eve/__init__.py`
- Create: `rlcd/eve/configuration_eve.py`, `rlcd/eve/modeling_eve.py` (downloaded)
- Create: `rlcd/compat.py` (`eve_config(**kw)`, `load_eve_config(path_or_id)`; added during execution after the `top_k` collision was found)
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Produces: `rlcd.compat.eve_config`, `rlcd.compat.load_eve_config`, importable package `rlcd`, vendored `rlcd.eve.configuration_eve.EveConfig` and `rlcd.eve.modeling_eve.EveMoEForCausalLM`.

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "eve-rlcd"
version = "0.1.0"
description = "Toy recreation of RLCD (RL for Calibrated Decisions) on Eve-2-MoE-IT-272M"
requires-python = ">=3.12,<3.13"
dependencies = [
    "torch>=2.7",
    "transformers>=4.51,<5",
    "datasets>=3.2",
    "safetensors>=0.5",
    "huggingface-hub>=0.30",
    "numpy>=2",
    "matplotlib>=3.9",
    "tqdm>=4.66",
]

[dependency-groups]
dev = ["pytest>=8"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "network: needs internet access",
    "gpu: needs a CUDA device",
]
addopts = "-m 'not network and not gpu'"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rlcd"]
```

- [ ] **Step 2: Create the environment**

Run from `D:\hermes\eve-rlcd`:
```powershell
uv python pin 3.12
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: prints a 2.7+ version, `True`, and `NVIDIA GeForce RTX 4080`. If `cuda.is_available()` is False, the cu128 index was not used; run `uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128` and re-check.

- [ ] **Step 3: Vendor the Eve model code**

```powershell
New-Item -ItemType Directory -Force rlcd, rlcd\eve, tests, scripts | Out-Null
New-Item -ItemType File rlcd\__init__.py, rlcd\eve\__init__.py | Out-Null
curl.exe -sL https://huggingface.co/anthonym21/Eve-2-MoE-IT-272M/raw/main/configuration_eve.py -o rlcd\eve\configuration_eve.py
curl.exe -sL https://huggingface.co/anthonym21/Eve-2-MoE-IT-272M/raw/main/modeling_eve.py -o rlcd\eve\modeling_eve.py
```
`modeling_eve.py` already contains `from .configuration_eve import EveConfig` with a non-relative fallback, so it works as a package module without edits. Do not modify either file.

- [ ] **Step 4: Write the scaffold test**

`tests/test_scaffold.py`:
```python
import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM


def tiny_config():
    return eve_config(
        vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16,
        block_size=128, num_experts=2, top_k=1,
        expert_intermediate_size=64, shared_expert_intermediate_size=64,
    )


def test_tiny_eve_forward_shape():
    torch.manual_seed(0)
    model = EveMoEForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, 50304, (2, 7))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 7, 50304)


def test_tiny_eve_is_tied():
    model = EveMoEForCausalLM(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()
```

- [ ] **Step 5: Run the test**

Run: `uv run pytest tests/test_scaffold.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml uv.lock .python-version rlcd tests
git commit -m "Scaffold project and vendor Eve-2 model code"
```

---

### Task 2: Question schema and prompt rendering

**Files:**
- Create: `rlcd/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `LETTERS: list[str]` (A..Z), `MAX_CHOICES = 26`, `NOTA = "None of the above"`, `NEG = -1e4`
  - `@dataclass Question(primitive: str, context: str, question: str, choices: list[str], ordered: bool = False, answer: int | None = None, source: str = "", id: str = "")` with `.validate() -> Question`, `.to_json() -> str`, `Question.from_json(s) -> Question`, `.k -> int`
  - `render_prompt(q: Question) -> str`
  - `letter_token_ids(tokenizer) -> list[int]` (length 26)
  - `read_jsonl(path) -> list[Question]`, `write_jsonl(path, questions) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_schema.py`:
```python
import pytest

from rlcd.schema import (LETTERS, MAX_CHOICES, NOTA, Question, letter_token_ids,
                         read_jsonl, render_prompt, write_jsonl)


def q(**kw) -> Question:
    base = dict(primitive="choice", context="ctx", question="Which?", choices=["a", "b", "c"], answer=1)
    base.update(kw)
    return Question(**base)


def test_letters():
    assert LETTERS[0] == "A" and LETTERS[-1] == "Z" and len(LETTERS) == MAX_CHOICES == 26


def test_render_prompt_exact():
    text = render_prompt(q(context="  Ticket: printer on fire  ", question="Which department?"))
    assert text == (
        "User: Context:\n"
        "Ticket: printer on fire\n"
        "\n"
        "Question: Which department?\n"
        "Options:\n"
        "A) a\n"
        "B) b\n"
        "C) c\n"
        "Answer with the letter only.\n"
        "Assistant: The answer is"
    )


def test_validate_rejects_bad_cardinality():
    with pytest.raises(ValueError):
        q(choices=["only"]).validate()
    with pytest.raises(ValueError):
        q(choices=[str(i) for i in range(27)], answer=0).validate()


def test_validate_rejects_duplicates():
    with pytest.raises(ValueError):
        q(choices=["a", "a", "b"]).validate()


def test_validate_noul_choices():
    q(primitive="noul", choices=["true", "false"], answer=0).validate()
    with pytest.raises(ValueError):
        q(primitive="noul", choices=["yes", "no"], answer=0).validate()


def test_validate_score_needs_ordered():
    with pytest.raises(ValueError):
        q(primitive="score", ordered=False).validate()
    q(primitive="score", ordered=True).validate()


def test_validate_answer_range():
    with pytest.raises(ValueError):
        q(answer=3).validate()


def test_json_roundtrip():
    original = q(source="unit", id="x1")
    assert Question.from_json(original.to_json()) == original


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "qs.jsonl"
    items = [q(id="1"), q(id="2", choices=["x", "y"], answer=0)]
    write_jsonl(path, items)
    assert read_jsonl(path) == items


def test_k():
    assert q().k == 3


def test_letter_token_ids_gpt2():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("gpt2")
    ids = letter_token_ids(tok)
    assert ids[:4] == [317, 347, 327, 360]
    assert len(set(ids)) == 26


def test_nota_constant():
    assert NOTA == "None of the above"
```

Note: `test_letter_token_ids_gpt2` downloads the gpt2 tokenizer (a few hundred KB) once. Mark it `@pytest.mark.network` if the machine is offline; the expected ids come from Eve's `tokenizer.json`, which is the GPT-2 vocabulary.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schema.py -v`
Expected: ImportError on `rlcd.schema`.

- [ ] **Step 3: Implement schema.py**

```python
"""Question types, validation, prompt rendering, and letter token ids."""
from __future__ import annotations

import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path

LETTERS: list[str] = list(string.ascii_uppercase)
MAX_CHOICES = 26
NOTA = "None of the above"
NEG = -1e4  # finite mask value for out-of-range letters
PRIMITIVES = ("choice", "score", "noul")


@dataclass
class Question:
    primitive: str
    context: str
    question: str
    choices: list[str]
    ordered: bool = False
    answer: int | None = None
    source: str = ""
    id: str = ""

    @property
    def k(self) -> int:
        return len(self.choices)

    def validate(self) -> "Question":
        if self.primitive not in PRIMITIVES:
            raise ValueError(f"unknown primitive {self.primitive!r}")
        n = len(self.choices)
        if n < 2 or n > MAX_CHOICES:
            raise ValueError(f"need 2..{MAX_CHOICES} choices, got {n}")
        if len(set(self.choices)) != n:
            raise ValueError("duplicate choices")
        if self.primitive == "noul" and self.choices != ["true", "false"]:
            raise ValueError('noul choices must be exactly ["true", "false"]')
        if self.primitive == "score" and not self.ordered:
            raise ValueError("score questions must set ordered=True")
        if self.answer is not None and not (0 <= self.answer < n):
            raise ValueError(f"answer {self.answer} out of range for {n} choices")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Question":
        return cls(**json.loads(s)).validate()


def render_prompt(q: Question) -> str:
    lines = [
        "User: Context:",
        q.context.strip(),
        "",
        f"Question: {q.question.strip()}",
        "Options:",
    ]
    for letter, choice in zip(LETTERS, q.choices):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer with the letter only.")
    lines.append("Assistant: The answer is")
    return "\n".join(lines)


def letter_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for letter in LETTERS:
        toks = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"' {letter}' is not a single token: {toks}")
        ids.append(toks[0])
    if len(set(ids)) != MAX_CHOICES:
        raise ValueError("letter token ids collide")
    return ids


def read_jsonl(path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question.from_json(line) for line in f if line.strip()]


def write_jsonl(path, questions: list[Question]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.validate().to_json() + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schema.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```powershell
git add rlcd/schema.py tests/test_schema.py
git commit -m "Add question schema and prompt rendering"
```

---

### Task 3: Policy adapter (sliced softmax over Eve)

**Files:**
- Create: `rlcd/policy.py`
- Create: `scripts/smoke_eve.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `rlcd.schema.NEG`, `letter_token_ids`, `render_prompt`, vendored Eve classes.
- Produces:
  - `EVE_ID = "anthonym21/Eve-2-MoE-IT-272M"`, `PAD_ID = 50256`
  - `load_eve(path_or_id: str = EVE_ID, device: str = "cuda") -> tuple[EveMoEForCausalLM, tokenizer]` (fp32 weights on device, tie verified and repaired from the checkpoint file)
  - `encode_batch(tokenizer, prompts: list[str], max_len: int = 512, device: str = "cpu") -> tuple[LongTensor (B,T), LongTensor (B,)]` (right padded, last real index)
  - `decision_logits(model, input_ids, last_idx, letter_ids: list[int], k: LongTensor (B,)) -> tuple[FloatTensor (B,26), Tensor scalar aux_loss]` (fp32 logits, positions >= k filled with NEG)
  - `questions_to_batch(tokenizer, questions, max_len, device) -> tuple[input_ids, last_idx, k]`

- [ ] **Step 1: Write the failing tests**

`tests/test_policy.py`:
```python
import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policy import PAD_ID, decision_logits, encode_batch
from rlcd.schema import NEG


class FakeTok:
    """Maps each character to a token id; enough to test padding logic."""
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny_model() -> EveMoEForCausalLM:
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16,
                    block_size=128, num_experts=2, top_k=1,
                    expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg).eval()


def test_encode_batch_right_pads_and_tracks_last_index():
    ids, last = encode_batch(FakeTok(), ["abc", "abcdef"], max_len=512)
    assert ids.shape == (2, 6)
    assert ids[0, 3:].tolist() == [PAD_ID] * 3
    assert last.tolist() == [2, 5]


def test_encode_batch_truncates_from_the_left():
    ids, last = encode_batch(FakeTok(), ["abcdefgh"], max_len=4)
    assert ids.shape == (1, 4)
    assert ids[0].tolist() == FakeTok().encode("efgh")
    assert last.tolist() == [3]


def test_padded_batch_matches_single_rows():
    model = tiny_model()
    tok = FakeTok()
    letter_ids = list(range(100, 126))
    prompts = ["short one", "a considerably longer prompt here"]
    ids, last = encode_batch(tok, prompts)
    k = torch.tensor([3, 5])
    with torch.no_grad():
        batched, _ = decision_logits(model, ids, last, letter_ids, k)
        for i, p in enumerate(prompts):
            ids1, last1 = encode_batch(tok, [p])
            single, _ = decision_logits(model, ids1, last1, letter_ids, k[i:i + 1])
            assert torch.allclose(batched[i, : k[i]], single[0, : k[i]], atol=1e-4)


def test_decision_logits_masks_beyond_k():
    model = tiny_model()
    ids, last = encode_batch(FakeTok(), ["xyz"])
    k = torch.tensor([2])
    with torch.no_grad():
        logits, aux = decision_logits(model, ids, last, list(range(100, 126)), k)
    assert logits.shape == (1, 26)
    assert torch.all(logits[0, 2:] == NEG)
    assert torch.isfinite(logits).all()
    assert torch.softmax(logits, -1)[0, 2:].sum() == 0
    assert aux.ndim == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: ImportError on `rlcd.policy`.

- [ ] **Step 3: Implement policy.py**

```python
"""The sliced-softmax policy over Eve-2: one forward pass, 26 letter logits."""
from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rlcd.compat import load_eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.schema import MAX_CHOICES, NEG, Question, render_prompt

EVE_ID = "anthonym21/Eve-2-MoE-IT-272M"
PAD_ID = 50256


def _checkpoint_file(path_or_id: str) -> str:
    if os.path.isdir(path_or_id):
        return os.path.join(path_or_id, "model.safetensors")
    return hf_hub_download(path_or_id, "model.safetensors")


def _repair_tie(model: EveMoEForCausalLM, path_or_id: str) -> None:
    """Eve ties wte and lm_head. Checkpoints store only one of them, so copy the
    stored tensor into wte and re-point lm_head at it. Idempotent."""
    sd = load_file(_checkpoint_file(path_or_id))
    ref = sd.get("lm_head.weight", sd.get("transformer.wte.weight"))
    if ref is None:
        raise RuntimeError("checkpoint has neither lm_head.weight nor transformer.wte.weight")
    with torch.no_grad():
        model.transformer.wte.weight.copy_(ref.to(model.transformer.wte.weight.dtype))
    model.lm_head.weight = model.transformer.wte.weight
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise RuntimeError("failed to tie lm_head to wte")


def load_eve(path_or_id: str = EVE_ID, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(path_or_id)
    model = EveMoEForCausalLM.from_pretrained(path_or_id, config=load_eve_config(path_or_id),
                                              torch_dtype=torch.float32)
    if model.config.top_k != model.transformer.h[0].mlp.top_k:
        raise RuntimeError("MoE routing top_k mismatch between config and built model")
    _repair_tie(model, path_or_id)
    model.to(device)
    return model, tokenizer


def encode_batch(tokenizer, prompts: list[str], max_len: int = 512, device: str = "cpu"):
    seqs = [tokenizer.encode(p, add_special_tokens=False)[-max_len:] for p in prompts]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    last = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        last[i] = len(s) - 1
    return ids.to(device), last.to(device)


def questions_to_batch(tokenizer, questions: list[Question], max_len: int = 512, device: str = "cpu"):
    ids, last = encode_batch(tokenizer, [render_prompt(q) for q in questions], max_len, device)
    k = torch.tensor([q.k for q in questions], dtype=torch.long, device=device)
    return ids, last, k


def eve_hidden(model: EveMoEForCausalLM, input_ids: torch.Tensor):
    """Final hidden states (B,T,d) and the summed router aux loss."""
    x = model.transformer.wte(input_ids)
    aux = torch.zeros((), device=x.device, dtype=torch.float32)
    for block in model.transformer.h:
        x, block_aux = block(x, model.freqs_cis)
        aux = aux + block_aux.float()
    return model.transformer.ln_f(x), aux


def decision_logits(model, input_ids, last_idx, letter_ids: list[int], k: torch.Tensor):
    """fp32 logits over the 26 letters at each row's decision position.
    Positions at or beyond k are set to NEG so their softmax mass is exactly zero."""
    hidden, aux = eve_hidden(model, input_ids)
    rows = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
    weight = model.lm_head.weight[torch.as_tensor(letter_ids, device=hidden.device)]
    logits = rows.float() @ weight.float().t()
    positions = torch.arange(MAX_CHOICES, device=logits.device)[None, :]
    mask = positions >= k.to(logits.device)[:, None]
    return logits.masked_fill(mask, NEG), aux
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the GPU smoke script**

`scripts/smoke_eve.py`:
```python
"""Load the real Eve-2-IT, verify the tie, and check zero-shot letter behavior."""
import torch

from rlcd.policy import decision_logits, load_eve, questions_to_batch
from rlcd.schema import Question, letter_token_ids

model, tok = load_eve(device="cuda")
model.eval()
letters = letter_token_ids(tok)

# 1. Plain generation still works (proves weights loaded sensibly).
# The HF wrapper has no GenerationMixin on current transformers, so decode greedily by hand.
ids = tok.encode("User: What is the capital of France?\nAssistant:", return_tensors="pt").cuda()
start = ids.shape[1]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(20):
        nxt = model(input_ids=ids).logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
print("GEN:", repr(tok.decode(ids[0][start:])))

# 2. Zero-shot decisions on three obvious questions.
qs = [
    Question("choice", "The customer writes: my card was charged twice for one order.",
             "What is the customer's intent?", ["cancel order", "refund request", "track shipment", "change password"], answer=1),
    Question("noul", "Paris is the capital of France.", "Is the statement true?", ["true", "false"], answer=0),
    Question("score", "This movie was an absolute masterpiece, I cried.", "What is the sentiment?",
             ["very negative", "negative", "neutral", "positive", "very positive"], ordered=True, answer=4),
]
ids, last, k = questions_to_batch(tok, qs, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits, aux = decision_logits(model, ids, last, letters, k)
probs = torch.softmax(logits, -1)
for q, p in zip(qs, probs):
    top = p[: q.k].tolist()
    print(q.primitive, "answer", q.answer, "probs", [round(x, 3) for x in top])
print("aux", float(aux))
print("VRAM MB", torch.cuda.max_memory_allocated() // 2**20)
```

- [ ] **Step 6: Run the smoke script**

Run: `uv run python scripts/smoke_eve.py`
Expected: `GEN:` prints coherent English (any sentence about Paris or France). Probabilities print for all three questions and sum to about 1 over the first k entries. Zero-shot accuracy may be poor; that is fine. If `GEN:` prints garbage, the tie repair failed: open `_repair_tie` and confirm `model.safetensors` from the hub contains `lm_head.weight` (it does as of 2026-09-17) and that the copy ran on the same dtype.

- [ ] **Step 7: Commit**

```powershell
git add rlcd/policy.py tests/test_policy.py scripts/smoke_eve.py
git commit -m "Add sliced-softmax policy adapter over Eve-2"
```

---

### Task 4: Bandit environment and reward objectives

**Files:**
- Create: `rlcd/env.py`, `rlcd/rewards.py`
- Test: `tests/test_rewards.py`

**Interfaces:**
- Consumes: nothing beyond torch.
- Produces:
  - `BanditEnv(answers: LongTensor)` with `.step(idx: LongTensor (B,), actions: LongTensor (B,G)) -> FloatTensor (B,G)` and `.reveal(idx) -> LongTensor (B,)` and `len(env)`
  - `reward_rlvr(outcomes, p_a) -> Tensor`, `reward_rlcd(outcomes, p_a) -> Tensor`, `REWARDS: dict[str, callable]`
  - `policy_gradient_loss(logp: (B,26), actions: (B,G), rewards: (B,G)) -> scalar` (leave-one-out baseline)
  - `kl_categorical(logp, logp_ref) -> Tensor (B,)`
  - `supervised_loss(logp, answers) -> scalar`

- [ ] **Step 1: Write the failing tests**

`tests/test_rewards.py`:
```python
import torch
import torch.nn.functional as F

from rlcd.env import BanditEnv
from rlcd.rewards import (REWARDS, kl_categorical, policy_gradient_loss, reward_rlcd,
                          reward_rlvr, supervised_loss)
from rlcd.schema import NEG


def test_env_step_reveals_only_correctness():
    env = BanditEnv(torch.tensor([2, 0, 1]))
    idx = torch.tensor([0, 2])
    actions = torch.tensor([[2, 1], [1, 1]])
    out = env.step(idx, actions)
    assert out.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert len(env) == 3
    assert env.reveal(idx).tolist() == [2, 1]


def test_reward_functions():
    outcomes = torch.tensor([[1.0, 0.0]])
    p_a = torch.tensor([[0.9, 0.9]])
    assert reward_rlvr(outcomes, p_a).tolist() == [[1.0, 0.0]]
    assert torch.allclose(reward_rlcd(outcomes, p_a), torch.tensor([[0.1, -0.9]]))
    assert set(REWARDS) == {"rlvr", "rlcd"}


def _exact_reinforce_grad(theta, y, reward_fn):
    """Enumerate every action: E_a[ r(a) grad log p_a ]."""
    p = torch.softmax(theta, 0)
    logp = torch.log_softmax(theta, 0)
    pd = p.detach()
    surrogate = torch.zeros(())
    for a in range(theta.numel()):
        c = torch.tensor(float(a == y))
        r = reward_fn(c[None, None], pd[a][None, None])[0, 0]
        surrogate = surrogate + pd[a] * r * logp[a]
    (g,) = torch.autograd.grad(surrogate, theta)
    return g


def test_rlcd_reinforce_is_unbiased_for_brier_gradient():
    torch.manual_seed(0)
    theta = torch.randn(5, requires_grad=True)
    y = 2
    p = torch.softmax(theta, 0)
    brier = -((p - F.one_hot(torch.tensor(y), 5).float()) ** 2).sum()
    (g_brier,) = torch.autograd.grad(brier, theta)
    g_reinforce = _exact_reinforce_grad(theta, y, reward_rlcd)
    assert torch.allclose(2 * g_reinforce, g_brier, atol=1e-6)


def test_rlvr_reinforce_is_accuracy_gradient():
    torch.manual_seed(1)
    theta = torch.randn(5, requires_grad=True)
    y = 3
    p = torch.softmax(theta, 0)
    (g_acc,) = torch.autograd.grad(p[y], theta)
    g_reinforce = _exact_reinforce_grad(theta, y, reward_rlvr)
    assert torch.allclose(g_reinforce, g_acc, atol=1e-6)


def test_policy_gradient_loss_leave_one_out_baseline():
    logp = torch.log_softmax(torch.zeros(1, 26), -1).requires_grad_(True)
    actions = torch.tensor([[0, 1, 2]])
    rewards = torch.tensor([[1.0, 0.0, 0.0]])
    loss = policy_gradient_loss(logp, actions, rewards)
    # baselines: for a0 mean(0,0)=0 -> adv 1; for a1 mean(1,0)=.5 -> adv -.5; a2 -> -.5
    expected = -(1.0 * logp[0, 0] - 0.5 * logp[0, 1] - 0.5 * logp[0, 2]) / 3
    assert torch.allclose(loss, expected)


def test_policy_gradient_loss_single_sample_has_no_baseline():
    logp = torch.log_softmax(torch.zeros(2, 26), -1)
    loss = policy_gradient_loss(logp, torch.tensor([[0], [1]]), torch.tensor([[1.0], [-1.0]]))
    assert torch.allclose(loss, torch.tensor(0.0))


def test_kl_is_zero_for_identical_and_finite_with_masking():
    logits = torch.tensor([[1.0, 2.0, NEG, NEG]])
    logp = torch.log_softmax(logits, -1)
    assert torch.allclose(kl_categorical(logp, logp), torch.zeros(1))
    other = torch.log_softmax(torch.tensor([[2.0, 1.0, NEG, NEG]]), -1)
    kl = kl_categorical(logp, other)
    assert torch.isfinite(kl).all() and kl.item() > 0


def test_supervised_loss_is_nll():
    logp = torch.log_softmax(torch.tensor([[0.0, 1.0, 2.0]]), -1)
    assert torch.allclose(supervised_loss(logp, torch.tensor([2])), -logp[0, 2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rewards.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement env.py**

```python
"""Bandit environment: reveals only whether the sampled action was correct."""
from __future__ import annotations

import torch


class BanditEnv:
    def __init__(self, answers: torch.Tensor):
        self._answers = answers.clone().long()

    def __len__(self) -> int:
        return int(self._answers.numel())

    def step(self, idx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """idx (B,), actions (B,G) -> outcomes (B,G) in {0.0, 1.0}."""
        truth = self._answers[idx.cpu()].to(actions.device)
        return (actions == truth[:, None]).float()

    def reveal(self, idx: torch.Tensor) -> torch.Tensor:
        """Full labels. Only the supervised oracle arm may call this."""
        return self._answers[idx.cpu()]
```

- [ ] **Step 4: Implement rewards.py**

```python
"""Reward functions and the losses built on them."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reward_rlvr(outcomes: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """Verifiable outcome only."""
    return outcomes


def reward_rlcd(outcomes: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """Outcome minus the stated probability of the taken action.
    REINFORCE with this reward is an unbiased estimator of half the Brier
    score gradient using only bandit feedback."""
    return outcomes - p_a


REWARDS = {"rlvr": reward_rlvr, "rlcd": reward_rlcd}


def gather_logp(logp: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.gather(logp, 1, actions)


def policy_gradient_loss(logp: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """logp (B,K), actions (B,G), rewards (B,G). Leave-one-out baseline over the group."""
    _, group = rewards.shape
    if group > 1:
        baseline = (rewards.sum(1, keepdim=True) - rewards) / (group - 1)
    else:
        baseline = torch.zeros_like(rewards)
    advantage = (rewards - baseline).detach()
    return -(advantage * gather_logp(logp, actions)).mean()


def kl_categorical(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """KL(p || p_ref) per row. Masked positions carry zero mass in both, so the
    product is exactly zero there as long as the logits are finite (NEG, not -inf)."""
    p = logp.exp()
    return (p * (logp - logp_ref)).sum(-1)


def supervised_loss(logp: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(logp, answers.to(logp.device))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rewards.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```powershell
git add rlcd/env.py rlcd/rewards.py tests/test_rewards.py
git commit -m "Add bandit environment and RLCD/RLVR reward objectives"
```

---

### Task 5: Metrics

**Files:**
- Create: `rlcd/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces (all numpy in, python floats or numpy out):
  - `brier(probs: (N,K) ndarray, answers: (N,) ndarray) -> float` (multiclass, sum over K then mean over N)
  - `ece(conf: (N,), correct: (N,), n_bins: int = 15) -> float`
  - `reliability_bins(conf, correct, n_bins=15) -> tuple[ndarray bin_conf, ndarray bin_acc, ndarray bin_count]` (NaN for empty bins)
  - `coverage_error(conf, correct) -> tuple[ndarray coverage, ndarray error]` (sorted by confidence descending, cumulative)
  - `nota_rate(pred: (N,), answers: (N,), nota_index: (N,)) -> float` (nota_index is -1 when absent; returns fraction of NOTA-correct rows predicted NOTA, or NaN if none)
  - `entropy_confidence(probs: (N,K), k: (N,)) -> ndarray` (1 - H/log k)

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np

from rlcd.metrics import (brier, coverage_error, ece, entropy_confidence, nota_rate,
                          reliability_bins)


def test_brier_multiclass():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    answers = np.array([0, 0])
    # row0: (0.3^2 + 0.3^2) = .18 ; row1: (0.8^2 + 0.8^2) = 1.28 ; mean .73
    assert abs(brier(probs, answers) - 0.73) < 1e-9


def test_ece_perfect_and_worst():
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    assert abs(ece(conf, correct, n_bins=10) - 0.0) < 1e-9
    assert abs(ece(conf, np.zeros(10), n_bins=10) - 0.9) < 1e-9


def test_ece_weights_bins_by_count():
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1, 1, 0, 0])
    # bin .9-1: |1 - .95| = .05, weight .5 ; bin .5-.6: |0 - .55| = .55, weight .5
    assert abs(ece(conf, correct, n_bins=10) - 0.30) < 1e-9


def test_reliability_bins_shape_and_nan():
    conf = np.array([0.95, 0.15])
    correct = np.array([1, 0])
    bc, ba, bn = reliability_bins(conf, correct, n_bins=10)
    assert bc.shape == ba.shape == bn.shape == (10,)
    assert bn[9] == 1 and bn[1] == 1 and bn[5] == 0
    assert np.isnan(ba[5])
    assert ba[9] == 1.0 and ba[1] == 0.0


def test_coverage_error_curve():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])
    cov, err = coverage_error(conf, correct)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert np.allclose(err, [0.0, 0.0, 1 / 3, 0.25])


def test_nota_rate():
    pred = np.array([3, 3, 0, 1])
    answers = np.array([3, 2, 3, 1])
    nota_index = np.array([3, 3, 3, -1])
    # rows where nota is the answer: 0 and 2; predicted nota in row 0 only
    assert abs(nota_rate(pred, answers, nota_index) - 0.5) < 1e-9
    assert np.isnan(nota_rate(pred, answers, np.full(4, -1)))


def test_entropy_confidence():
    probs = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    k = np.array([3, 2])
    out = entropy_confidence(probs, k)
    assert abs(out[0] - 1.0) < 1e-9
    assert abs(out[1] - 0.0) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement metrics.py**

```python
"""Calibration and selective-prediction metrics."""
from __future__ import annotations

import numpy as np


def brier(probs: np.ndarray, answers: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(answers)), answers] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def _bin_ids(conf: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.searchsorted(edges, conf, side="right") - 1
    return np.clip(ids, 0, n_bins - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    ids = _bin_ids(np.asarray(conf, float), n_bins)
    correct = np.asarray(correct, float)
    bin_conf = np.full(n_bins, np.nan)
    bin_acc = np.full(n_bins, np.nan)
    bin_count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = ids == b
        bin_count[b] = int(m.sum())
        if bin_count[b]:
            bin_conf[b] = conf[m].mean()
            bin_acc[b] = correct[m].mean()
    return bin_conf, bin_acc, bin_count


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bin_conf, bin_acc, bin_count = reliability_bins(conf, correct, n_bins)
    n = bin_count.sum()
    if n == 0:
        return float("nan")
    m = bin_count > 0
    return float((bin_count[m] / n * np.abs(bin_acc[m] - bin_conf[m])).sum())


def coverage_error(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(correct, float)[order]
    n = len(c)
    covered = np.arange(1, n + 1)
    coverage = covered / n
    error = 1.0 - np.cumsum(c) / covered
    return coverage, error


def nota_rate(pred: np.ndarray, answers: np.ndarray, nota_index: np.ndarray) -> float:
    is_nota_answer = (nota_index >= 0) & (answers == nota_index)
    if not is_nota_answer.any():
        return float("nan")
    return float((pred[is_nota_answer] == nota_index[is_nota_answer]).mean())


def entropy_confidence(probs: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), 1e-12, 1.0)
    h = -(np.asarray(probs, float) * np.log(p)).sum(1)
    return 1.0 - h / np.log(np.asarray(k, float))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```powershell
git add rlcd/metrics.py tests/test_metrics.py
git commit -m "Add calibration and coverage metrics"
```

---

### Task 6: Data pipeline

**Files:**
- Create: `rlcd/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `rlcd.schema.Question`, `NOTA`, `write_jsonl`.
- Produces:
  - `subset_choices(all_choices: list[str], truth: str, n: int, rng: random.Random) -> tuple[list[str], int]`
  - `inject_nota(q: Question, rng, p_remove: float = 0.15, p_add: float = 0.35) -> Question`
  - `synthetic_triage(n_records: int, rng) -> list[Question]` (3 questions per record)
  - `SOURCES: dict[str, callable]` mapping source name to a loader returning `list[Question]` (network)
  - `build(out_dir: str, per_source: int = 8000, n_val: int = 1000, n_test: int = 1000, seed: int = 0) -> dict` (writes train/val/test.jsonl and stats.json)
  - CLI: `python -m rlcd.data build --out data --per-source 8000 --seed 0`

- [ ] **Step 1: Write the failing tests (offline parts only)**

`tests/test_data.py`:
```python
import random
from collections import Counter

from rlcd.data import inject_nota, split_source, subset_choices, synthetic_triage
from rlcd.schema import NOTA, Question


def test_subset_choices_contains_truth_once_and_size():
    rng = random.Random(0)
    pool = [f"intent_{i}" for i in range(77)]
    choices, answer = subset_choices(pool, "intent_5", 25, rng)
    assert len(choices) == 25 and len(set(choices)) == 25
    assert choices[answer] == "intent_5"
    assert choices.count("intent_5") == 1


def test_subset_choices_small_pool_keeps_all():
    choices, answer = subset_choices(["a", "b", "c"], "b", 25, random.Random(0))
    assert sorted(choices) == ["a", "b", "c"] and choices[answer] == "b"


def _q(answer=1, primitive="choice", **kw):
    base = dict(primitive=primitive, context="c", question="q", choices=["x", "y", "z"], answer=answer)
    base.update(kw)
    return Question(**base).validate()


def test_inject_nota_mix_over_many_draws():
    rng = random.Random(0)
    kinds = Counter()
    for _ in range(4000):
        out = inject_nota(_q(), rng)
        if NOTA not in out.choices:
            kinds["absent"] += 1
        elif out.choices[out.answer] == NOTA:
            kinds["nota_correct"] += 1
            assert "y" not in out.choices
            assert len(out.choices) == 3
        else:
            kinds["nota_distractor"] += 1
            assert out.choices[out.answer] == "y"
            assert len(out.choices) == 4
    assert 0.10 < kinds["nota_correct"] / 4000 < 0.20
    assert 0.30 < kinds["nota_distractor"] / 4000 < 0.40
    assert 0.45 < kinds["absent"] / 4000 < 0.55


def test_inject_nota_never_touches_noul():
    q = _q(primitive="noul", choices=["true", "false"], answer=0)
    for _ in range(50):
        assert inject_nota(q, random.Random(1)) == q


def test_inject_nota_respects_cardinality_cap():
    q = _q(choices=[str(i) for i in range(26)], answer=3)
    rng = random.Random(0)
    for _ in range(200):
        out = inject_nota(q, rng)
        assert out.k <= 26
        out.validate()


def test_synthetic_triage_shape():
    qs = synthetic_triage(10, random.Random(0))
    assert len(qs) == 30
    prims = Counter(q.primitive for q in qs)
    assert prims == {"choice": 10, "score": 10, "noul": 10}
    for q in qs:
        q.validate()
        assert q.source == "triage"
    score_q = next(q for q in qs if q.primitive == "score")
    assert score_q.ordered and score_q.choices == ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]


def test_synthetic_triage_is_deterministic():
    a = synthetic_triage(5, random.Random(7))
    b = synthetic_triage(5, random.Random(7))
    assert a == b


def test_split_source_sizes_and_disjoint():
    qs = [_q(id=str(i)) for i in range(100)]
    train, val, test = split_source(qs, per_source=50, n_val=20, n_test=20, rng=random.Random(0))
    assert len(train) == 50 and len(val) == 20 and len(test) == 20
    ids = [q.id for q in train + val + test]
    assert len(set(ids)) == 90
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement data.py**

```python
"""Dataset converters, NOTA injection, synthetic triage, and the build CLI."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rlcd.schema import MAX_CHOICES, NOTA, Question, write_jsonl


# ---------- helpers ----------

def subset_choices(all_choices: list[str], truth: str, n: int, rng: random.Random) -> tuple[list[str], int]:
    """n-1 random distractors plus the truth at a random position."""
    distractors = [c for c in all_choices if c != truth]
    rng.shuffle(distractors)
    chosen = distractors[: max(n - 1, 0)]
    pos = rng.randrange(len(chosen) + 1)
    chosen.insert(pos, truth)
    return chosen, pos


def inject_nota(q: Question, rng: random.Random, p_remove: float = 0.15, p_add: float = 0.35) -> Question:
    """With p_remove: drop the true choice and make NOTA correct.
    With p_add: append NOTA as a distractor. Otherwise unchanged. Never for noul."""
    if q.primitive == "noul" or NOTA in q.choices:
        return q
    r = rng.random()
    if r < p_remove:
        choices = [c for i, c in enumerate(q.choices) if i != q.answer]
        if len(choices) >= MAX_CHOICES:
            choices = choices[: MAX_CHOICES - 1]
        choices.append(NOTA)
        return Question(q.primitive, q.context, q.question, choices, q.ordered,
                        len(choices) - 1, q.source, q.id)
    if r < p_remove + p_add:
        choices = list(q.choices)
        answer = q.answer
        if len(choices) >= MAX_CHOICES:
            drop = rng.choice([i for i in range(len(choices)) if i != answer])
            choices.pop(drop)
            if drop < answer:
                answer -= 1
        choices.append(NOTA)
        return Question(q.primitive, q.context, q.question, choices, q.ordered, answer, q.source, q.id)
    return q


def split_source(qs: list[Question], per_source: int, n_val: int, n_test: int, rng: random.Random):
    qs = list(qs)
    rng.shuffle(qs)
    test = qs[:n_test]
    val = qs[n_test:n_test + n_val]
    train = qs[n_test + n_val:n_test + n_val + per_source]
    return train, val, test


# ---------- synthetic triage ----------

DEPARTMENTS = ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]
PRIORITIES = ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]  # ascending severity

DEPT_CUES = {
    "BILLING": ["invoice total is wrong", "charged twice this month", "refund has not arrived",
                "credit card declined at renewal", "tax line looks incorrect"],
    "INFRASTRUCTURE": ["database CPU pinned at 100 percent", "API latency spiked to 4 seconds",
                       "pods crash looping in prod", "disk on the primary node is full",
                       "load balancer returning 502s"],
    "SECURITY": ["login from an unknown country", "suspected leaked API key",
                 "phishing email hit the whole team", "MFA bypass reported", "unexpected admin role grant"],
    "PRODUCT_SUPPORT": ["export button does nothing", "cannot find the settings page",
                        "how do I invite a teammate", "dark mode resets on reload", "search ignores filters"],
}
PRIO_CUES = {
    "P3_LOW": ["no rush", "cosmetic", "whenever you get a chance"],
    "P2_NORMAL": ["please look into this", "affecting one user", "not blocking"],
    "P1_HIGH": ["blocking our team", "several customers affected", "need this today"],
    "P0_CRITICAL": ["production is down", "all customers affected", "revenue impact right now"],
}
CUSTOMER_TIERS = ["free tier", "startup plan", "enterprise, tier 1", "enterprise, tier 2"]


def synthetic_triage(n_records: int, rng: random.Random) -> list[Question]:
    out: list[Question] = []
    for i in range(n_records):
        dept = rng.choice(DEPARTMENTS)
        prio_idx = rng.randrange(4)
        cues = [rng.choice(DEPT_CUES[dept])]
        label_dept = dept
        if rng.random() < 0.3:  # genuinely ambiguous record
            other = rng.choice([d for d in DEPARTMENTS if d != dept])
            cues.append(rng.choice(DEPT_CUES[other]))
            if rng.random() < 0.4:
                label_dept = other
        rng.shuffle(cues)
        text = (f"Ticket #{1000 + i} from a {rng.choice(CUSTOMER_TIERS)} customer. "
                f"Report: {'; '.join(cues)}. Note: {rng.choice(PRIO_CUES[PRIORITIES[prio_idx]])}.")
        escalate = prio_idx >= 2
        if rng.random() < 0.1:
            escalate = not escalate
        rid = f"triage-{i}"
        out.append(Question("choice", text, "Which department should handle this ticket?",
                            list(DEPARTMENTS), False, DEPARTMENTS.index(label_dept), "triage", rid + "-dept"))
        out.append(Question("score", text, "What is the priority of this ticket?",
                            list(PRIORITIES), True, prio_idx, "triage", rid + "-prio"))
        out.append(Question("noul", text, "Should an on-call engineer be paged immediately?",
                            ["true", "false"], False, 0 if escalate else 1, "triage", rid + "-esc"))
    return out


# ---------- public dataset converters (network) ----------

def _ds(name: str, split: str, **kw):
    from datasets import load_dataset
    return load_dataset(name, split=split, **kw)


def load_bitext(rng: random.Random) -> list[Question]:
    ds = _ds("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "train")
    intents = sorted(set(ds["intent"]))
    out = []
    for i, row in enumerate(ds):
        choices, ans = subset_choices(intents, row["intent"], 25, rng)
        out.append(Question("choice", row["instruction"], "What is the customer's intent?",
                            choices, False, ans, "bitext", f"bitext-{i}"))
    return out


def load_banking77(rng: random.Random) -> list[Question]:
    ds = _ds("PolyAI/banking77", "train+test")
    names = ds.features["label"].names
    out = []
    for i, row in enumerate(ds):
        truth = names[row["label"]]
        choices, ans = subset_choices(names, truth, 25, rng)
        out.append(Question("choice", row["text"], "Which banking intent does the message express?",
                            choices, False, ans, "banking77", f"banking77-{i}"))
    return out


def load_ag_news(rng: random.Random) -> list[Question]:
    ds = _ds("fancyzhx/ag_news", "train[:12000]+test[:3000]")
    names = ["World", "Sports", "Business", "Sci/Tech"]
    return [Question("choice", row["text"], "What is the topic of this article?",
                     list(names), False, int(row["label"]), "ag_news", f"ag_news-{i}")
            for i, row in enumerate(ds)]


def load_mnli(rng: random.Random) -> list[Question]:
    ds = _ds("nyu-mll/multi_nli", "train[:12000]+validation_matched[:3000]")
    names = ["entailment", "neutral", "contradiction"]
    out = []
    for i, row in enumerate(ds):
        if row["label"] < 0:
            continue
        out.append(Question("choice", row["premise"],
                            f'What is the relation of the context to the hypothesis: "{row["hypothesis"]}"?',
                            list(names), False, int(row["label"]), "mnli", f"mnli-{i}"))
    return out


def load_sst5(rng: random.Random) -> list[Question]:
    ds = _ds("SetFit/sst5", "train+validation+test")
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    return [Question("score", row["text"], "What is the sentiment of the text?",
                     list(levels), True, int(row["label"]), "sst5", f"sst5-{i}")
            for i, row in enumerate(ds)]


def load_yelp(rng: random.Random) -> list[Question]:
    ds = _ds("Yelp/yelp_review_full", "train[:12000]+test[:3000]")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    return [Question("score", row["text"][:1500], "How many stars did the reviewer give?",
                     list(levels), True, int(row["label"]), "yelp", f"yelp-{i}")
            for i, row in enumerate(ds)]


def load_boolq(rng: random.Random) -> list[Question]:
    ds = _ds("google/boolq", "train+validation")
    out = []
    for i, row in enumerate(ds):
        proposition = row["question"].strip().rstrip("?") + "?"
        out.append(Question("noul", row["passage"][:1500], f"Is the answer to this question yes: {proposition}",
                            ["true", "false"], False, 0 if row["answer"] else 1, "boolq", f"boolq-{i}"))
    return out


def load_triage(rng: random.Random) -> list[Question]:
    return synthetic_triage(4000, rng)


SOURCES = {
    "bitext": load_bitext,
    "banking77": load_banking77,
    "ag_news": load_ag_news,
    "mnli": load_mnli,
    "sst5": load_sst5,
    "yelp": load_yelp,
    "boolq": load_boolq,
    "triage": load_triage,
}


# ---------- build ----------

def build(out_dir: str, per_source: int = 8000, n_val: int = 1000, n_test: int = 1000,
          seed: int = 0, sources: list[str] | None = None) -> dict:
    rng = random.Random(seed)
    train, val, test = [], [], []
    stats = {}
    for name in sources or list(SOURCES):
        qs = SOURCES[name](rng)
        tr, va, te = split_source(qs, per_source, n_val, n_test, rng)
        tr = [inject_nota(q, rng) for q in tr]
        va = [inject_nota(q, rng) for q in va]
        te = [inject_nota(q, rng) for q in te]
        train += tr; val += va; test += te
        stats[name] = {"train": len(tr), "val": len(va), "test": len(te)}
    rng.shuffle(train)
    out = Path(out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    stats["total"] = {"train": len(train), "val": len(val), "test": len(test)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data")
    b.add_argument("--per-source", type=int, default=8000)
    b.add_argument("--n-val", type=int, default=1000)
    b.add_argument("--n-test", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sources", nargs="*", default=None)
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.per_source, args.n_val, args.n_test, args.seed, args.sources), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 8 passed.

- [ ] **Step 5: Build the offline slice first, then the full dataset**

```powershell
uv run python -m rlcd.data build --out data/triage-only --sources triage --per-source 500 --n-val 100 --n-test 100
uv run python -m rlcd.data build --out data --per-source 8000 --seed 0
```
Expected: first command finishes in seconds and prints stats with `"total": {"train": 500, ...}`. Second command downloads the public sets (several hundred MB total, several minutes) and prints per-source counts; total train should be roughly 50,000 to 60,000 (Bitext, Banking77, SST-5 and BoolQ have fewer than 10,000 rows after carving val and test, so they come in under 8,000 train). If a dataset name fails to load, check the hub page for the current repo id and fix the id in `SOURCES`; do not silently drop the source.

- [ ] **Step 6: Inspect a few rendered examples**

```powershell
uv run python -c "from rlcd.schema import read_jsonl, render_prompt; qs=read_jsonl('data/train.jsonl'); [print(render_prompt(q), '\n=> answer', q.answer, '\n') for q in qs[:3]]"
```
Expected: three prompts in the exact format, answers in range, at least one showing `None of the above` after a few reruns with different slices.

- [ ] **Step 7: Commit**

```powershell
git add rlcd/data.py tests/test_data.py
git commit -m "Add dataset converters, NOTA injection, and synthetic triage"
```

---

### Task 7: Shared training utilities and warmup SFT

**Files:**
- Create: `rlcd/loop.py`, `rlcd/train_sft.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `load_eve`, `questions_to_batch`, `decision_logits`, `letter_token_ids`, `supervised_loss`, `read_jsonl`.
- Produces:
  - `cosine_lr(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float`
  - `batch_indices(n: int, batch_size: int, rng: random.Random) -> Iterator[list[int]]` (shuffled, last partial batch kept)
  - `save_checkpoint(model, tokenizer, out_dir: str, meta: dict) -> None` (writes HF files plus `meta.json`)
  - `JsonlLogger(path)` with `.log(**kw)`
  - CLI `python -m rlcd.train_sft --data data/train.jsonl --out runs/sft --n 2000 --epochs 1 --micro 16 --accum 4 --lr 5e-5 --max-len 512 --seed 0`

- [ ] **Step 1: Write the failing tests**

`tests/test_loop.py`:
```python
import json
import random

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr


def test_cosine_lr_warmup_peak_floor():
    assert cosine_lr(0, 100, 1.0, warmup=10) == 0.0
    assert abs(cosine_lr(10, 100, 1.0, warmup=10) - 1.0) < 1e-9
    assert abs(cosine_lr(100, 100, 1.0, warmup=10) - 0.1) < 1e-9
    assert 0.1 < cosine_lr(55, 100, 1.0, warmup=10) < 1.0


def test_batch_indices_cover_everything_once():
    batches = list(batch_indices(10, 4, random.Random(0)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(i for b in batches for i in b) == list(range(10))


def test_jsonl_logger(tmp_path):
    log = JsonlLogger(tmp_path / "log.jsonl")
    log.log(step=1, loss=0.5)
    log.log(step=2, loss=0.25)
    rows = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rows == [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loop.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement loop.py**

```python
"""Small shared pieces for the training scripts."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterator


def cosine_lr(step: int, total: int, peak: float, warmup: int, floor: float = 0.1) -> float:
    if step < warmup:
        return peak * step / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def batch_indices(n: int, batch_size: int, rng: random.Random) -> Iterator[list[int]]:
    order = list(range(n))
    rng.shuffle(order)
    for i in range(0, n, batch_size):
        yield order[i:i + batch_size]


def save_checkpoint(model, tokenizer, out_dir: str, meta: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, **kw) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Implement train_sft.py**

```python
"""Warmup SFT: teach the letter format on a small labeled slice."""
from __future__ import annotations

import argparse
import random
import time

import torch

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr, save_checkpoint
from rlcd.policy import EVE_ID, decision_logits, load_eve, questions_to_batch
from rlcd.rewards import supervised_loss
from rlcd.schema import letter_token_ids, read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--init", default=EVE_ID)
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    qs = read_jsonl(args.data)
    rng.shuffle(qs)
    qs = qs[: args.n]
    answers = torch.tensor([q.answer for q in qs])

    model, tok = load_eve(args.init, device="cuda")
    model.train()
    letters = letter_token_ids(tok)
    aux_coef = model.config.router_aux_loss_coef
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = (len(qs) + args.micro * args.accum - 1) // (args.micro * args.accum)
    total = steps_per_epoch * args.epochs
    log = JsonlLogger(f"{args.out}/train_log.jsonl")

    step, micro_step, t0 = 0, 0, time.time()
    for epoch in range(args.epochs):
        for idx in batch_indices(len(qs), args.micro, rng):
            batch = [qs[i] for i in idx]
            ids, last, k = questions_to_batch(tok, batch, args.max_len, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = decision_logits(model, ids, last, letters, k)
            logp = torch.log_softmax(logits, -1)
            loss = supervised_loss(logp, answers[idx]) + aux_coef * aux
            (loss / args.accum).backward()
            micro_step += 1
            if micro_step % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, args.lr, warmup=max(1, total // 20))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                with torch.no_grad():
                    acc = (logp.argmax(-1).cpu() == answers[idx]).float().mean().item()
                    conf = logp.exp().max(-1).values.mean().item()
                if step % 5 == 0 or step == total:
                    log.log(step=step, epoch=epoch, loss=loss.item(), acc=acc, mean_conf=conf,
                            lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
    save_checkpoint(model, tok, args.out, vars(args) | {"steps": step})
    print("saved", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke run on the triage-only slice**

```powershell
uv run python -m rlcd.train_sft --data data/triage-only/train.jsonl --out runs/smoke-sft --n 200 --micro 8 --accum 2
```
Expected: about 13 optimizer steps, loss falling from around 1.4 toward under 1.0, `acc` rising, checkpoint written to `runs/smoke-sft` with `model.safetensors`, `config.json`, tokenizer files, `meta.json`. VRAM stays well under 16 GB. If you see `CUDA out of memory`, lower `--micro` to 8 and raise `--accum`.

- [ ] **Step 7: Verify the checkpoint reloads with the tie intact**

```powershell
uv run python -c "from rlcd.policy import load_eve; m,t=load_eve('runs/smoke-sft'); print('ok', m.lm_head.weight.data_ptr()==m.transformer.wte.weight.data_ptr())"
```
Expected: `ok True`.

- [ ] **Step 8: Run the real warmup**

```powershell
uv run python -m rlcd.train_sft --data data/train.jsonl --out runs/sft --n 2000 --epochs 1 --micro 16 --accum 4 --lr 5e-5
```
Expected: about 32 steps, a few minutes. Final `acc` above chance for a mixed batch (chance is roughly 0.15 given the cardinality mix). Note the final loss in `runs/sft/train_log.jsonl`.

- [ ] **Step 9: Commit**

```powershell
git add rlcd/loop.py rlcd/train_sft.py tests/test_loop.py
git commit -m "Add training utilities and warmup SFT"
```

---

### Task 8: RL training loop with rlvr, rlcd, and oracle arms

**Files:**
- Create: `rlcd/train_rl.py`
- Test: `tests/test_train_rl.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 4, 7.
- Produces:
  - `rl_step(model, ref_model, tok, letters, batch: list[Question], idx: LongTensor, env: BanditEnv, arm: str, group: int, kl_coef: float, aux_coef: float, max_len: int) -> tuple[Tensor loss, dict stats]`
  - CLI `python -m rlcd.train_rl --init runs/sft --arm rlcd --out runs/rlcd --data data/train.jsonl --epochs 1 --micro 16 --accum 8 --lr 2e-5 --group 4 --kl 0.05 --max-len 512 --limit 0 --seed 0`

- [ ] **Step 1: Write the failing test**

`tests/test_train_rl.py`:
```python
import copy

import torch

from rlcd.env import BanditEnv
from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.schema import Question
from rlcd.train_rl import rl_step


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny():
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=256,
                    num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg)


def _batch():
    return [Question("choice", "ctx one", "q", ["a", "b", "c"], answer=1),
            Question("noul", "ctx two", "q", ["true", "false"], answer=0)]


def test_rl_step_each_arm_produces_finite_loss_and_grads():
    model = tiny().train()
    ref = copy.deepcopy(model).eval()
    env = BanditEnv(torch.tensor([1, 0]))
    letters = list(range(100, 126))
    for arm in ("rlvr", "rlcd", "oracle"):
        model.zero_grad(set_to_none=True)
        loss, stats = rl_step(model, ref, FakeTok(), letters, _batch(), torch.tensor([0, 1]), env,
                              arm=arm, group=3, kl_coef=0.05, aux_coef=0.01, max_len=64, device="cpu")
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert set(stats) >= {"loss", "reward", "sampled_acc", "mean_conf", "kl"}


def test_bandit_arms_never_call_reveal(monkeypatch):
    model = tiny().train()
    ref = copy.deepcopy(model).eval()
    env = BanditEnv(torch.tensor([1, 0]))

    def boom(idx):
        raise AssertionError("reveal called by a bandit arm")

    monkeypatch.setattr(env, "reveal", boom)
    for arm in ("rlvr", "rlcd"):
        rl_step(model, ref, FakeTok(), list(range(100, 126)), _batch(), torch.tensor([0, 1]), env,
                arm=arm, group=2, kl_coef=0.0, aux_coef=0.0, max_len=64, device="cpu")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_train_rl.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement train_rl.py**

```python
"""RL over the sliced-softmax policy. Arms: rlvr (r = c), rlcd (r = c - p_a), oracle (full-label NLL)."""
from __future__ import annotations

import argparse
import copy
import random
import time
from contextlib import nullcontext

import torch

from rlcd.env import BanditEnv
from rlcd.loop import JsonlLogger, batch_indices, cosine_lr, save_checkpoint
from rlcd.policy import decision_logits, load_eve, questions_to_batch
from rlcd.rewards import REWARDS, kl_categorical, policy_gradient_loss, supervised_loss
from rlcd.schema import Question, letter_token_ids, read_jsonl

ARMS = ("rlvr", "rlcd", "oracle")


def rl_step(model, ref_model, tok, letters, batch: list[Question], idx: torch.Tensor, env: BanditEnv,
            arm: str, group: int, kl_coef: float, aux_coef: float, max_len: int, device: str = "cuda"):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    ids, last, k = questions_to_batch(tok, batch, max_len, device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    with autocast:
        logits, aux = decision_logits(model, ids, last, letters, k)
        with torch.no_grad():
            ref_logits, _ = decision_logits(ref_model, ids, last, letters, k)
    logp = torch.log_softmax(logits, -1)
    logp_ref = torch.log_softmax(ref_logits, -1)
    probs = logp.exp()
    stats = {}

    if arm == "oracle":
        answers = env.reveal(idx).to(device)
        objective = supervised_loss(logp, answers)
        stats["reward"] = float("nan")
        stats["sampled_acc"] = (probs.argmax(-1) == answers).float().mean().item()
    else:
        actions = torch.multinomial(probs.detach(), group, replacement=True)  # (B,G)
        outcomes = env.step(idx, actions)                                     # (B,G), bandit feedback only
        p_a = torch.gather(probs.detach(), 1, actions)
        rewards = REWARDS[arm](outcomes, p_a)
        objective = policy_gradient_loss(logp, actions, rewards)
        stats["reward"] = rewards.mean().item()
        stats["sampled_acc"] = outcomes.mean().item()

    kl = kl_categorical(logp, logp_ref).mean()
    loss = objective + kl_coef * kl + aux_coef * aux
    stats.update(loss=loss.item(), kl=kl.item(), mean_conf=probs.max(-1).values.mean().item(),
                 aux=float(aux))
    return loss, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/sft")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--kl", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N examples (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    qs = read_jsonl(args.data)
    if args.limit:
        qs = qs[: args.limit]
    env = BanditEnv(torch.tensor([q.answer for q in qs]))

    model, tok = load_eve(args.init, device="cuda")
    model.train()
    # Keep the reference in fp32. Eve's RoPE buffer is complex64 and Module.to(bfloat16)
    # would silently drop its imaginary part. fp32 costs about 1.1 GB, which fits.
    ref_model = copy.deepcopy(model).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    letters = letter_token_ids(tok)
    aux_coef = model.config.router_aux_loss_coef
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = (len(qs) + args.micro * args.accum - 1) // (args.micro * args.accum)
    total = steps_per_epoch * args.epochs
    log = JsonlLogger(f"{args.out}/train_log.jsonl")

    step, micro_step, t0 = 0, 0, time.time()
    running: dict[str, float] = {}
    for epoch in range(args.epochs):
        for idx in batch_indices(len(qs), args.micro, rng):
            batch = [qs[i] for i in idx]
            loss, stats = rl_step(model, ref_model, tok, letters, batch, torch.tensor(idx), env,
                                  args.arm, args.group, args.kl, aux_coef, args.max_len)
            (loss / args.accum).backward()
            for key, val in stats.items():
                running[key] = running.get(key, 0.0) + val / args.accum
            micro_step += 1
            if micro_step % args.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = cosine_lr(step, total, args.lr, warmup=max(1, total // 20))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0 or step == total:
                    log.log(step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], sec=time.time() - t0, **running)
                running = {}
    save_checkpoint(model, tok, args.out, vars(args) | {"steps": step})
    print("saved", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_train_rl.py -v`
Expected: 2 passed.

- [ ] **Step 5: Smoke run all three arms on 400 examples**

```powershell
uv run python -m rlcd.train_rl --init runs/sft --arm rlcd   --out runs/smoke-rlcd   --limit 400 --micro 8 --accum 2
uv run python -m rlcd.train_rl --init runs/sft --arm rlvr   --out runs/smoke-rlvr   --limit 400 --micro 8 --accum 2
uv run python -m rlcd.train_rl --init runs/sft --arm oracle --out runs/smoke-oracle --limit 400 --micro 8 --accum 2
```
Expected: each runs 25 steps in under two minutes, logs finite `loss`, `kl` near 0 at the start and small (under 0.1) throughout, and writes a checkpoint. For `rlvr` expect `mean_conf` to climb faster than for `rlcd`.

- [ ] **Step 6: Full runs (one epoch over all training data, about 400 to 450 steps each)**

```powershell
uv run python -m rlcd.train_rl --init runs/sft --arm rlcd   --out runs/rlcd
uv run python -m rlcd.train_rl --init runs/sft --arm rlvr   --out runs/rlvr
uv run python -m rlcd.train_rl --init runs/sft --arm oracle --out runs/oracle
```
Expected: 20 to 40 minutes each on the 4080. Watch `sampled_acc` rise in all arms. If `kl` grows past about 0.5 in an RL arm, stop and rerun with `--kl 0.1`. Record final log lines for the README.

- [ ] **Step 7: Commit**

```powershell
git add rlcd/train_rl.py tests/test_train_rl.py
git commit -m "Add RL loop with rlvr, rlcd, and oracle arms"
```

---

### Task 9: Evaluation, comparison, and temperature scaling

**Files:**
- Create: `rlcd/eval.py`, `rlcd/calibrate.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `load_eve`, `questions_to_batch`, `decision_logits`, `letter_token_ids`, `read_jsonl`, metrics from Task 5.
- Produces:
  - `predict(model, tok, questions, max_len=512, batch_size=32, device="cuda") -> list[dict]` with keys `id, source, primitive, k, answer, logits (list of k floats), nota_index`
  - `summarize(preds: list[dict], temperature: float = 1.0) -> dict` with keys `overall`, `by_primitive`, `by_source`, each a dict of `n, acc, brier, ece, nota_rate, mean_conf`
  - `plot_reliability(preds_by_name: dict[str, list[dict]], out_png, temperature_by_name=None)`, `plot_coverage(preds_by_name, out_png, temperature_by_name=None)`
  - `fit_temperature(preds: list[dict]) -> float`
  - CLI `python -m rlcd.eval run --model runs/rlcd --split data/test.jsonl --out runs/rlcd/eval [--temperature 1.0]`
  - CLI `python -m rlcd.eval compare --runs zeroshot=runs/zeroshot/eval sft=runs/sft/eval rlvr=runs/rlvr/eval rlcd=runs/rlcd/eval oracle=runs/oracle/eval --out runs/compare`
  - CLI `python -m rlcd.calibrate --preds runs/rlcd/eval-val/preds.jsonl --out runs/rlcd/temperature.json`

- [ ] **Step 1: Write the failing tests**

`tests/test_eval.py`:
```python
import math

import numpy as np

from rlcd.calibrate import fit_temperature
from rlcd.eval import summarize


def _pred(logits, answer, primitive="choice", source="s", nota_index=-1):
    return {"id": "x", "source": source, "primitive": primitive, "k": len(logits),
            "answer": answer, "logits": logits, "nota_index": nota_index}


def test_summarize_basic():
    preds = [_pred([2.0, 0.0], 0), _pred([0.0, 2.0], 0), _pred([0.0, 0.0, 3.0], 2, "score", "t")]
    s = summarize(preds)
    assert s["overall"]["n"] == 3
    assert abs(s["overall"]["acc"] - 2 / 3) < 1e-9
    assert set(s["by_primitive"]) == {"choice", "score"}
    assert set(s["by_source"]) == {"s", "t"}
    assert 0.0 <= s["overall"]["ece"] <= 1.0
    assert math.isnan(s["overall"]["nota_rate"])


def test_summarize_temperature_flattens_confidence():
    preds = [_pred([4.0, 0.0], 0)]
    assert summarize(preds, temperature=1.0)["overall"]["mean_conf"] > summarize(preds, temperature=4.0)["overall"]["mean_conf"]


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(2000):
        true_logits = rng.normal(size=4)
        answer = int(rng.choice(4, p=np.exp(true_logits) / np.exp(true_logits).sum()))
        preds.append(_pred((true_logits * 3.0).tolist(), answer))  # overconfident by 3x
    t = fit_temperature(preds)
    assert 2.4 < t < 3.6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement calibrate.py**

```python
"""Post-hoc temperature scaling on saved validation logits."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def fit_temperature(preds: list[dict]) -> float:
    """Minimize NLL over log T with LBFGS. Rows have variable k; pad with a large negative."""
    kmax = max(p["k"] for p in preds)
    logits = torch.full((len(preds), kmax), -1e4)
    for i, p in enumerate(preds):
        logits[i, : p["k"]] = torch.tensor(p["logits"])
    answers = torch.tensor([p["answer"] for p in preds])
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), answers)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    preds = [json.loads(l) for l in open(args.preds, encoding="utf-8") if l.strip()]
    result = {"overall": fit_temperature(preds)}
    for prim in sorted({p["primitive"] for p in preds}):
        result[prim] = fit_temperature([p for p in preds if p["primitive"] == prim])
    json.dump(result, open(args.out, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement eval.py**

```python
"""Predictions, metrics, and plots for one checkpoint; comparison across checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rlcd.metrics import brier, coverage_error, ece, nota_rate, reliability_bins
from rlcd.policy import decision_logits, load_eve, questions_to_batch
from rlcd.schema import NOTA, Question, letter_token_ids, read_jsonl


def predict(model, tok, questions: list[Question], max_len: int = 512, batch_size: int = 32,
            device: str = "cuda") -> list[dict]:
    model.eval()
    letters = letter_token_ids(tok)
    out = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        ids, last, k = questions_to_batch(tok, batch, max_len, device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad()
        with torch.no_grad(), ctx:
            logits, _ = decision_logits(model, ids, last, letters, k)
        for q, row in zip(batch, logits.float().cpu()):
            out.append({"id": q.id, "source": q.source, "primitive": q.primitive, "k": q.k,
                        "answer": q.answer, "logits": row[: q.k].tolist(),
                        "nota_index": q.choices.index(NOTA) if NOTA in q.choices else -1})
    return out


def _arrays(preds: list[dict], temperature: float):
    kmax = max(p["k"] for p in preds)
    probs = np.zeros((len(preds), kmax))
    for i, p in enumerate(preds):
        z = np.asarray(p["logits"], float) / temperature
        z = np.exp(z - z.max())
        probs[i, : p["k"]] = z / z.sum()
    answers = np.array([p["answer"] for p in preds])
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == answers).astype(float)
    nota_index = np.array([p["nota_index"] for p in preds])
    return probs, answers, pred, conf, correct, nota_index


def _group_metrics(preds: list[dict], temperature: float) -> dict:
    probs, answers, pred, conf, correct, nota_index = _arrays(preds, temperature)
    return {"n": len(preds), "acc": float(correct.mean()), "brier": brier(probs, answers),
            "ece": ece(conf, correct), "nota_rate": nota_rate(pred, answers, nota_index),
            "mean_conf": float(conf.mean())}


def summarize(preds: list[dict], temperature: float = 1.0) -> dict:
    result = {"overall": _group_metrics(preds, temperature), "by_primitive": {}, "by_source": {}}
    for key, field in (("by_primitive", "primitive"), ("by_source", "source")):
        for name in sorted({p[field] for p in preds}):
            result[key][name] = _group_metrics([p for p in preds if p[field] == name], temperature)
    return result


def plot_reliability(preds_by_name: dict[str, list[dict]], out_png, temperature_by_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, preds in preds_by_name.items():
        t = (temperature_by_name or {}).get(name, 1.0)
        _, _, _, conf, correct, _ = _arrays(preds, t)
        bc, ba, bn = reliability_bins(conf, correct, 15)
        m = bn > 0
        ax.plot(bc[m], ba[m], marker="o", label=f"{name} (ECE {ece(conf, correct):.3f})")
    ax.set_xlabel("stated confidence"); ax.set_ylabel("empirical accuracy"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_coverage(preds_by_name: dict[str, list[dict]], out_png, temperature_by_name=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, preds in preds_by_name.items():
        t = (temperature_by_name or {}).get(name, 1.0)
        _, _, _, conf, correct, _ = _arrays(preds, t)
        cov, err = coverage_error(conf, correct)
        ax.plot(cov, err, label=name)
    ax.set_xlabel("coverage (fraction answered autonomously)"); ax.set_ylabel("error rate among answered")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def _load_preds(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cmd_run(args):
    model, tok = load_eve(args.model, device="cuda")
    qs = read_jsonl(args.split)
    if args.limit:
        qs = qs[: args.limit]
    preds = predict(model, tok, qs, args.max_len, args.batch_size)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    summary = summarize(preds, args.temperature)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    name = Path(args.model).name
    plot_reliability({name: preds}, out / "reliability.png", {name: args.temperature})
    plot_coverage({name: preds}, out / "coverage.png", {name: args.temperature})
    print(json.dumps(summary["overall"], indent=2))


def cmd_compare(args):
    runs = dict(item.split("=", 1) for item in args.runs)
    preds_by_name = {name: _load_preds(Path(path) / "preds.jsonl") for name, path in runs.items()}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plot_reliability(preds_by_name, out / "reliability.png")
    plot_coverage(preds_by_name, out / "coverage.png")
    rows = ["| run | n | acc | brier | ECE | mean conf | NOTA rate |", "|---|---|---|---|---|---|---|"]
    table = {}
    for name, preds in preds_by_name.items():
        m = summarize(preds)["overall"]
        table[name] = m
        nota = "n/a" if math.isnan(m["nota_rate"]) else f"{m['nota_rate']:.3f}"
        rows.append(f"| {name} | {m['n']} | {m['acc']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {m['mean_conf']:.3f} | {nota} |")
    (out / "table.md").write_text("\n".join(rows) + "\n")
    (out / "metrics.json").write_text(json.dumps(table, indent=2))
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--split", default="data/test.jsonl")
    r.add_argument("--out", required=True)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--max-len", type=int, default=512)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--limit", type=int, default=0)
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True, help="name=path/to/eval-dir")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval.py -v`
Expected: 3 passed.

- [ ] **Step 6: Evaluate every arm plus the zero-shot base on the test split**

```powershell
uv run python -m rlcd.eval run --model anthonym21/Eve-2-MoE-IT-272M --out runs/zeroshot/eval
uv run python -m rlcd.eval run --model runs/sft    --out runs/sft/eval
uv run python -m rlcd.eval run --model runs/rlvr   --out runs/rlvr/eval
uv run python -m rlcd.eval run --model runs/rlcd   --out runs/rlcd/eval
uv run python -m rlcd.eval run --model runs/oracle --out runs/oracle/eval
uv run python -m rlcd.eval compare --runs zeroshot=runs/zeroshot/eval sft=runs/sft/eval rlvr=runs/rlvr/eval rlcd=runs/rlcd/eval oracle=runs/oracle/eval --out runs/compare
```
Expected: each `run` prints overall metrics and writes `preds.jsonl`, `metrics.json`, `reliability.png`, `coverage.png`. `compare` prints a markdown table and writes overlaid plots. The hypothesis under test: `rlcd` has lower ECE than `rlvr` at similar accuracy, and `rlvr` has the highest `mean_conf`. Report whatever the numbers say, including if the hypothesis fails.

- [ ] **Step 7: Fit and report temperatures on validation (reference only)**

```powershell
uv run python -m rlcd.eval run --model runs/rlcd --split data/val.jsonl --out runs/rlcd/eval-val
uv run python -m rlcd.eval run --model runs/rlvr --split data/val.jsonl --out runs/rlvr/eval-val
uv run python -m rlcd.calibrate --preds runs/rlcd/eval-val/preds.jsonl --out runs/rlcd/temperature.json
uv run python -m rlcd.calibrate --preds runs/rlvr/eval-val/preds.jsonl --out runs/rlvr/temperature.json
```
Expected: a fitted `overall` temperature near 1.0 for `rlcd` and noticeably above 1.0 for `rlvr` if the calibration story holds. These numbers go in the README as a secondary check; headline metrics stay at temperature 1.0.

- [ ] **Step 8: Commit**

```powershell
git add rlcd/eval.py rlcd/calibrate.py tests/test_eval.py
git commit -m "Add evaluation, comparison plots, and temperature fitting"
```

---

### Task 10: Amputation and the decision-only inference API

**Files:**
- Create: `rlcd/amputate.py`, `rlcd/infer.py`
- Test: `tests/test_infer.py`

**Interfaces:**
- Consumes: `load_eve`, `eve_hidden`, `encode_batch`, `letter_token_ids`, `Question`, `render_prompt`, `NEG`, `MAX_CHOICES`, `entropy_confidence`.
- Produces:
  - `amputate(src: str, out_dir: str) -> None` writes `out_dir/config.json`, `out_dir/transformer.safetensors` (everything except `lm_head.weight`), `out_dir/decision_head.safetensors` (`weight` of shape (26, n_embd)), `out_dir/decision.json` (`{"letters": [...], "letter_ids": [...]}`), tokenizer files
  - `DecisionModel.load(path: str, device: str = "cuda") -> DecisionModel`
  - `DecisionModel.ask(context: str, questions: list[Question]) -> list[dict]`
  - `DecisionModel.choice(context, question, choices) -> dict` with `probs: dict[str, float]`, `value: str`, `confidence: float`, `entropy_confidence: float`
  - `DecisionModel.score(context, question, levels) -> dict` (choice keys plus `score: float` in [0, 1])
  - `DecisionModel.noul(context, proposition) -> dict` with `p_true: float`
  - CLI `python -m rlcd.amputate --src runs/rlcd --out runs/rlcd/decision`

- [ ] **Step 1: Write the failing tests**

`tests/test_infer.py`:
```python
import torch

from rlcd.amputate import amputate, build_decision_head
from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.infer import DecisionModel
from rlcd.schema import Question


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]

    def save_pretrained(self, path):
        pass


def tiny():
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=256,
                    num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg).eval()


def test_decision_head_is_the_letter_rows():
    model = tiny()
    letter_ids = list(range(100, 126))
    head = build_decision_head(model, letter_ids)
    assert head.weight.shape == (26, 32)
    assert torch.equal(head.weight, model.lm_head.weight[letter_ids])


def test_amputated_model_matches_original_and_cannot_emit_text(tmp_path):
    model = tiny()
    letter_ids = list(range(100, 126))
    amputate(model, FakeTok(), letter_ids, tmp_path)
    dm = DecisionModel.load(tmp_path, device="cpu", tokenizer=FakeTok())
    assert dm.head.out_features == 26
    assert not hasattr(dm.model, "lm_head")
    qs = [Question("choice", "ctx", "q", ["a", "b", "c"]), Question("noul", "ctx", "q", ["true", "false"])]
    results = dm.ask("ctx", qs)
    # Compare against the original model's sliced softmax.
    from rlcd.policy import decision_logits, questions_to_batch
    ids, last, k = questions_to_batch(FakeTok(), qs)
    with torch.no_grad():
        logits, _ = decision_logits(model, ids, last, letter_ids, k)
    ref = torch.softmax(logits, -1)
    assert abs(results[0]["probs"]["a"] - ref[0, 0].item()) < 1e-4
    assert abs(results[1]["p_true"] - ref[1, 0].item()) < 1e-4


def test_primitive_helpers(tmp_path):
    model = tiny()
    amputate(model, FakeTok(), list(range(100, 126)), tmp_path)
    dm = DecisionModel.load(tmp_path, device="cpu", tokenizer=FakeTok())
    c = dm.choice("ctx", "q", ["x", "y"])
    assert set(c) == {"probs", "value", "confidence", "entropy_confidence"}
    assert abs(sum(c["probs"].values()) - 1.0) < 1e-6
    s = dm.score("ctx", "q", ["low", "mid", "high"])
    assert 0.0 <= s["score"] <= 1.0
    n = dm.noul("ctx", "the sky is blue")
    assert 0.0 <= n["p_true"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_infer.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement amputate.py**

```python
"""Replace the vocabulary head with the 26 decision rows and save a model that cannot talk."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from rlcd.policy import load_eve
from rlcd.schema import LETTERS, letter_token_ids


def build_decision_head(model, letter_ids: list[int]) -> torch.nn.Linear:
    head = torch.nn.Linear(model.config.n_embd, len(letter_ids), bias=False)
    with torch.no_grad():
        head.weight.copy_(model.lm_head.weight[torch.as_tensor(letter_ids)].detach().float())
    return head


def amputate(model, tokenizer, letter_ids: list[int], out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = build_decision_head(model, letter_ids)
    body = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(body, str(out / "transformer.safetensors"))
    save_file({"weight": head.weight.detach().cpu().contiguous()}, str(out / "decision_head.safetensors"))
    model.config.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "decision.json").write_text(json.dumps({"letters": LETTERS, "letter_ids": letter_ids}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model, tok = load_eve(args.src, device="cpu")
    amputate(model, tok, letter_token_ids(tok), args.out)
    print("saved decision-only model to", args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement infer.py**

```python
"""Choice / Score / Noul over an amputated decision model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rlcd.compat import load_eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.metrics import entropy_confidence
from rlcd.policy import encode_batch, eve_hidden
from rlcd.schema import MAX_CHOICES, NEG, Question, render_prompt


class DecisionModel:
    def __init__(self, model, head: torch.nn.Linear, tokenizer, device: str, max_len: int = 512):
        self.model, self.head, self.tok, self.device, self.max_len = model, head, tokenizer, device, max_len

    @classmethod
    def load(cls, path, device: str = "cuda", tokenizer=None, max_len: int = 512) -> "DecisionModel":
        path = Path(path)
        config = load_eve_config(str(path))
        config.tie_word_embeddings = False
        model = EveMoEForCausalLM(config)
        model.load_state_dict(load_file(str(path / "transformer.safetensors")), strict=False)
        del model.lm_head  # the model can no longer map hidden states to vocabulary
        head = torch.nn.Linear(config.n_embd, MAX_CHOICES, bias=False)
        head.load_state_dict(load_file(str(path / "decision_head.safetensors")))
        model.to(device).eval(); head.to(device).eval()
        tok = tokenizer or AutoTokenizer.from_pretrained(path)
        return cls(model, head, tok, device, max_len)

    @torch.no_grad()
    def _probs(self, questions: list[Question]) -> torch.Tensor:
        ids, last = encode_batch(self.tok, [render_prompt(q) for q in questions], self.max_len, self.device)
        k = torch.tensor([q.k for q in questions], device=self.device)
        hidden, _ = eve_hidden(self.model, ids)
        rows = hidden[torch.arange(len(questions), device=self.device), last].float()
        logits = self.head(rows)
        mask = torch.arange(MAX_CHOICES, device=self.device)[None, :] >= k[:, None]
        return torch.softmax(logits.masked_fill(mask, NEG), -1).cpu()

    def ask(self, context: str, questions: list[Question]) -> list[dict]:
        questions = [Question(q.primitive, context, q.question, q.choices, q.ordered).validate() for q in questions]
        probs = self._probs(questions).numpy()
        ks = np.array([q.k for q in questions])
        ent_conf = entropy_confidence(probs, ks)
        out = []
        for q, p, ec in zip(questions, probs, ent_conf):
            p = p[: q.k]
            best = int(p.argmax())
            item = {"probs": {c: float(x) for c, x in zip(q.choices, p)}, "value": q.choices[best],
                    "confidence": float(p[best]), "entropy_confidence": float(ec)}
            if q.primitive == "score":
                item["score"] = float((p * np.arange(q.k)).sum() / (q.k - 1))
            if q.primitive == "noul":
                item = {"p_true": float(p[0]), "confidence": float(max(p[0], 1 - p[0]))}
            out.append(item)
        return out

    def choice(self, context: str, question: str, choices: list[str]) -> dict:
        return self.ask(context, [Question("choice", context, question, choices)])[0]

    def score(self, context: str, question: str, levels: list[str]) -> dict:
        return self.ask(context, [Question("score", context, question, levels, ordered=True)])[0]

    def noul(self, context: str, proposition: str) -> dict:
        return self.ask(context, [Question("noul", context, f"Is this true: {proposition}", ["true", "false"])])[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_infer.py -v`
Expected: 3 passed.

- [ ] **Step 6: Amputate the RLCD model and run a demo**

```powershell
uv run python -m rlcd.amputate --src runs/rlcd --out runs/rlcd/decision
uv run python -c "from rlcd.infer import DecisionModel; from rlcd.schema import Question; dm=DecisionModel.load('runs/rlcd/decision'); ctx='Ticket #4411 from an enterprise, tier 1 customer. Report: API latency spiked to 4 seconds; charged twice this month. Note: production is down.'; import json; print(json.dumps(dm.ask(ctx,[Question('choice',ctx,'Which department should handle this ticket?',['BILLING','INFRASTRUCTURE','SECURITY','PRODUCT_SUPPORT']),Question('score',ctx,'What is the priority of this ticket?',['P3_LOW','P2_NORMAL','P1_HIGH','P0_CRITICAL'],ordered=True),Question('noul',ctx,'Should an on-call engineer be paged immediately?',['true','false'])]),indent=2))"
```
Expected: three JSON results. The department question should split probability between INFRASTRUCTURE and BILLING rather than committing to one, the priority score should sit near the top of the range, and `p_true` for paging should be high. Paste this output into the README.

- [ ] **Step 7: Commit**

```powershell
git add rlcd/amputate.py rlcd/infer.py tests/test_infer.py
git commit -m "Add head amputation and decision-only inference API"
```

---

### Task 11: README and full test pass

**Files:**
- Create: `README.md`
- Modify: nothing else

- [ ] **Step 1: Run the whole offline suite**

Run: `uv run pytest -v`
Expected: all tests pass (scaffold 2, schema 12, policy 4, rewards 8, metrics 7, data 8, loop 3, train_rl 2, eval 3, infer 3 = 52).

- [ ] **Step 2: Write README.md**

Write it in Anthony's voice (use the writing-in-anthonys-voice skill). Required sections, in this order:

1. Title `# eve-rlcd` and a two-sentence statement: a toy recreation of RLCD on a 272M model, the model returns decisions with probabilities and cannot emit text.
2. `## What RLCD is here`: the bandit setting, the reward `r = c - p_a`, one paragraph on why REINFORCE with that reward is an unbiased Brier gradient estimator (copy the argument from the design doc), and the three arms.
3. `## Results`: paste `runs/compare/table.md`, embed `runs/compare/reliability.png` and `runs/compare/coverage.png` (copy both into `docs/img/`), list the fitted temperatures from `runs/rlcd/temperature.json` and `runs/rlvr/temperature.json`, and state plainly whether the calibration hypothesis held. Include the demo output from Task 10 Step 6.
4. `## Reproduce`: the exact commands from Task 1 Step 2, Task 6 Step 5, Task 7 Step 8, Task 8 Step 6, Task 9 Steps 6 and 7, Task 10 Step 6, in order.
5. `## Limitations`: cardinality 26, no KV-cache broadcast, small model, labels from public datasets not operational outcomes, reference labels for `triage` are synthetic.
6. `## Credits`: the Substack post, the harshatheg parallel constrained decoding repo, Eve-2, Rewarding Doubt, Guo et al. 2017.

No emojis, no em dashes, arrows as `->`.

- [ ] **Step 3: Copy plots into the repo**

```powershell
New-Item -ItemType Directory -Force docs\img | Out-Null
Copy-Item runs\compare\reliability.png docs\img\reliability.png
Copy-Item runs\compare\coverage.png docs\img\coverage.png
```

- [ ] **Step 4: Commit**

```powershell
git add README.md docs/img
git commit -m "Add README with results and reproduction steps"
```

---

### Task 12: Model card (only if results hold up)

**Gate:** Do this task only if `runs/compare/table.md` shows the RLCD arm with lower ECE than the RLVR arm at comparable accuracy (within 3 points). If the hypothesis failed, skip to Task 13 and write the article about the negative result instead; do not publish a model.

**Files:**
- Create: `MODEL_CARD.md` (becomes `README.md` of the Hugging Face model repo)

- [ ] **Step 1: Write MODEL_CARD.md** using the writing-in-anthonys-voice skill. YAML front matter: `license: mit`, `base_model: anthonym21/Eve-2-MoE-IT-272M`, `tags: [moe, eve-moe, calibration, rlcd, decision-model, classification, custom_code]`, `language: [en]`, `datasets` listing the seven public sources. Sections in order: what the model is (a decision-only model, 26-row head, cannot emit text); the three primitives with one `DecisionModel` code example each; how it was trained (bandit feedback, reward `c - p_a`, warmup SFT, KL to reference, hyperparameters copied from `runs/rlcd/meta.json`); results table and both plots; the RLVR versus RLCD comparison stated plainly; limitations (272M, cardinality 26, synthetic triage labels, calibration only measured in-distribution, thresholds are coupled to this checkpoint); how to reproduce (link to the GitHub repo); credits; citation block. Every number must be copied from `runs/compare/metrics.json` or `runs/*/temperature.json`, never typed from memory.
- [ ] **Step 2: ASCII check** on the file, then commit with message `Add model card`.
- [ ] **Step 3: Do not upload.** Publishing to the Hugging Face hub is Anthony's call; leave the upload command in the final report.

### Task 13: Article, about 1500 words

**Files:**
- Create: `docs/article/2026-09-rlcd-toy.md`

- [ ] **Step 1: Draft** with the writing-in-anthonys-voice skill, Substack register (not the fast-typing X register). Target 1400 to 1600 words. It is a follow-up to "Jev: The Language Model That Won't Talk". Required beats: the previous post ended on "whether RLCD delivers is unproven"; TypeSafe published nothing, so here is a from-scratch guess at what RLCD could be; the bandit framing and why it makes this RL and not supervised learning; the reward `c - p_a` explained with the 90 percent right / 90 percent wrong / 30 percent right examples; the one-subtraction difference between RLVR and RLCD and what it does to the reliability curve; the actual results with both figures, including anything that did not work; the amputation step and the demo output; what this does and does not say about Jev (it says nothing about their method, it shows the objective is coherent and cheap to test); limitations; link to repo and model. Every number comes from the results files.
- [ ] **Step 2: Word count and ASCII check.** `(Get-Content docs\article\2026-09-rlcd-toy.md | Measure-Object -Word).Words` must land between 1400 and 1600.
- [ ] **Step 3: Commit** with message `Add article draft`.

---

## Self-review notes

- Spec coverage: policy and prompt (Task 2, 3), three primitives and NOTA (Task 2, 6, 10), bandit feedback and reward with the gradient-identity test (Task 4), RLVR/RLCD/oracle ablation (Task 8), warmup SFT (Task 7), temperature as a reference-only check (Task 9), amputation (Task 10), metrics including coverage-error and NOTA rate (Task 5, 9), datasets and synthetic triage (Task 6), right-padding rule (Task 3 test), aux loss in every objective (Tasks 7, 8), README (Task 11). Optional Qwen zero-shot reference from the spec is deliberately omitted from v1.
- Type consistency: `decision_logits` returns `(logits, aux)` everywhere; `BanditEnv.step(idx, actions)` takes `(B,)` and `(B,G)`; `summarize` returns `overall / by_primitive / by_source`; `Question` positional order is `(primitive, context, question, choices, ordered, answer, source, id)` and every positional construction follows it.
- Known judgment calls: `mean_conf` in training logs uses max probability; `entropy_confidence` is only exposed through `DecisionModel`. The `oracle` arm shares the RL loop (same LR, KL, epochs) for a fair comparison rather than the SFT script.
