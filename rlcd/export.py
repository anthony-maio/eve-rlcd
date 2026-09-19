"""Decision-only export: the model body, the tokenizer, the 26 letter rows of the output head,
and a record of everything the decision API needs. No language-model head is saved, so the
directory is a decision model, not a text generator, as far as this code base is concerned.

    python -m rlcd.export --src runs/q-rlcd --out runs/q-rlcd/decision
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from rlcd.decide import CONFIDENCE_DEFINITIONS, NOUL_OPTIONS
from rlcd.policies import (HFDecoderPolicy, _adds_bos, _check_rope, _from_pretrained_strict, _load_tokenizer,
                           load_policy)
from rlcd.schema import LETTERS, MAX_CHOICES, letter_token_ids

DECISION_FILE = "decision.json"
HEAD_FILE = "decision_head.safetensors"
FORMAT = "rlcd-decision-only-v1"

PROMPT = {
    "prefix": "User: Context:\n{state}\n\n",
    "suffix": "Question: {question}\nOptions:\n{options}\nAnswer with the letter only.\nAssistant: The answer is",
    "option_line": "{letter}) {option}",
    "options_joiner": "\n",
    "state": "state.strip(), left-truncated to max_state_tokens tokens; the header always survives",
    "question": "question.strip(), left-truncated until the suffix fits max_question_tokens; the options block "
                "and the footer are never cut",
    "readout": "softmax over the 26 letter rows at the last prompt token, masked at and beyond the option count",
    "split": "the prefix is tokenized alone and prefilled once; every suffix is tokenized alone and run with "
             "positions continuing from the prefix length; this equals tokenizing and running the whole prompt",
}

PRIMITIVES = {
    "choice": {"fields": ["question", "options"], "options": "2..26 unordered options, one letter each",
               "answer": ["kind", "value", "probs", "confidence", "entropy_confidence"]},
    "score": {"fields": ["question", "levels"], "options": "2..26 ordered levels, low to high, one letter each",
              "answer": ["kind", "value", "probs", "confidence", "entropy_confidence", "score"]},
    "noul": {"fields": ["question"], "options": "a yes/no question rendered verbatim with the options "
             + repr(list(NOUL_OPTIONS)) + ", as the noul rows of the training data are",
             "answer": ["kind", "p_true", "confidence"]},
}

NOTE_TIED = ("{model} ties its output head to the input embedding, so the vocabulary projection is recoverable "
             "from the embedding; the export removes the generation path from the API, it does not make generation "
             "physically impossible.")
NOTE_UNTIED = ("{model} has a separate output head, of which only the 26 letter rows are exported; the export "
               "removes the generation path from the API and leaves the rest of the vocabulary projection behind, "
               "but the body can still be re-attached to a head by anyone who has one.")

README = """# Decision-only export

A decision model built from `{source}`: the transformer body, its tokenizer, the 26 letter rows of the
output head (`{head_file}`), and `{decision_file}` with the prompt template, the primitives, the
confidence definitions and the training summary. There is no language-model head in this directory and
no text is generated: the model reads a state and typed questions and returns probabilities over the
declared options.

Load it with `rlcd.decide.Decider.load("{out}")` and call `ask(state, questions)`. `decision.json`
records the sha256 of `model.safetensors` and `{head_file}`, and the loader verifies both.

Note: {note}

Tokenizer warning: under transformers 4.57 loading this tokenizer printed "The tokenizer you are loading
... with an incorrect regex pattern ... fix_mistral_regex". The warning was spurious for this tokenizer (it
is not a Mistral tokenizer) and transformers 5 no longer prints it; the encodings are the same under both
and are the ones the model was trained on.
"""


def _short_name(base: str | None) -> str:
    """The model name for the honesty note: Qwen3-0.6B for the Qwen3-0.6B lineage, else the id."""
    if base and "Qwen3-0.6B" in base:
        return "Qwen3-0.6B"
    return base or "this model"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_export(out: Path, body, tok, weight: torch.Tensor, bias: torch.Tensor | None, record: dict) -> dict:
    """Write everything and return the record with the hashes of the weight files filled in."""
    out.mkdir(parents=True, exist_ok=True)
    body.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    tensors = {"weight": weight.detach().to("cpu", torch.float32).contiguous()}
    if bias is not None:
        tensors["bias"] = bias.detach().to("cpu", torch.float32).contiguous()
    save_file(tensors, str(out / HEAD_FILE), metadata={"format": "pt"})
    record = dict(record, sha256={name: _sha256(out / name) for name in ("model.safetensors", HEAD_FILE)})
    (out / DECISION_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README.format(source=record["source"], head_file=HEAD_FILE,
                                                 decision_file=DECISION_FILE, out=out.as_posix(),
                                                 note=record["note"]), encoding="utf-8")
    return record


def export(src: str, out: str) -> dict:
    """Write the decision-only export of the checkpoint at src into out and return its record."""
    src, out_path = os.fspath(src), Path(out)
    policy = load_policy(src, device="cpu")
    if not isinstance(policy, HFDecoderPolicy):
        raise ValueError(f"{src} is a {policy.name} policy; the export supports hf-decoder checkpoints only")
    if policy.lora:
        raise ValueError(f"{src} is a LoRA adapter; merge it into a full checkpoint before exporting")
    policy.eval()
    model = policy.model
    head, embed = model.get_output_embeddings(), model.get_input_embeddings()
    tied = head is not None and embed is not None and head.weight.data_ptr() == embed.weight.data_ptr()
    weight, bias = policy.letter_head()
    record_in = _read_json(Path(src) / "policy.json")
    base = record_in.get("origin") or record_in.get("base") or (src if not os.path.isdir(src) else None)
    name = _short_name(base)
    record = {
        "format": FORMAT,
        "source": src,
        "base_model": base,
        "model_type": model.config.model_type,
        "hidden_size": int(weight.shape[1]),
        "letters": list(LETTERS),
        "letter_ids": list(policy.letters),
        "max_choices": MAX_CHOICES,
        "prepend_bos": policy.prepend_bos,
        "pad_id": policy.pad_id,
        "tied_embeddings": tied,
        "head_file": HEAD_FILE,
        "prompt": PROMPT,
        "primitives": PRIMITIVES,
        "confidence": CONFIDENCE_DEFINITIONS,
        "training": _read_json(Path(src) / "meta.json"),
        "policy": record_in,
        "note": (NOTE_TIED if tied else NOTE_UNTIED).format(model=name),
    }
    return _write_export(out_path, policy._body(), policy.tok, weight, bias, record)


class DecisionOnlyPolicy(HFDecoderPolicy):
    """An hf-decoder policy over an exported body: the model is the body itself and the head
    is the saved (26, d) letter matrix. decision_logits and the cached-prefix path work as on
    the full model."""
    name = "decision-only"

    def __init__(self, body, tok, weight: torch.Tensor, bias: torch.Tensor | None, record: dict):
        super().__init__(body, tok, letters=record["letter_ids"], origin=record.get("base_model"))
        self.weight, self.bias, self.record = weight, bias, record
        self.prepend_bos = bool(record["prepend_bos"])
        self.pad_id = record["pad_id"]

    def _body(self):
        return self.model

    def letter_head(self):
        return self.weight, self.bias

    def save(self, out_dir: str, meta: dict) -> None:
        _write_export(Path(out_dir), self.model, self.tok, self.weight, self.bias, self.record | meta)


def load_decision_only(path: str, device: str = "cuda") -> DecisionOnlyPolicy:
    """The policy of an export directory, fp32 on device, in eval mode; strict about every
    tensor and about the tokenizer still giving the recorded letter ids and BOS behaviour."""
    from transformers import AutoModel
    path = os.fspath(path)
    record = json.loads(Path(path, DECISION_FILE).read_text(encoding="utf-8"))
    if record.get("format") != FORMAT:
        raise RuntimeError(f"{path} has export format {record.get('format')!r}, expected {FORMAT!r}")
    for name, digest in record["sha256"].items():
        actual = _sha256(Path(path, name))
        if actual != digest:
            raise RuntimeError(f"{path}/{name} has sha256 {actual}, but {DECISION_FILE} records {digest}")
    tok = _load_tokenizer(path)
    if letter_token_ids(tok) != list(record["letter_ids"]):
        raise RuntimeError(f"{path}: the tokenizer no longer gives the recorded letter ids")
    if _adds_bos(tok) != bool(record["prepend_bos"]):
        raise RuntimeError(f"{path} was exported with prepend_bos={record['prepend_bos']}, but its tokenizer "
                           f"now gives {_adds_bos(tok)}")
    body = _from_pretrained_strict(AutoModel, path)
    if body.get_output_embeddings() is not None:
        raise RuntimeError(f"{path} loaded as {type(body).__name__}, which has an output head; expected a body")
    _check_rope(body.config)
    head = load_file(str(Path(path, record.get("head_file", HEAD_FILE))))
    weight = head["weight"]
    if weight.shape != (MAX_CHOICES, body.config.hidden_size):
        raise RuntimeError(f"{path}: decision head has shape {tuple(weight.shape)}, expected "
                           f"({MAX_CHOICES}, {body.config.hidden_size})")
    body.to(device).eval()
    bias = head["bias"].to(device) if "bias" in head else None
    return DecisionOnlyPolicy(body, tok, weight.to(device), bias, record).eval()


def main(argv=None):
    ap = argparse.ArgumentParser(description="write a decision-only export of a trained checkpoint")
    ap.add_argument("--src", required=True, help="checkpoint directory written by Policy.save")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    record = export(args.src, args.out)
    files = sorted(p for p in Path(args.out).iterdir() if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"wrote {args.out} ({total / 1e6:.1f} MB):")
    for p in files:
        print(f"  {p.name:<32} {p.stat().st_size / 1e6:9.1f} MB")
    print(f"note: {record['note']}")


if __name__ == "__main__":
    main()
