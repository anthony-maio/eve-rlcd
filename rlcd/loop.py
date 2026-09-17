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


def plan_steps(n_rows: int, micro: int, accum: int, epochs: int) -> int:
    """Optimizer steps in a run. Accumulation windows carry across epoch boundaries, so count
    micro-batches over the whole run first; only the very last window can be partial."""
    n_micro = epochs * ((n_rows + micro - 1) // micro)
    return (n_micro + accum - 1) // accum


class GradWindow:
    """Gradient accumulation whose result is the mean gradient over the EXAMPLES in the window,
    whatever the number of micro-batches and however short the last micro-batch is. Each
    micro-batch loss must be a mean over its rows: backward() weights it by its row count and
    finish() divides the summed gradients by the window's row count."""

    def __init__(self):
        self.micro = 0
        self.examples = 0

    def backward(self, mean_loss, n_examples: int) -> None:
        (mean_loss * n_examples).backward()
        self.micro += 1
        self.examples += n_examples

    def finish(self, parameters) -> int:
        """Rescale the accumulated grads in place, reset, and return the window's row count."""
        if self.examples == 0:
            raise ValueError("finish() on an empty accumulation window")
        n = self.examples
        for p in parameters:
            if p.grad is not None:
                p.grad.div_(n)
        self.micro, self.examples = 0, 0
        return n


def slice_meta(questions: list, start: int) -> dict:
    """Which rows of the training file a run used, for meta.json."""
    return {"slice_start": start, "slice_rows": len(questions),
            "slice_first_id": questions[0].id if questions else None,
            "slice_last_id": questions[-1].id if questions else None}


def save_checkpoint(model, tokenizer, out_dir: str, meta: dict) -> None:
    """Model and tokenizer go in the same directory: load_eve reads both from there."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


class JsonlLogger:
    def __init__(self, path, overwrite: bool = False):
        """Starts an empty log. Refuses to wipe an existing one unless overwrite is set, so a
        rerun cannot silently destroy a finished curve."""
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} already exists; pass --overwrite to replace it")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, **kw) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw) + "\n")
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()))
