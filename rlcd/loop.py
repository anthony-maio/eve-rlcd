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
    """Model and tokenizer go in the same directory: load_eve reads both from there."""
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
