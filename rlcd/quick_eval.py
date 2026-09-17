"""Fast held-out evaluation shared by the SFT and RL training loops."""
from __future__ import annotations

import numpy as np
import torch

from rlcd.metrics import brier, ece
from rlcd.schema import NEG, Question


def stride_sample(items: list, n: int) -> list:
    """min(n, len) items spread evenly over the whole list. Eval files are sorted by source, so
    a head slice (or a fixed integer stride followed by truncation) would miss sources."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    m = min(n, len(items))
    return [items[i * len(items) // m] for i in range(m)]


def predict_logits(policy, questions: list[Question], max_len: int = 512,
                   batch_size: int = 32, device: str = "cuda") -> list[list[float]]:
    """The first k fp32 decision logits of every question, in input order. Runs in eval mode
    without grad and restores the previous training mode, also when the forward raises."""
    device_type = torch.device(device).type
    was_training = policy.training
    policy.eval()
    rows: list[list[float]] = []
    try:
        with torch.no_grad():
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                    logits, _ = policy.decision_logits(batch, max_len, device)
                for q, row in zip(batch, logits.float().cpu()):
                    rows.append(row[: q.k].tolist())
    finally:
        policy.train(was_training)
    return rows


def log_probs(rows: list[list[float]], temperature: float = 1.0) -> np.ndarray:
    """Row-wise log-softmax of variable-length logit rows at the given temperature, as one
    (n, max k) float64 array. Padding holds NEG, so exp() of it is exactly zero mass."""
    out = np.full((len(rows), max(len(r) for r in rows)), NEG)
    for i, r in enumerate(rows):
        z = np.asarray(r, float) / temperature
        z = z - z.max()
        out[i, : len(r)] = z - np.log(np.exp(z).sum())
    return out


def evaluate(policy, questions: list[Question], max_len: int = 512, batch_size: int = 32,
             device: str = "cuda") -> dict:
    """Accuracy, NLL of the true answer, max-probability confidence, last-option rate, ECE
    (15 bins on max-probability confidence), Brier loss, and mean entropy in nats over the
    declared options, plus accuracy per source. Runs in eval mode without grad and restores
    the previous training mode."""
    logp = log_probs(predict_logits(policy, questions, max_len, batch_size, device))
    p = np.exp(logp)
    answers = np.array([q.answer for q in questions])
    k = np.array([q.k for q in questions])
    pred = p.argmax(1)
    conf = p.max(1)
    hits = (pred == answers).astype(float)
    # Padding sits at the finite log-prob NEG with exactly zero mass, so it adds nothing to
    # the entropy.
    entropy = -(p * logp).sum(1)
    out = {"eval_acc": float(hits.mean()),
           "eval_nll": float(-logp[np.arange(len(answers)), answers].mean()),
           "eval_conf": float(conf.mean()),
           "eval_pred_last": float((pred == k - 1).mean()),
           "eval_ece": ece(conf, hits, n_bins=15),
           "eval_brier": brier(p, answers),
           "eval_entropy": float(entropy.mean())}
    sources = np.array([q.source for q in questions])
    for src in sorted(set(sources.tolist())):
        out[f"eval_acc_{src}"] = float(hits[sources == src].mean())
    return out
