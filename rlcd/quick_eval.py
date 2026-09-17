"""Fast held-out evaluation shared by the SFT and RL training loops."""
from __future__ import annotations

import numpy as np
import torch

from rlcd.metrics import brier, ece
from rlcd.policy import decision_logits, questions_to_batch
from rlcd.schema import Question


def stride_sample(questions: list, n: int) -> list:
    """Every (len // n)-th row, at most n of them. Eval files are sorted by source, so a head
    slice would miss most sources. Returns all rows if there are n or fewer."""
    if n <= 0 or len(questions) <= n:
        return list(questions)
    return questions[:: len(questions) // n][:n]


def evaluate(model, tok, letters, questions: list[Question], max_len: int = 512, batch_size: int = 32,
             device: str = "cuda") -> dict:
    """Accuracy, NLL of the true answer, max-probability confidence, last-option rate, ECE
    (15 bins on max-probability confidence), Brier loss, and mean entropy in nats over the
    declared options, plus accuracy per source. Runs in eval mode without grad and restores
    the previous training mode."""
    device_type = torch.device(device).type
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            ids, last, k = questions_to_batch(tok, batch, max_len, device)
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                logits, _ = decision_logits(model, ids, last, letters, k)
            rows.append(torch.log_softmax(logits, -1).double().cpu().numpy())
    model.train(was_training)

    logp = np.concatenate(rows)
    p = np.exp(logp)
    answers = np.array([q.answer for q in questions])
    k = np.array([q.k for q in questions])
    pred = p.argmax(1)
    conf = p.max(1)
    hits = (pred == answers).astype(float)
    # Masked letters sit at a finite log-prob near NEG with exactly zero mass, so they add
    # nothing to the entropy.
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
