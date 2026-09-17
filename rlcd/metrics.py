"""Calibration and selective-prediction metrics."""
from __future__ import annotations

import numpy as np


def brier(probs: np.ndarray, answers: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(answers)), answers] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def _bin_ids(conf: np.ndarray, n_bins: int) -> np.ndarray:
    """Bins are [lo, hi) with 1.0 in the last bin. The epsilon keeps values that sit on an
    edge (0.3, 0.6, 0.7 with 10 bins) from falling into the lower bin through float error."""
    return np.clip(np.floor(conf * n_bins + 1e-9).astype(int), 0, n_bins - 1)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    ids = _bin_ids(conf, n_bins)
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
    """Error of the most confident fraction, for every fraction. Rows with equal confidence
    cannot be ranked against each other, so each takes the mean correctness of its tie group
    and the curve does not depend on input order."""
    conf = np.asarray(conf, float)
    order = np.argsort(-conf, kind="stable")
    c = np.asarray(correct, float)[order]
    _, group = np.unique(conf[order], return_inverse=True)
    c = (np.bincount(group, weights=c) / np.bincount(group))[group]
    n = len(c)
    covered = np.arange(1, n + 1)
    coverage = covered / n
    error = 1.0 - np.cumsum(c) / covered
    return coverage, error


def nota_rate(pred: np.ndarray, answers: np.ndarray, nota_index: np.ndarray) -> float:
    """Recall: among rows where NOTA is the answer, the fraction predicted NOTA."""
    pred, answers, nota_index = np.asarray(pred), np.asarray(answers), np.asarray(nota_index)
    is_nota_answer = (nota_index >= 0) & (answers == nota_index)
    if not is_nota_answer.any():
        return float("nan")
    return float((pred[is_nota_answer] == nota_index[is_nota_answer]).mean())


def nota_false_alarm(pred: np.ndarray, answers: np.ndarray, nota_index: np.ndarray) -> float:
    """Among rows where NOTA is offered but is not the answer, the fraction predicted NOTA.
    Read it next to nota_rate: a policy that always abstains scores 1.0 on both."""
    pred, answers, nota_index = np.asarray(pred), np.asarray(answers), np.asarray(nota_index)
    is_distractor = (nota_index >= 0) & (answers != nota_index)
    if not is_distractor.any():
        return float("nan")
    return float((pred[is_distractor] == nota_index[is_distractor]).mean())


def entropy_confidence(probs: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, float), 1e-12, 1.0)
    h = -(np.asarray(probs, float) * np.log(p)).sum(1)
    return np.clip(1.0 - h / np.log(np.asarray(k, float)), 0.0, 1.0)


def bootstrap_ci(fn, arrays: tuple, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 percent percentile bootstrap interval of fn(*arrays). Rows are resampled with
    replacement and the same row indices are applied to every array, so paired columns
    (confidence and correctness, say) stay paired."""
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if n == 0 or any(len(a) != n for a in arrays):
        raise ValueError("bootstrap_ci needs non-empty arrays of equal length")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(*(a[idx] for a in arrays))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
