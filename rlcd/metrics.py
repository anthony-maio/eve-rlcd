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


def _row_aligned(arrays: tuple) -> tuple:
    arrays = tuple(np.asarray(a) for a in arrays)
    if not arrays or len(arrays[0]) == 0 or any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("the bootstrap needs non-empty arrays of equal length")
    return arrays


def _bootstrap_groups(groups, n: int) -> tuple[np.ndarray, int] | None:
    if groups is None:
        return None
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(groups) != n:
        raise ValueError("bootstrap groups must be one-dimensional and aligned with the rows")
    _, inverse = np.unique(groups, return_inverse=True)
    return inverse, int(inverse.max()) + 1


def _bootstrap_indices(n: int, rng: np.random.Generator,
                       group_info: tuple[np.ndarray, int] | None) -> np.ndarray:
    if group_info is None:
        return rng.integers(0, n, n)
    inverse, n_groups = group_info
    multiplicity = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
    return np.repeat(np.arange(n), multiplicity[inverse])


def bootstrap_stats(fn, arrays: tuple, n_boot: int = 1000, seed: int = 0, groups=None) -> np.ndarray:
    """fn(*arrays) on n_boot bootstrap resamples. Rows are sampled by default. When groups are
    provided, whole groups are drawn with replacement so correlated rows stay together. The same
    sampled indices are applied to every array."""
    arrays = _row_aligned(arrays)
    n = len(arrays[0])
    group_info = _bootstrap_groups(groups, n)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = _bootstrap_indices(n, rng, group_info)
        stats[b] = fn(*(a[idx] for a in arrays))
    return stats


def bootstrap_ci(fn, arrays: tuple, n_boot: int = 1000, seed: int = 0, groups=None) -> tuple[float, float]:
    """95 percent percentile bootstrap interval of fn(*arrays)."""
    lo, hi = np.percentile(bootstrap_stats(fn, arrays, n_boot, seed, groups), [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_diff(stat_fn, arrays_a: tuple, arrays_b: tuple, n_boot: int = 1000,
                          seed: int = 0, groups=None) -> tuple[float, float, float]:
    """stat_fn(*arrays_a) minus stat_fn(*arrays_b) with a 95 percent percentile interval, for two
    runs scored on the same rows in the same order. Every resample draws one set of rows or whole
    groups and applies it to both runs, so agreement between the runs cancels out of the difference
    instead of widening its interval. Identical runs give exactly (0, 0, 0)."""
    both = _row_aligned(tuple(arrays_a) + tuple(arrays_b))
    a, b = both[: len(arrays_a)], both[len(arrays_a):]
    if not a or not b:
        raise ValueError("paired_bootstrap_diff needs arrays for both runs")
    n = len(a[0])
    group_info = _bootstrap_groups(groups, n)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _bootstrap_indices(n, rng, group_info)
        diffs[i] = stat_fn(*(x[idx] for x in a)) - stat_fn(*(x[idx] for x in b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(stat_fn(*a) - stat_fn(*b)), float(lo), float(hi)
