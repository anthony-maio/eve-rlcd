import numpy as np

from rlcd.metrics import (brier, coverage_error, ece, entropy_confidence, nota_false_alarm,
                          nota_rate, reliability_bins)


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


def test_bin_edges_go_to_the_upper_bin():
    conf = np.array([0.0, 0.1, 0.3, 0.6, 0.7, 0.9, 1.0])
    _, _, bn = reliability_bins(conf, np.ones(len(conf)), n_bins=10)
    expected = np.zeros(10, dtype=int)
    for b in [0, 1, 3, 6, 7, 9, 9]:
        expected[b] += 1
    assert bn.tolist() == expected.tolist()


def test_metrics_accept_plain_lists():
    conf = [0.95, 0.95, 0.55, 0.55]
    correct = [1, 1, 0, 0]
    assert abs(ece(conf, correct, n_bins=10) - 0.30) < 1e-9
    bc, ba, bn = reliability_bins(conf, correct, n_bins=10)
    assert bn[9] == 2 and bn[5] == 2
    assert abs(bc[9] - 0.95) < 1e-9 and ba[5] == 0.0


def test_brier_mixed_k_zero_padded():
    probs = np.array([[0.6, 0.4, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]])
    answers = np.array([0, 3])
    # row0 (k=2): .4^2 + .4^2 = .32 ; row1 (k=4): .01 + .04 + .09 + .36 = .50 ; mean .41
    assert abs(brier(probs, answers) - 0.41) < 1e-9


def test_ece_and_bins_with_confidence_exactly_one():
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([1, 1, 1, 0])
    bc, ba, bn = reliability_bins(conf, correct, n_bins=10)
    assert bn.tolist() == [0] * 9 + [4]
    assert bc[9] == 1.0 and ba[9] == 0.75
    assert abs(ece(conf, correct, n_bins=10) - 0.25) < 1e-9


def test_coverage_error_is_independent_of_order_within_ties():
    conf = np.array([1.0, 1.0, 1.0, 1.0, 0.5])
    cov_a, err_a = coverage_error(conf, np.array([1, 1, 0, 0, 1]))
    cov_b, err_b = coverage_error(conf, np.array([0, 0, 1, 1, 1]))
    assert np.allclose(cov_a, cov_b) and np.allclose(err_a, err_b)
    assert np.allclose(err_a[:4], 0.5)
    assert abs(err_a[4] - 0.4) < 1e-9


def test_nota_recall_and_false_alarm():
    #                     nota answer | nota distractor | no nota
    answers = np.array([3, 3, 0, 1, 1])
    nota_index = np.array([3, 3, 3, 3, -1])
    pred = np.array([3, 0, 3, 1, 1])
    assert abs(nota_rate(pred, answers, nota_index) - 0.5) < 1e-9
    assert abs(nota_false_alarm(pred, answers, nota_index) - 0.5) < 1e-9
    always_nota = np.where(nota_index >= 0, nota_index, 0)
    assert nota_rate(always_nota, answers, nota_index) == 1.0
    assert nota_false_alarm(always_nota, answers, nota_index) == 1.0
    never_nota = np.zeros(5, dtype=int)
    assert nota_rate(never_nota, answers, nota_index) == 0.0
    assert nota_false_alarm(never_nota, answers, nota_index) == 0.0
    assert np.isnan(nota_false_alarm(pred, answers, np.full(5, -1)))
    assert np.isnan(nota_false_alarm(np.array([3]), np.array([3]), np.array([3])))


def test_nota_metrics_accept_plain_lists():
    assert nota_rate([3, 0], [3, 3], [3, 3]) == 0.5
    assert nota_false_alarm([3, 0], [0, 0], [3, 3]) == 0.5


def test_entropy_confidence_intermediate_value():
    out = entropy_confidence(np.array([[0.5, 0.25, 0.25, 0.0]]), np.array([3]))
    # H = 1.0397 nats, log 3 = 1.0986
    assert abs(out[0] - 0.0536) < 1e-3


def test_entropy_confidence_is_clipped_to_unit_interval():
    probs = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]])
    out = entropy_confidence(probs, np.array([2, 4]))  # first row: k understates the support
    assert out[0] == 0.0 and out[1] == 1.0
    third = np.full((1, 3), 1 / 3)
    assert 0.0 <= entropy_confidence(third, np.array([3]))[0] <= 1.0


def test_bootstrap_ci_constant_statistic_has_zero_width():
    from rlcd.metrics import bootstrap_ci
    lo, hi = bootstrap_ci(lambda x: 0.25, (np.arange(50.0),))
    assert lo == hi == 0.25


def test_bootstrap_ci_covers_the_sample_mean_and_is_reproducible():
    from rlcd.metrics import bootstrap_ci
    x = np.random.default_rng(1).normal(loc=3.0, scale=2.0, size=400)
    lo, hi = bootstrap_ci(lambda a: float(a.mean()), (x,))
    assert lo < x.mean() < hi
    # 95 percent interval of a mean is about +-1.96 * sd / sqrt(n) = +-0.196 here.
    assert 0.25 < hi - lo < 0.55
    assert (lo, hi) == bootstrap_ci(lambda a: float(a.mean()), (x,))
    assert (lo, hi) != bootstrap_ci(lambda a: float(a.mean()), (x,), seed=1)


def test_bootstrap_ci_resamples_rows_jointly():
    from rlcd.metrics import bootstrap_ci
    a = np.arange(200.0)
    # The statistic is zero only if both arrays are indexed with the same resampled rows.
    lo, hi = bootstrap_ci(lambda x, y: float(np.abs(x - y).max()), (a, a.copy()), n_boot=50)
    assert lo == hi == 0.0


def test_bootstrap_stats_feeds_bootstrap_ci():
    from rlcd.metrics import bootstrap_ci, bootstrap_stats
    x = np.random.default_rng(2).normal(size=300)
    stats = bootstrap_stats(lambda a: float(a.mean()), (x,), n_boot=200, seed=3)
    assert stats.shape == (200,)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    assert (float(lo), float(hi)) == bootstrap_ci(lambda a: float(a.mean()), (x,), n_boot=200, seed=3)


def test_paired_bootstrap_diff_is_exactly_zero_for_identical_runs():
    from rlcd.metrics import ece, paired_bootstrap_diff
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, size=500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    assert paired_bootstrap_diff(lambda c: float(c.mean()), (correct,), (correct.copy(),)) == (0.0, 0.0, 0.0)
    assert paired_bootstrap_diff(lambda f, c: ece(f, c, 15), (conf, correct),
                                 (conf.copy(), correct.copy())) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_diff_sign_and_pairing():
    from rlcd.metrics import bootstrap_ci, paired_bootstrap_diff
    rng = np.random.default_rng(1)
    b = (rng.uniform(size=1000) < 0.5).astype(float)
    a = b.copy()
    wrong = np.flatnonzero(b == 0)
    a[wrong[:50]] = 1.0          # a is right on every row b is right on, plus 50 more
    mean = lambda c: float(c.mean())  # noqa: E731
    diff, lo, hi = paired_bootstrap_diff(mean, (a,), (b,))
    assert abs(diff - 0.05) < 1e-12
    assert 0.0 < lo < diff < hi
    # b minus a flips the sign of everything.
    rdiff, rlo, rhi = paired_bootstrap_diff(mean, (b,), (a,))
    assert abs(rdiff + 0.05) < 1e-12 and rlo < rdiff < rhi < 0.0
    # Pairing is the point: the interval is far tighter than the unpaired intervals would allow.
    alo, ahi = bootstrap_ci(mean, (a,))
    assert hi - lo < 0.5 * (ahi - alo)
    assert (diff, lo, hi) == paired_bootstrap_diff(mean, (a,), (b,))


def test_paired_bootstrap_diff_rejects_unequal_lengths():
    import pytest

    from rlcd.metrics import paired_bootstrap_diff
    with pytest.raises(ValueError):
        paired_bootstrap_diff(lambda c: float(c.mean()), (np.ones(5),), (np.ones(6),))
