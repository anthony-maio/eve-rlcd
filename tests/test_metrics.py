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
