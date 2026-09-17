import random
from collections import Counter

import pytest

from rlcd.data import (clean_labels, drop_empty_contexts, inject_nota, shuffle_choices,
                       split_source, subset_choices, synthetic_triage)
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


def test_subset_choices_full_alphabet():
    pool = [f"intent_{i}" for i in range(77)]
    choices, answer = subset_choices(pool, "intent_40", 26, random.Random(3))
    assert len(choices) == 26 and len(set(choices)) == 26
    assert choices[answer] == "intent_40"


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
            assert len(out.choices) == 3
            assert out.choices[-1] == NOTA
    assert 0.10 < kinds["nota_correct"] / 4000 < 0.20
    assert 0.30 < kinds["nota_distractor"] / 4000 < 0.40
    assert 0.45 < kinds["absent"] / 4000 < 0.55


def test_inject_nota_never_changes_the_option_count():
    q = _q(choices=["a", "b", "c", "d", "e"], answer=2)
    rng = random.Random(0)
    truth_present = truth_absent = 0
    for _ in range(2000):
        out = inject_nota(q, rng)
        out.validate()
        assert len(out.choices) == 5
        if NOTA in out.choices:
            assert out.choices[-1] == NOTA
            if "c" in out.choices:
                truth_present += 1
                assert out.choices[out.answer] == "c"
            else:
                truth_absent += 1
                assert out.choices[out.answer] == NOTA
    assert truth_present > 0 and truth_absent > 0


def test_inject_nota_two_choice_question():
    q = _q(choices=["x", "y"], answer=1)
    rng = random.Random(0)
    seen = set()
    for _ in range(300):
        out = inject_nota(q, rng).validate()
        assert len(out.choices) == 2
        seen.add((tuple(out.choices), out.answer))
    assert seen == {(("x", "y"), 1), (("y", NOTA), 0), (("x", NOTA), 1)}


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
    qs = [_q(id=str(i), context=f"c{i}") for i in range(100)]
    train, val, test = split_source(qs, per_source=50, n_val=20, n_test=20, rng=random.Random(0))
    assert len(train) == 50 and len(val) == 20 and len(test) == 20
    ids = [q.id for q in train + val + test]
    assert len(set(ids)) == 90


def test_inject_nota_on_score_keeps_levels_in_order():
    q = _q(primitive="score", ordered=True, choices=["low", "mid", "high"], answer=1)
    rng = random.Random(0)
    for _ in range(200):
        out = inject_nota(q, rng)
        levels = [c for c in out.choices if c != NOTA]
        assert levels in (["low", "mid", "high"], ["low", "high"], ["low", "mid"], ["mid", "high"])
        assert out.choices[out.answer] == ("mid" if "mid" in levels else NOTA)
        assert out.ordered and (NOTA not in out.choices or out.choices[-1] == NOTA)


def test_clean_labels_strips_and_flattens():
    assert clean_labels([" a_b ", "two\nlines", "c"], "src") == ["a_b", "two lines", "c"]


def test_clean_labels_rejects_duplicates_and_empties():
    with pytest.raises(ValueError, match="duplicate"):
        clean_labels(["x", " x "], "src")
    with pytest.raises(ValueError, match="empty"):
        clean_labels(["x", "  "], "src")


def test_drop_empty_contexts_counts_skips():
    qs = [_q(id="a"), _q(id="b", context="  \n "), _q(id="c", context="")]
    kept, n = drop_empty_contexts(qs)
    assert [q.id for q in kept] == ["a"] and n == 2


def test_split_source_keeps_each_context_in_one_split():
    qs = [_q(id=f"{c}-{j}", context=f"ctx {c}") for c in range(40) for j in range(3)]
    train, val, test = split_source(qs, per_source=60, n_val=20, n_test=20, rng=random.Random(0))
    assert len(train) == 60 and len(val) == 20 and len(test) == 20
    ids = [q.id for q in train + val + test]
    assert len(set(ids)) == len(ids)
    tr, va, te = ({q.context for q in part} for part in (train, val, test))
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_shuffle_choices_preserves_answer_text_and_spreads_positions():
    q = _q(choices=["a", "b", "c", "d"], answer=2)
    rng = random.Random(0)
    pos = Counter()
    for _ in range(2000):
        out = shuffle_choices(q, rng).validate()
        assert sorted(out.choices) == ["a", "b", "c", "d"]
        assert out.choices[out.answer] == "c"
        pos[out.answer] += 1
    assert all(pos[i] / 2000 >= 0.15 for i in range(4))


def test_shuffle_choices_keeps_nota_last():
    rng = random.Random(0)
    orders = set()
    for answer in (1, 3):
        q = _q(choices=["a", "b", "c", NOTA], answer=answer)
        for _ in range(200):
            out = shuffle_choices(q, rng).validate()
            assert out.choices[-1] == NOTA
            assert out.choices[out.answer] == q.choices[answer]
            orders.add(tuple(out.choices))
    assert len(orders) == 6


def test_shuffle_choices_leaves_score_and_noul_alone():
    score = _q(primitive="score", ordered=True, choices=["low", "mid", "high"], answer=0)
    noul = _q(primitive="noul", choices=["true", "false"], answer=1)
    ordered_choice = _q(ordered=True)
    rng = random.Random(0)
    for _ in range(50):
        assert shuffle_choices(score, rng) == score
        assert shuffle_choices(noul, rng) == noul
        assert shuffle_choices(ordered_choice, rng) == ordered_choice
