import math
import random

import pytest
import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policy import decision_logits, questions_to_batch
from rlcd.quick_eval import evaluate, stride_sample
from rlcd.schema import Question


class FakeTok:
    """Maps each character to a token id; enough to drive the policy on CPU."""
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny_model() -> EveMoEForCausalLM:
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16,
                    block_size=128, num_experts=2, top_k=1,
                    expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg)


def _questions() -> list[Question]:
    qs = [Question("choice", "ctx", "pick one", ["x", "y", "z"], answer=i % 3, source="alpha")
          for i in range(5)]
    qs += [Question("noul", "ctx", "is it so", ["true", "false"], answer=i % 2, source="beta")
           for i in range(4)]
    return qs


def test_evaluate_reports_metrics_and_restores_train_mode():
    qs = _questions()
    model = tiny_model().train()
    out = evaluate(model, FakeTok(), list(range(100, 126)), qs, max_len=64, device="cpu")
    assert set(out) == {"eval_acc", "eval_nll", "eval_conf", "eval_pred_last", "eval_ece",
                        "eval_brier", "eval_entropy", "eval_acc_alpha", "eval_acc_beta"}
    for key in ("eval_acc", "eval_conf", "eval_pred_last", "eval_ece", "eval_acc_alpha", "eval_acc_beta"):
        assert 0.0 <= out[key] <= 1.0
    assert out["eval_nll"] > 0.0
    assert 0.0 <= out["eval_brier"] <= 2.0
    # Entropy is over the declared options only, so it cannot exceed log(3) on this set.
    assert 0.0 < out["eval_entropy"] <= math.log(3) + 1e-6
    assert all(isinstance(v, float) for v in out.values())
    assert model.training


def test_evaluate_restores_eval_mode_and_does_not_depend_on_batch_size():
    qs = _questions()
    model = tiny_model().eval()
    a = evaluate(model, FakeTok(), list(range(100, 126)), qs, max_len=64, batch_size=32, device="cpu")
    b = evaluate(model, FakeTok(), list(range(100, 126)), qs, max_len=64, batch_size=2, device="cpu")
    assert not model.training
    for key in a:
        assert abs(a[key] - b[key]) < 1e-5, key


def test_evaluate_matches_a_row_at_a_time_reference_and_leaves_no_grads():
    rng = random.Random(0)
    qs = []
    for i in range(45):  # batches of 32 and 13, with mixed k inside each
        k = rng.choice([2, 3, 5, 9])
        qs.append(Question("choice", f"context number {i} " * rng.randint(1, 4), "pick one",
                           [f"option {j}" for j in range(k)], answer=rng.randrange(k), source="mixed"))
    model = tiny_model().eval()
    letters = list(range(100, 126))
    nll, hits = [], []
    with torch.no_grad():
        for q in qs:
            ids, last, k = questions_to_batch(FakeTok(), [q], 128, "cpu")
            logp = torch.log_softmax(decision_logits(model, ids, last, letters, k)[0], -1)[0]
            nll.append(-logp[q.answer].item())
            hits.append(float(logp.argmax().item() == q.answer))
    out = evaluate(model, FakeTok(), letters, qs, max_len=128, device="cpu")
    assert abs(out["eval_nll"] - sum(nll) / len(nll)) < 1e-5
    assert out["eval_acc"] == sum(hits) / len(hits)
    assert all(p.grad is None for p in model.parameters())
    assert not model.training


def test_evaluate_restores_the_mode_when_the_forward_raises():
    class Boom:
        def encode(self, s, add_special_tokens=False):
            raise RuntimeError("tokenizer failed")

    model = tiny_model().train()
    with pytest.raises(RuntimeError):
        evaluate(model, Boom(), list(range(100, 126)), _questions(), max_len=64, device="cpu")
    assert model.training


def test_stride_sample_hits_every_block_evenly():
    items = [block for block in range(8) for _ in range(1000)]
    picked = stride_sample(items, 3000)
    assert len(picked) == 3000
    shares = [picked.count(block) / len(picked) for block in range(8)]
    assert min(shares) > 0.0
    assert max(shares) - min(shares) < 0.02
    assert stride_sample(list(range(100)), 10) == list(range(0, 100, 10))


def test_stride_sample_returns_everything_when_n_covers_the_list():
    rows = list(range(100))
    assert stride_sample(rows, 100) == rows
    assert stride_sample(rows, 500) == rows


def test_stride_sample_rejects_non_positive_n():
    for n in (0, -1):
        with pytest.raises(ValueError):
            stride_sample(list(range(10)), n)


def test_predict_logits_returns_first_k_fp32_logits_per_row():
    from rlcd.quick_eval import predict_logits
    qs = _questions()
    model = tiny_model().train()
    letters = list(range(100, 126))
    rows = predict_logits(model, FakeTok(), letters, qs, max_len=64, batch_size=4, device="cpu")
    assert model.training
    assert [len(r) for r in rows] == [q.k for q in qs]
    assert all(isinstance(v, float) for r in rows for v in r)
    model.eval()
    with torch.no_grad():
        for q, row in zip(qs, rows):
            ids, last, k = questions_to_batch(FakeTok(), [q], 64, "cpu")
            ref = decision_logits(model, ids, last, letters, k)[0][0, : q.k]
            assert torch.allclose(torch.tensor(row), ref, atol=1e-5)
    assert all(p.grad is None for p in model.parameters())
