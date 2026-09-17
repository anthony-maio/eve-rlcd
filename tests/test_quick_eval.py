import math

import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
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


def test_stride_sample_covers_the_whole_file():
    rows = list(range(100))
    assert stride_sample(rows, 10) == list(range(0, 100, 10))
    assert stride_sample(rows, 100) == rows
    assert stride_sample(rows, 500) == rows
    picked = stride_sample(list(range(8000)), 2000)
    assert len(picked) == 2000 and picked[0] == 0 and picked[-1] >= 7996
    assert len(stride_sample(list(range(105)), 10)) == 10
