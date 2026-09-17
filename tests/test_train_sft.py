import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.schema import Question
from rlcd.train_sft import evaluate


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


def test_evaluate_reports_metrics_and_restores_train_mode():
    qs = [Question("choice", "ctx", "pick one", ["x", "y", "z"], answer=i % 3, source="alpha")
          for i in range(5)]
    qs += [Question("noul", "ctx", "is it so", ["true", "false"], answer=i % 2, source="beta")
           for i in range(4)]
    model = tiny_model().train()
    out = evaluate(model, FakeTok(), list(range(100, 126)), qs, max_len=64)
    assert set(out) == {"eval_acc", "eval_nll", "eval_conf", "eval_pred_last",
                        "eval_acc_alpha", "eval_acc_beta"}
    for key in ("eval_acc", "eval_conf", "eval_pred_last", "eval_acc_alpha", "eval_acc_beta"):
        assert 0.0 <= out[key] <= 1.0
    assert out["eval_nll"] > 0.0
    assert all(isinstance(v, float) for v in out.values())
    assert model.training
