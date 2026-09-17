import copy

import torch

from rlcd import train_rl, train_sft
from rlcd.compat import eve_config
from rlcd.env import BanditEnv
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policies import EvePolicy
from rlcd.schema import Question
from rlcd.train_rl import rl_step, should_stop

STAT_KEYS = {"loss", "reward", "sampled_acc", "mean_conf", "kl", "aux", "policy_entropy", "p_taken"}


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny():
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=256,
                    num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EvePolicy(EveMoEForCausalLM(cfg), FakeTok(), list(range(100, 126)))


def _batch():
    return [Question("choice", "ctx one", "q", ["a", "b", "c"], answer=1),
            Question("noul", "ctx two", "q", ["true", "false"], answer=0)]


def test_rl_step_each_arm_produces_finite_loss_and_grads():
    policy = tiny().train()
    ref = copy.deepcopy(policy).eval()
    env = BanditEnv(torch.tensor([1, 0]))
    for arm in ("rlvr", "rlcd", "oracle"):
        policy.model.zero_grad(set_to_none=True)
        loss, stats = rl_step(policy, ref, _batch(), torch.tensor([0, 1]), env,
                              arm=arm, group=3, kl_coef=0.05, aux_coef=0.01, max_len=64, device="cpu")
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in policy.trainable_parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert set(stats) >= STAT_KEYS
        assert all(isinstance(v, float) for v in stats.values())
        assert ("nll" in stats) == (arm == "oracle")


def test_logged_loss_excludes_the_aux_term():
    policy = tiny().train()
    env = BanditEnv(torch.tensor([1, 0]))
    loss, stats = rl_step(policy, None, _batch(), torch.tensor([0, 1]), env,
                          arm="oracle", group=1, kl_coef=0.0, aux_coef=0.5, max_len=64, device="cpu")
    assert stats["aux"] > 0.0
    assert abs(stats["loss"] - stats["nll"]) < 1e-6
    assert abs(loss.item() - (stats["loss"] + 0.5 * stats["aux"])) < 1e-5


def test_bandit_arms_never_call_reveal(monkeypatch):
    policy = tiny().train()
    ref = copy.deepcopy(policy).eval()
    env = BanditEnv(torch.tensor([1, 0]))

    def boom(idx):
        raise AssertionError("reveal called by a bandit arm")

    monkeypatch.setattr(env, "reveal", boom)
    for arm in ("rlvr", "rlcd"):
        rl_step(policy, ref, _batch(), torch.tensor([0, 1]), env,
                arm=arm, group=2, kl_coef=0.0, aux_coef=0.0, max_len=64, device="cpu")


def test_rl_step_without_a_reference_model_reports_zero_kl():
    policy = tiny().train()
    env = BanditEnv(torch.tensor([1, 0]))
    for arm in ("rlvr", "rlcd", "oracle"):
        policy.model.zero_grad(set_to_none=True)
        loss, stats = rl_step(policy, None, _batch(), torch.tensor([0, 1]),
                              env, arm=arm, group=3, kl_coef=0.0, aux_coef=0.01, max_len=64, device="cpu")
        assert torch.isfinite(loss)
        loss.backward()
        assert stats["kl"] == 0.0
        assert 0.0 < stats["policy_entropy"] <= 1.1
        if arm != "oracle":
            assert 0.0 < stats["p_taken"] <= 1.0


def test_default_sft_and_rl_slices_are_disjoint_and_cover_the_file():
    ids = [f"row-{i}" for i in range(64000)]
    sft_args = train_sft.build_parser().parse_args([])
    rl_args = train_rl.build_parser().parse_args(["--arm", "rlcd", "--out", "unused"])
    warm = train_sft.select_slice(ids, sft_args.start, sft_args.n)
    rl = train_rl.select_slice(ids, rl_args.start, rl_args.limit)
    assert len(warm) == 32000 and len(rl) == 32000
    assert not set(warm) & set(rl)
    assert warm + rl == ids


def test_rl_slice_limit_counts_from_start():
    ids = list(range(100))
    assert train_rl.select_slice(ids, 40, 0) == list(range(40, 100))
    assert train_rl.select_slice(ids, 40, 10) == list(range(40, 50))


def test_rl_defaults_match_the_amended_task():
    args = train_rl.build_parser().parse_args(["--arm", "rlvr", "--out", "unused"])
    assert (args.kl, args.epochs, args.micro, args.accum, args.lr, args.group, args.max_len) == \
        (0.0, 2, 8, 16, 5e-5, 4, 512)
    assert args.no_save is False


def test_should_stop_needs_three_consecutive_bad_evals():
    base = 0.50
    assert not should_stop([0.50, 0.39, 0.39], base)
    assert should_stop([0.50, 0.39, 0.39, 0.39], base)
    assert not should_stop([0.50, 0.39, 0.39, 0.41, 0.39, 0.39], base)
    assert not should_stop([0.50, 0.40, 0.40, 0.40], base)   # exactly 0.10 below is not "more than"
    assert not should_stop([], base)


def test_rl_step_runs_on_the_hf_decoder_backend_with_zero_aux():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    from rlcd.policies import HFDecoderPolicy
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
    policy = HFDecoderPolicy(LlamaForCausalLM(cfg), AutoTokenizer.from_pretrained("gpt2")).train()
    env = BanditEnv(torch.tensor([1, 0]))
    for arm in ("rlvr", "rlcd", "oracle"):
        policy.model.zero_grad(set_to_none=True)
        loss, stats = rl_step(policy, None, _batch(), torch.tensor([0, 1]), env, arm=arm, group=3,
                              kl_coef=0.0, aux_coef=policy.aux_coef(), max_len=64, device="cpu")
        assert torch.isfinite(loss)
        loss.backward()
        assert stats["aux"] == 0.0
        assert any(p.grad is not None for p in policy.trainable_parameters())


def test_training_clis_take_a_backend_lora_and_grad_checkpointing():
    for parser, extra in ((train_sft.build_parser(), []),
                          (train_rl.build_parser(), ["--arm", "rlcd", "--out", "unused"])):
        args = parser.parse_args(extra)
        assert (args.backend, args.lora, args.grad_checkpointing) == ("auto", False, False)
        args = parser.parse_args(extra + ["--backend", "hf-decoder", "--lora", "--grad-checkpointing"])
        assert (args.backend, args.lora, args.grad_checkpointing) == ("hf-decoder", True, True)
