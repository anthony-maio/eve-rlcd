import json

import torch

from rlcd.compat import eve_config, load_eve_config
from rlcd.eve.configuration_eve import EveConfig
from rlcd.eve.modeling_eve import EveMoEForCausalLM

TINY_KWARGS = dict(
    vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16,
    block_size=128, num_experts=2, top_k=1,
    expert_intermediate_size=64, shared_expert_intermediate_size=64,
)


def tiny_config() -> EveConfig:
    return eve_config(**TINY_KWARGS)


def test_tiny_eve_forward_shape():
    torch.manual_seed(0)
    model = EveMoEForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, 50304, (2, 7))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 7, 50304)


def test_tiny_eve_is_tied():
    model = EveMoEForCausalLM(tiny_config())
    assert model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def test_eve_config_preserves_routing_top_k():
    assert eve_config(**TINY_KWARGS).top_k == 1
    assert eve_config().top_k == 2


def test_load_eve_config_restores_top_k_from_disk(tmp_path):
    tiny_config().save_pretrained(tmp_path)
    with open(tmp_path / "config.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["top_k"] == 1
    assert load_eve_config(str(tmp_path)).top_k == 1
