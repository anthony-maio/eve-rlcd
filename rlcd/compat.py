"""Workarounds for incompatibilities between the vendored Eve code and current transformers.

Always build and load Eve configs through these helpers, and import this module before
building an EveMoEForCausalLM.

Tied weights: the vendored model declares `_tied_weights_keys = ["lm_head.weight"]`, the
transformers 4.x list form. transformers 5 expects a `{target: source}` mapping and raises on
the list when post_init ties the weights. The mapping is installed on the class here, since
the vendored files are not edited.

The routing `top_k`: under transformers 4.x PretrainedConfig.__init__ reset `self.top_k` to
its legacy generation default 50 after EveConfig.__init__ had set the routing value.
transformers 5 keeps no generation defaults on the config, so `EveConfig(top_k=1).top_k` is 1
again; the helpers still re-apply the value from the caller or the raw config.json, so the
routing width can never silently depend on the transformers version."""
from __future__ import annotations

import json
import os

from huggingface_hub import hf_hub_download

from rlcd.eve.configuration_eve import EveConfig
from rlcd.eve.modeling_eve import EveMoEForCausalLM

DEFAULT_TOP_K = 2
TIED_WEIGHTS = {"lm_head.weight": "transformer.wte.weight"}

if not isinstance(EveMoEForCausalLM._tied_weights_keys, dict):
    EveMoEForCausalLM._tied_weights_keys = dict(TIED_WEIGHTS)


def eve_config(**kwargs) -> EveConfig:
    """Construct an EveConfig with the routing top_k guaranteed to be the one asked for."""
    top_k = kwargs.get("top_k", DEFAULT_TOP_K)
    config = EveConfig(**kwargs)
    config.top_k = top_k
    return config


def load_eve_config(path_or_id: str | os.PathLike) -> EveConfig:
    """Load an EveConfig from a local directory or hub id with the routing top_k taken from
    the raw config.json."""
    path_or_id = os.fspath(path_or_id)
    if os.path.isdir(path_or_id):
        config_file = os.path.join(path_or_id, "config.json")
    else:
        config_file = hf_hub_download(path_or_id, "config.json")
    with open(config_file, encoding="utf-8") as f:
        raw = json.load(f)
    config = EveConfig.from_pretrained(path_or_id)
    config.top_k = raw.get("top_k", DEFAULT_TOP_K)
    return config
