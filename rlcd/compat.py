"""Workarounds for incompatibilities between the vendored Eve code and current transformers.

EveConfig names its MoE routing field `top_k`. EveConfig.__init__ sets `self.top_k` first,
then PretrainedConfig.__init__ (transformers 4.x) resets `self.top_k` to its legacy
generation default 50, silently clobbering the routing value. Always build and load Eve
configs through these helpers so routing uses the real value."""
from __future__ import annotations

import json
import os

from huggingface_hub import hf_hub_download

from rlcd.eve.configuration_eve import EveConfig

DEFAULT_TOP_K = 2


def eve_config(**kwargs) -> EveConfig:
    """Construct an EveConfig and re-apply the routing top_k after __init__."""
    top_k = kwargs.get("top_k", DEFAULT_TOP_K)
    config = EveConfig(**kwargs)
    config.top_k = top_k
    return config


def load_eve_config(path_or_id: str | os.PathLike) -> EveConfig:
    """Load an EveConfig from a local directory or hub id with the routing top_k restored
    from the raw config.json."""
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
