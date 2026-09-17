"""The sliced-softmax policy over Eve-2: one forward pass, 26 letter logits."""
from __future__ import annotations

import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from rlcd.compat import load_eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.schema import MAX_CHOICES, NEG, Question, render_prompt

EVE_ID = "anthonym21/Eve-2-MoE-IT-272M"
PAD_ID = 50256


def _checkpoint_file(path_or_id: str) -> str:
    if os.path.isdir(path_or_id):
        path = os.path.join(path_or_id, "model.safetensors")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"no model.safetensors in {path_or_id}")
        return path
    return hf_hub_download(path_or_id, "model.safetensors")


def _load_weights(model: EveMoEForCausalLM, path_or_id: str) -> None:
    """Copy the checkpoint into the model and tie lm_head to wte.

    Read the safetensors file directly instead of going through from_pretrained. The hub
    file carries no "format" metadata, which transformers refuses, and from_pretrained
    only warns about missing weights where this raises. Eve ties wte and lm_head and a
    checkpoint stores only one of them: the hub file keeps lm_head.weight, save_pretrained
    keeps transformer.wte.weight. Accept either."""
    sd = load_file(_checkpoint_file(path_or_id))
    ref = sd.get("lm_head.weight", sd.get("transformer.wte.weight"))
    if ref is None:
        raise RuntimeError("checkpoint has neither lm_head.weight nor transformer.wte.weight")
    sd["lm_head.weight"] = sd["transformer.wte.weight"] = ref
    model.lm_head.weight = model.transformer.wte.weight
    result = model.load_state_dict(sd, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"checkpoint does not match the model: missing {result.missing_keys}, "
                           f"unexpected {result.unexpected_keys}")
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise RuntimeError("failed to tie lm_head to wte")


def load_eve(path_or_id: str = EVE_ID, device: str = "cuda"):
    """Eve and its tokenizer from a hub id or a local checkpoint directory, fp32 on device."""
    path_or_id = os.fspath(path_or_id)
    tokenizer = AutoTokenizer.from_pretrained(path_or_id)
    model = EveMoEForCausalLM(load_eve_config(path_or_id))
    if model.config.top_k != model.transformer.h[0].mlp.top_k:
        raise RuntimeError("MoE routing top_k mismatch between config and built model")
    _load_weights(model, path_or_id)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise RuntimeError("Eve weights must stay fp32")
    model.to(device)
    return model, tokenizer


def encode_batch(tokenizer, prompts: list[str], max_len: int = 512, device: str = "cpu"):
    seqs = [tokenizer.encode(p, add_special_tokens=False)[-max_len:] for p in prompts]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
    last = torch.zeros(len(seqs), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        last[i] = len(s) - 1
    return ids.to(device), last.to(device)


def questions_to_batch(tokenizer, questions: list[Question], max_len: int = 512, device: str = "cpu"):
    ids, last = encode_batch(tokenizer, [render_prompt(q) for q in questions], max_len, device)
    k = torch.tensor([q.k for q in questions], dtype=torch.long, device=device)
    return ids, last, k


def eve_hidden(model: EveMoEForCausalLM, input_ids: torch.Tensor):
    """Final hidden states (B,T,d) and the summed router aux loss."""
    x = model.transformer.wte(input_ids)
    aux = torch.zeros((), device=x.device, dtype=torch.float32)
    for block in model.transformer.h:
        x, block_aux = block(x, model.freqs_cis)
        aux = aux + block_aux.float()
    return model.transformer.ln_f(x), aux


def decision_logits(model, input_ids, last_idx, letter_ids: list[int], k: torch.Tensor):
    """fp32 logits over the 26 letters at each row's decision position.
    Positions at or beyond k are set to NEG so their softmax mass is exactly zero."""
    hidden, aux = eve_hidden(model, input_ids)
    rows = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
    weight = model.lm_head.weight[torch.as_tensor(letter_ids, device=hidden.device)]
    logits = rows.float() @ weight.float().t()
    positions = torch.arange(MAX_CHOICES, device=logits.device)[None, :]
    mask = positions >= k.to(logits.device)[:, None]
    return logits.masked_fill(mask, NEG), aux
