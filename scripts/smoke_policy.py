"""Load a real base through load_policy and check the readout before spending GPU hours on it:
single-token letters, body-plus-letter-rows against the model's own full logits, padded batch
against single rows, and a rough zero-shot look.

    uv run python scripts/smoke_policy.py Qwen/Qwen3-0.6B-Base [--backend hf-decoder] [--lora]
"""
import argparse

import torch

from rlcd.policies import BACKENDS, load_policy
from rlcd.quick_eval import evaluate, stride_sample
from rlcd.schema import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--backend", choices=("auto",) + BACKENDS, default="auto")
ap.add_argument("--lora", action="store_true")
ap.add_argument("--data", default="data/val.jsonl")
ap.add_argument("--n", type=int, default=64)
args = ap.parse_args()

policy = load_policy(args.model, device="cuda", backend=args.backend, lora=args.lora).eval()
model = policy.model
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in policy.trainable_parameters())
print("backend", policy.name, "class", type(policy._hf_model()).__name__ if policy.name != "eve" else "Eve")
print(f"parameters {total / 1e6:.1f}M trainable {trainable / 1e6:.1f}M dtype {next(model.parameters()).dtype}")
print("letter ids", policy.letters)
print("letter tokens", [policy.tok.decode([i]) for i in policy.letters])
if policy.name != "eve":
    print("prepend_bos", policy.prepend_bos, "pad_id", policy.pad_id, "vocab", len(policy.tok))

qs = stride_sample(read_jsonl(args.data), args.n)
batch = qs[:8]
with torch.no_grad():
    got, _ = policy.decision_logits(batch, 512, "cuda")
    worst_single, worst_full = 0.0, 0.0
    for i, q in enumerate(batch):
        single, _ = policy.decision_logits([q], 512, "cuda")
        worst_single = max(worst_single, (single[0, : q.k] - got[i, : q.k]).abs().max().item())
        if policy.name != "eve":
            ids, mask, last = policy.encode([q], 512, "cuda")
            full = policy._hf_model()(input_ids=ids, attention_mask=mask).logits
            want = full[0, last[0], policy.letters].float()
            worst_full = max(worst_full, (want[: q.k] - got[i, : q.k]).abs().max().item())
print(f"fp32: max |batched - single| = {worst_single:.3e}, max |policy - full logits| = {worst_full:.3e}")
assert worst_single < 1e-3 and worst_full < 1e-3, "the readout does not match the model's own logits"

out = evaluate(policy, qs, 512)
print({k: round(v, 4) for k, v in out.items()})
print(f"peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")
