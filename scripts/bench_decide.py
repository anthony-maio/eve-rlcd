"""Latency of ask() against ask_sequential() on one GPU, bf16 autocast, medians of 7 runs after 2
warmups: (a) 1..64 questions against one 800-token state, (b) state length 200, 800, 1500 tokens
at 8 questions, (c) option count 2, 5, 10, 26 at 8 questions.

Writes <out>/latency.json, <out>/latency.md and <out>/latency.png.

    uv run python scripts/bench_decide.py --model runs/q-rlcd --out runs/decide-bench
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from rlcd.data import DEPARTMENTS, DEPT_CUES, PRIO_CUES, PRIORITIES, synthetic_triage
from rlcd.decide import ChoiceQ, Decider

STATE_TOKENS = (200, 800, 1500)
QUESTION_COUNTS = (1, 2, 4, 8, 16, 32, 64)
OPTION_COUNTS = (2, 5, 10, 26)


def make_state(tok, n_tokens: int, seed: int = 0) -> str:
    """A ticket log in the style of the synthetic triage set, cut to exactly n_tokens tokens."""
    rng = random.Random(seed)
    lines = [q.context for q in synthetic_triage(n_tokens // 8 + 8, rng)[0::3]]
    text = "\n".join(lines)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < n_tokens:
        raise ValueError(f"only {len(ids)} tokens of state text for {n_tokens}")
    state = tok.decode(ids[:n_tokens])
    assert len(tok.encode(state, add_special_tokens=False)) == n_tokens
    return state


def make_questions(m: int, k: int, seed: int = 0) -> list[ChoiceQ]:
    """m distinct choice questions with k options each, in the style of the triage questions."""
    rng = random.Random(seed)
    cues = [c for phrases in DEPT_CUES.values() for c in phrases] + [c for phrases in PRIO_CUES.values() for c in phrases]
    labels = DEPARTMENTS + PRIORITIES + [f"TEAM_{i}" for i in range(26)]
    out = []
    for i in range(m):
        options = [labels[j] for j in rng.sample(range(len(labels)), k)]
        out.append(ChoiceQ(f"Ticket {1000 + i}: which team should handle '{rng.choice(cues)}'?", options))
    return out


def timed(fn, warmups: int = 2, runs: int = 7) -> float:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000.0


def bench(decider: Decider, tok) -> dict:
    results = {"questions": [], "state_tokens": [], "options": []}
    state = make_state(tok, 800)
    for m in QUESTION_COUNTS:
        qs = make_questions(m, 4)
        ask = timed(lambda: decider.ask(state, qs))
        seq = timed(lambda: decider.ask_sequential(state, qs))
        results["questions"].append({"m": m, "ask_ms": ask, "sequential_ms": seq})
        print(f"M={m:<3} k=4  state=800   ask {ask:8.1f} ms   sequential {seq:8.1f} ms   speedup {seq / ask:5.2f}x")
    qs = make_questions(8, 4)
    for n in STATE_TOKENS:
        s = make_state(tok, n)
        ask = timed(lambda: decider.ask(s, qs))
        seq = timed(lambda: decider.ask_sequential(s, qs))
        results["state_tokens"].append({"state_tokens": n, "ask_ms": ask, "sequential_ms": seq})
        print(f"M=8   k=4  state={n:<5} ask {ask:8.1f} ms   sequential {seq:8.1f} ms   speedup {seq / ask:5.2f}x")
    for k in OPTION_COUNTS:
        qs = make_questions(8, k)
        ask = timed(lambda: decider.ask(state, qs))
        seq = timed(lambda: decider.ask_sequential(state, qs))
        results["options"].append({"k": k, "ask_ms": ask, "sequential_ms": seq})
        print(f"M=8   k={k:<2} state=800   ask {ask:8.1f} ms   sequential {seq:8.1f} ms   speedup {seq / ask:5.2f}x")
    return results


def table(results: dict, device: str) -> str:
    lines = [f"Latency of the decision API on {device}, bf16 autocast, fp32 head. Medians of 7 runs after 2 warmups, "
             "in milliseconds. ask: one prefill of the state, one batched forward over the question suffixes. "
             "sequential: every full prompt through the training-time path, right-padded in batches of 32.", ""]
    lines += ["**(a) question count, one 800-token state, 4 options each**", "",
              "| questions | ask (ms) | sequential (ms) | speedup |", "|---|---|---|---|"]
    for r in results["questions"]:
        lines.append(f"| {r['m']} | {r['ask_ms']:.1f} | {r['sequential_ms']:.1f} | {r['sequential_ms'] / r['ask_ms']:.2f}x |")
    lines += ["", "**(b) state length, 8 questions, 4 options each**", "",
              "| state tokens | ask (ms) | sequential (ms) | speedup |", "|---|---|---|---|"]
    for r in results["state_tokens"]:
        lines.append(f"| {r['state_tokens']} | {r['ask_ms']:.1f} | {r['sequential_ms']:.1f} | "
                     f"{r['sequential_ms'] / r['ask_ms']:.2f}x |")
    lines += ["", "**(c) option count, 8 questions, one 800-token state**", "",
              "| options | ask (ms) | sequential (ms) | speedup |", "|---|---|---|---|"]
    for r in results["options"]:
        lines.append(f"| {r['k']} | {r['ask_ms']:.1f} | {r['sequential_ms']:.1f} | {r['sequential_ms'] / r['ask_ms']:.2f}x |")
    return "\n".join(lines) + "\n"


def plot(results: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("questions", "m", "questions (one 800-token state, 4 options)"),
              ("state_tokens", "state_tokens", "state tokens (8 questions, 4 options)"),
              ("options", "k", "options per question (8 questions, 800-token state)")]
    for ax, (key, xkey, label) in zip(axes, panels):
        rows = results[key]
        x = [r[xkey] for r in rows]
        ax.plot(x, [r["ask_ms"] for r in rows], marker="o", color="#2a78d6", label="ask (shared prefix)")
        ax.plot(x, [r["sequential_ms"] for r in rows], marker="s", color="#e34948", label="sequential (full prompts)")
        ax.set_xlabel(label)
        ax.set_ylabel("latency (ms, median of 7)")
        ax.set_ylim(bottom=0)
        if key == "questions":
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in x])
        ax.set_facecolor("white")
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/q-rlcd")
    ap.add_argument("--out", default="runs/decide-bench")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decider = Decider.load(args.model, args.device)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device
    results = bench(decider, decider.tok)
    results["device"] = device
    results["model"] = args.model
    results["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    (out / "latency.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "latency.md").write_text(table(results, device))
    plot(results, out / "latency.png")
    print(f"\npeak memory {results['max_memory_gb']:.2f} GB; wrote {out / 'latency.md'} and {out / 'latency.png'}")


if __name__ == "__main__":
    main()
