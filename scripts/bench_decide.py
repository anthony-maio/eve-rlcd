"""Latency of ask() against ask_sequential() on one GPU, with the body in fp32 (the default) and
under bf16 autocast (fast=True), medians of 7 runs after 2 warmups: (a) 1..64 questions against one
800-token state, (b) state length 200, 800, 1500 tokens at 8 questions, (c) option count 2, 5, 10, 26
at 8 questions.

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
COLUMNS = ("ask_fp32_ms", "ask_fast_ms", "sequential_fp32_ms", "sequential_fast_ms")
HEADERS = ("ask fp32", "ask fast", "sequential fp32", "sequential fast")


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


def measure(decider: Decider, state: str, qs: list[ChoiceQ]) -> dict:
    return {"ask_fp32_ms": timed(lambda: decider.ask(state, qs, fast=False)),
            "ask_fast_ms": timed(lambda: decider.ask(state, qs, fast=True)),
            "sequential_fp32_ms": timed(lambda: decider.ask_sequential(state, qs, fast=False)),
            "sequential_fast_ms": timed(lambda: decider.ask_sequential(state, qs, fast=True))}


def _line(label: str, r: dict) -> str:
    cells = "   ".join(f"{h} {r[c]:7.1f}" for h, c in zip(HEADERS, COLUMNS))
    return f"{label:<24} {cells} ms   speedup fp32 {r['sequential_fp32_ms'] / r['ask_fp32_ms']:.2f}x"


def bench(decider: Decider, tok) -> dict:
    results = {"questions": [], "state_tokens": [], "options": []}
    state = make_state(tok, 800)
    for m in QUESTION_COUNTS:
        r = {"m": m, **measure(decider, state, make_questions(m, 4))}
        results["questions"].append(r)
        print(_line(f"M={m} k=4 state=800", r))
    qs = make_questions(8, 4)
    for n in STATE_TOKENS:
        r = {"state_tokens": n, **measure(decider, make_state(tok, n), qs)}
        results["state_tokens"].append(r)
        print(_line(f"M=8 k=4 state={n}", r))
    for k in OPTION_COUNTS:
        r = {"k": k, **measure(decider, state, make_questions(8, k))}
        results["options"].append(r)
        print(_line(f"M=8 k={k} state=800", r))
    return results


def _table(rows: list[dict], xkey: str, xlabel: str) -> list[str]:
    lines = [f"| {xlabel} | " + " | ".join(f"{h} (ms)" for h in HEADERS) + " | speedup fp32 | speedup fast |",
             "|---" * 7 + "|"]
    for r in rows:
        lines.append(f"| {r[xkey]} | " + " | ".join(f"{r[c]:.1f}" for c in COLUMNS)
                     + f" | {r['sequential_fp32_ms'] / r['ask_fp32_ms']:.2f}x"
                     + f" | {r['sequential_fast_ms'] / r['ask_fast_ms']:.2f}x |")
    return lines


def table(results: dict, device: str) -> str:
    lines = [f"Latency of the decision API on {device}. Medians of 7 runs after 2 warmups, in milliseconds. "
             "ask: one prefill of the state, one batched forward over the question suffixes. sequential: every "
             "full prompt through the training-time path, right-padded in batches of 32. fp32: the body in fp32 "
             "(the default, exact against the training-time path). fast: the body under bf16 autocast "
             "(fast=True). The decision head is fp32 in every column. speedup fp32 is sequential fp32 over ask "
             "fp32; speedup fast is sequential fast over ask fast.", ""]
    lines += ["**(a) question count, one 800-token state, 4 options each**", ""]
    lines += _table(results["questions"], "m", "questions")
    lines += ["", "**(b) state length, 8 questions, 4 options each**", ""]
    lines += _table(results["state_tokens"], "state_tokens", "state tokens")
    lines += ["", "**(c) option count, 8 questions, one 800-token state**", ""]
    lines += _table(results["options"], "k", "options")
    return "\n".join(lines) + "\n"


def plot(results: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("questions", "m", "questions (one 800-token state, 4 options)"),
              ("state_tokens", "state_tokens", "state tokens (8 questions, 4 options)"),
              ("options", "k", "options per question (8 questions, 800-token state)")]
    styles = [("ask_fp32_ms", "ask, fp32 body", "#2a78d6", "o", "-"),
              ("ask_fast_ms", "ask, fast (bf16)", "#2a78d6", "o", "--"),
              ("sequential_fp32_ms", "sequential, fp32 body", "#e34948", "s", "-"),
              ("sequential_fast_ms", "sequential, fast (bf16)", "#e34948", "s", "--")]
    for ax, (key, xkey, label) in zip(axes, panels):
        rows = results[key]
        x = [r[xkey] for r in rows]
        for col, name, color, marker, ls in styles:
            ax.plot(x, [r[col] for r in rows], marker=marker, color=color, linestyle=ls, label=name)
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
