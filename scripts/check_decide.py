"""Proofs for the decision API on a trained checkpoint (GPU).

1. Equivalence: over test rows grouped by context, ask() (one prefill, parallel suffixes) against
   ask_sequential() (every full prompt through the training-time path), in bf16 autocast and with
   the body in fp32, with 1, 5 and 20 unrelated questions from other rows mixed in.
2. Independence: a question's probabilities when other questions are added, removed or reordered.
3. Accuracy sanity: ask() over the whole test split, grouped by context, scored like rlcd.eval
   (body in fp32 by default; --accuracy-fast for bf16 autocast, which is what the eval used).

The pass rule: the cached path must agree with the training-time path to within FLOOR_FACTOR
times the amount the training-time path disagrees with itself across batch sizes on the same rows
(max |sequential at batch 1 - sequential at batch 32|, its own rounding floor), and never worse than
TOL absolute (1e-5 in fp32, 1e-4 in bf16). Floor, threshold and observed maximum are recorded.

Writes <out>/equivalence.json and <out>/preds_ask.jsonl. Exit status 1 when a check fails; every
measured maximum is printed either way.

    uv run python scripts/check_decide.py --model runs/q-rlcd --out runs/decide-bench
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import time
from pathlib import Path

import torch

from rlcd.decide import ChoiceQ, Decider, NoulQ, ScoreQ
from rlcd.eval import summarize, to_json, write_rows
from rlcd.quick_eval import stride_sample
from rlcd.schema import NOTA, Question, read_jsonl

TOL = {"bf16": 1e-4, "fp32": 1e-5}
FLOOR_FACTOR = 3
CHECKS = ("ask_vs_sequential", "with_extras_vs_sequential", "add", "remove", "reorder")


def threshold(mode: str, floor: float) -> float:
    """The pass threshold: the cached path must agree with the training-time path to within
    FLOOR_FACTOR times the amount the training-time path disagrees with itself across batch
    sizes on the same rows (its own rounding floor), and never worse than TOL[mode] absolute."""
    return max(TOL[mode], FLOOR_FACTOR * floor)


def to_primitive(q: Question):
    """The primitive a dataset row stands for; each renders the row's prompt verbatim."""
    if q.primitive == "score":
        return ScoreQ(q.question, q.choices)
    if q.primitive == "noul":
        return NoulQ(q.question)
    return ChoiceQ(q.question, q.choices)


def group_rows(rows: list[Question]) -> list[list[Question]]:
    groups: dict[str, list[Question]] = collections.defaultdict(list)
    for q in rows:
        groups[q.context].append(q)
    return list(groups.values())


def pick_groups(groups: list[list[Question]], n_rows: int, seed: int) -> list[list[Question]]:
    """Groups covering about n_rows rows: every multi-question group is a candidate, singles
    fill the rest, spread over the split (which is sorted by source) with a stride."""
    multi = [g for g in groups if len(g) > 1]
    single = [g for g in groups if len(g) == 1]
    rng = random.Random(seed)
    rng.shuffle(multi)
    chosen: list[list[Question]] = []
    total = 0
    for g in multi:
        if total >= n_rows // 2:
            break
        chosen.append(g)
        total += len(g)
    chosen += stride_sample(single, n_rows - total)
    return chosen


def first_k(probs: torch.Tensor, qs: list[Question]) -> list[torch.Tensor]:
    return [probs[i, : q.k] for i, q in enumerate(qs)]


class Diffs:
    """The maximum and the mean of |a - b| over every probability compared, per check."""

    def __init__(self, keys):
        self.max = {k: 0.0 for k in keys}
        self.sum = {k: 0.0 for k in keys}
        self.count = {k: 0 for k in keys}

    def add(self, key: str, a: list[torch.Tensor], b: list[torch.Tensor]) -> None:
        for x, y in zip(a, b):
            d = (x - y).abs()
            self.max[key] = max(self.max[key], d.max().item())
            self.sum[key] += d.sum().item()
            self.count[key] += d.numel()

    def report(self) -> dict:
        return {k: {"max": self.max[k], "mean": self.sum[k] / self.count[k] if self.count[k] else None,
                    "n": self.count[k]} for k in self.max}


def check_equivalence(decider: Decider, chosen: list[list[Question]], pool: list[Question],
                      extras: tuple[int, ...], seed: int) -> dict:
    rng = random.Random(seed)
    diffs = Diffs(["ask_vs_sequential", "with_extras_vs_sequential", "add", "remove", "reorder",
                   "sequential_bs1_vs_bs32"])
    for g in chosen:
        state = g[0].context
        prims = [to_primitive(q) for q in g]
        base = first_k(decider.probs(state, prims), g)
        ref = first_k(torch.softmax(decider.logits_sequential(state, prims), -1), g)
        diffs.add("ask_vs_sequential", base, ref)
        ref1 = first_k(torch.softmax(decider.logits_sequential(state, prims, batch_size=1), -1), g)
        diffs.add("sequential_bs1_vs_bs32", ref1, ref)
        for n_extra in extras:
            others = [to_primitive(q) for q in rng.sample(pool, n_extra) if q.context != state][:n_extra]
            mixed = first_k(decider.probs(state, prims + others), g)
            diffs.add("add", mixed, base)
            ref_mixed = first_k(torch.softmax(decider.logits_sequential(state, prims + others), -1), g)
            diffs.add("with_extras_vs_sequential", mixed, ref_mixed)
        if len(g) > 1:
            alone = first_k(decider.probs(state, prims[:1]), g[:1])
            diffs.add("remove", alone, base[:1])
            rev = first_k(decider.probs(state, prims[::-1]), g[::-1])[::-1]
            diffs.add("reorder", rev, base)
    return diffs.report()


def check_against_fp32(decider: Decider, chosen: list[list[Question]]) -> dict:
    """How far each bf16 path sits from the fp32 single-pass result: the bf16 error budget of
    the training-time path itself, next to that of the cached path."""
    diffs = Diffs(["ask_bf16_vs_sequential_fp32", "sequential_bf16_vs_sequential_fp32"])
    for g in chosen:
        state = g[0].context
        prims = [to_primitive(q) for q in g]
        decider.fast = False
        truth = first_k(torch.softmax(decider.logits_sequential(state, prims), -1), g)
        decider.fast = True
        ask = first_k(decider.probs(state, prims), g)
        seq = first_k(torch.softmax(decider.logits_sequential(state, prims), -1), g)
        diffs.add("ask_bf16_vs_sequential_fp32", ask, truth)
        diffs.add("sequential_bf16_vs_sequential_fp32", seq, truth)
    return diffs.report()


def show(worst: dict) -> None:
    for key, d in worst.items():
        if isinstance(d, dict) and "max" in d:
            print(f"  |diff| {key:<40} max {d['max']:.3e}   mean {d['mean']:.3e}   over {d['n']} probabilities")
        elif key == "seconds":
            print(f"  {d:.1f} s")


def check_split_on_rows(tok, rows: list[Question]) -> dict:
    """Rows where tokenizing the prefix and the suffix separately differs from tokenizing the
    whole prompt, for the split used (after the blank line) and for the fallback split the task
    names (before the last newline of the prefix)."""
    from rlcd.schema import render_prefix, render_prompt, render_suffix
    failed, failed_fallback = [], []
    for q in rows:
        prefix, suffix = render_prefix(q.context), render_suffix(q)
        whole = tok.encode(render_prompt(q), add_special_tokens=False)
        if tok.encode(prefix, add_special_tokens=False) + tok.encode(suffix, add_special_tokens=False) != whole:
            failed.append(q.id)
        if (tok.encode(prefix[:-1], add_special_tokens=False)
                + tok.encode("\n" + suffix, add_special_tokens=False)) != whole:
            failed_fallback.append(q.id)
    return {"rows": len(rows), "failed_after_blank_line": failed, "failed_before_last_newline": len(failed_fallback)}


def check_accuracy(decider: Decider, groups: list[list[Question]], out: Path) -> dict:
    preds = []
    t0 = time.perf_counter()
    for g in groups:
        probs = decider.logits(g[0].context, [to_primitive(q) for q in g])
        for q, row in zip(g, probs.cpu()):
            preds.append({"id": q.id, "source": q.source, "primitive": q.primitive, "k": q.k, "answer": q.answer,
                          "logits": row[: q.k].tolist(), "nota_index": q.choices.index(NOTA) if NOTA in q.choices else -1})
    seconds = time.perf_counter() - t0
    write_rows(out / "preds_ask.jsonl", preds)
    summary = summarize(preds)["overall"]
    return {"n": len(preds), "acc": summary["acc"], "ece": summary["ece"], "brier": summary["brier"],
            "seconds": seconds, "groups": len(groups)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/q-rlcd")
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--metrics", default=None, help="metrics.json to reproduce (default <model>/eval/metrics.json)")
    ap.add_argument("--out", default="runs/decide-bench")
    ap.add_argument("--rows", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--accuracy-fast", action="store_true",
                    help="run the accuracy sanity under bf16 autocast (the eval's setting) instead of fp32")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    decider = Decider.load(args.model, args.device)
    rows = read_jsonl(args.split)
    groups = group_rows(rows)
    chosen = pick_groups(groups, args.rows, args.seed)
    n_chosen = sum(len(g) for g in chosen)
    print(f"{len(rows)} rows in {len(groups)} context groups; checking {n_chosen} rows in {len(chosen)} groups "
          f"({sum(1 for g in chosen if len(g) > 1)} multi-question)")

    report = {"model": args.model, "rows_checked": n_chosen, "groups_checked": len(chosen), "tolerances": TOL,
              "equivalence": {}, "passed": {}}
    split = check_split_on_rows(decider.tok, rows)
    report["split"] = split
    failed = bool(split["failed_after_blank_line"])
    report["passed"]["split"] = not failed
    print(f"tokenization split over {split['rows']} rows: {len(split['failed_after_blank_line'])} rows differ with "
          f"the split after the blank line{' ' + str(split['failed_after_blank_line'][:5]) if failed else ''}; "
          f"{split['failed_before_last_newline']} rows differ with the fallback split before the last newline")
    for mode in ("bf16", "fp32"):
        decider.fast = mode == "bf16"
        t0 = time.perf_counter()
        worst = check_equivalence(decider, chosen, rows, (1, 5, 20), args.seed)
        worst["seconds"] = time.perf_counter() - t0
        floor = worst["sequential_bs1_vs_bs32"]["max"]
        observed = max(worst[k]["max"] for k in CHECKS)
        limit = threshold(mode, floor)
        worst["rule"] = {"floor": floor, "floor_factor": FLOOR_FACTOR, "absolute": TOL[mode], "threshold": limit,
                         "observed_max": observed}
        report["equivalence"][mode] = worst
        ok = observed <= limit
        report["passed"][mode] = ok
        failed |= not ok
        print(f"\n[{mode}] {'PASS' if ok else 'FAIL'}: observed max {observed:.3e} against threshold {limit:.3e} "
              f"= max({TOL[mode]:g}, {FLOOR_FACTOR} x floor {floor:.3e}), the floor being the training-time path "
              f"against itself across batch sizes on the same rows")
        show(worst)
    report["bf16_error_budget"] = check_against_fp32(decider, chosen)
    print("\n[bf16 against the fp32 single pass]")
    show(report["bf16_error_budget"])
    decider.fast = args.accuracy_fast

    if not args.skip_accuracy:
        acc = check_accuracy(decider, groups, out)
        acc["fast"] = args.accuracy_fast
        metrics_path = Path(args.metrics) if args.metrics else Path(args.model) / "eval" / "metrics.json"
        want = json.loads(metrics_path.read_text())["overall"]
        acc["reference"] = {"acc": want["acc"], "ece": want["ece"], "brier": want["brier"], "path": str(metrics_path)}
        acc["acc_diff"] = acc["acc"] - want["acc"]
        acc["ece_diff"] = acc["ece"] - want["ece"]
        ok = abs(acc["acc_diff"]) <= 0.002 and abs(acc["ece_diff"]) <= 0.002
        report["accuracy"] = acc
        report["passed"]["accuracy"] = ok
        failed |= not ok
        print(f"\n[accuracy] ask ({'bf16 autocast' if args.accuracy_fast else 'fp32 body'}) over {acc['n']} rows "
              f"in {acc['groups']} groups, {acc['seconds']:.0f} s: {'PASS' if ok else 'FAIL'}")
        print(f"  acc {acc['acc']:.4f} (eval {want['acc']:.4f}, diff {acc['acc_diff']:+.4f})")
        print(f"  ece {acc['ece']:.4f} (eval {want['ece']:.4f}, diff {acc['ece_diff']:+.4f})")
        print(f"  brier {acc['brier']:.4f} (eval {want['brier']:.4f})")

    report["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    (out / "equivalence.json").write_text(to_json(report) + "\n")
    print(f"\nwrote {out / 'equivalence.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
