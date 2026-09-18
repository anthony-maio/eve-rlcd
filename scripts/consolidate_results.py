"""Consolidate the final Qwen3-0.6B-Base results into docs/results.md and docs/img/*.png.

Every number in the document is read from a file under runs/ (preds.jsonl, probe.json, meta.json,
train_log.jsonl) or under data/ (row counts) and computed here with rlcd.eval and rlcd.metrics; the
prose is a template filled from the same numbers. Run it from the repository root:

    uv run --no-sync python scripts/consolidate_results.py

The output is deterministic (bootstrap seed 0, Agg backend): running it twice gives byte-identical
files. It refuses to pair runs whose preds.jsonl rows are not the same ids in the same order.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rlcd.eval import (N_BINS, MIN_BIN_ROWS, _arrays, _finish, _pyplot, curve_points, load_preds,  # noqa: E402
                       paired_differences, plot_coverage, plot_reliability, run_style, stopped_step,
                       summarize)

DOC = ROOT / "docs" / "results.md"
IMG = ROOT / "docs" / "img"
LATENCY_SRC = ROOT / "runs" / "decide-bench" / "latency.png"
FP32_PREDS = ROOT / "runs" / "decide-bench" / "preds_ask.jsonl"  # runs/q-rlcd scored in fp32 by the decision API
RLVR_FIRST_ATTEMPT = ROOT / "runs" / "q-rlvr" / "train_log.jsonl"  # same seed and command as runs/q-rlvr-b

# ---------- the runs ----------

ARMS = {  # arm key -> label in the tables, and the RUN_STYLE name that gives its color
    "warmup": ("warmup (100-step SFT on 6,400 rows)", "q-warmup"),
    "warmup-cont": ("supervised continuation (500 more SFT steps, same 6,400 rows)", "q-warmup-cont"),
    "rlcd": ("RLCD", "q-rlcd"),
    "rlvr": ("RLVR, shared settings (lr 2e-5, stopped)", "q-rlvr"),
    "rlvr-lowlr": ("RLVR, low lr (4e-6)", "q-rlvr-lowlr"),
    "oracle": ("oracle (labels revealed, same RL rows)", "q-oracle"),
    "sft32k": ("32k-label SFT reference (500 steps on rows 0..31999)", "bake-qwen06"),
}

# key, arm, hardware, seed label, run directory (relative to the repo root)
RUNS = [
    ("local/warmup", "warmup", "local", "seed 0", "runs/q-warmup"),
    ("local/rlcd", "rlcd", "local", "seed 0", "runs/q-rlcd"),
    ("local/rlvr", "rlvr", "local", "seed 0", "runs/q-rlvr-b"),
    ("local/rlvr-lowlr", "rlvr-lowlr", "local", "seed 0", "runs/q-rlvr-lowlr"),
    ("local/oracle", "oracle", "local", "seed 0", "runs/q-oracle"),
    ("local/sft32k", "sft32k", "local", "seed 0", "runs/bake-qwen06"),
    ("colab/warmup", "warmup", "colab", "seed 0 (Colab)", "runs/colab/runs/q-warmup"),
    ("colab/warmup-cont", "warmup-cont", "colab", "seed 0 (Colab)", "runs/colab/runs/q-warmup-cont"),
    ("colab/rlcd-s1", "rlcd", "colab", "seed 1", "runs/colab/runs/q-rlcd-s1"),
    ("colab/rlcd-s2", "rlcd", "colab", "seed 2", "runs/colab/runs/q-rlcd-s2"),
    ("colab/rlvr-lowlr-s1", "rlvr-lowlr", "colab", "seed 1", "runs/colab/runs/q-rlvr-lowlr-s1"),
    ("colab/rlvr-lowlr-s2", "rlvr-lowlr", "colab", "seed 2", "runs/colab/runs/q-rlvr-lowlr-s2"),
    ("colab/oracle-s1", "oracle", "colab", "seed 1", "runs/colab/runs/q-oracle-s1"),
    ("colab/oracle-s2", "oracle", "colab", "seed 2", "runs/colab/runs/q-oracle-s2"),
]

# Paired comparisons: (block title, [(label, a, b), ...]). Each block pairs runs that started from
# the same warmup checkpoint, so the difference is the effect of what came after the warmup.
PAIRS = [
    ("Local, seed 0 (RTX 4080), all RL arms from runs/q-warmup", [
        ("RLCD minus warmup", "local/rlcd", "local/warmup"),
        ("RLCD minus RLVR low lr", "local/rlcd", "local/rlvr-lowlr"),
        ("RLCD minus RLVR shared settings (stopped at step 150)", "local/rlcd", "local/rlvr"),
        ("RLCD minus oracle", "local/rlcd", "local/oracle"),
        ("RLCD minus 32k-label SFT", "local/rlcd", "local/sft32k"),
        ("oracle minus 32k-label SFT", "local/oracle", "local/sft32k"),
    ]),
    ("Colab, seed 1 (A100), all RL arms from the Colab runs/q-warmup", [
        ("RLCD minus warmup", "colab/rlcd-s1", "colab/warmup"),
        ("RLCD minus supervised continuation", "colab/rlcd-s1", "colab/warmup-cont"),
        ("RLCD minus RLVR low lr", "colab/rlcd-s1", "colab/rlvr-lowlr-s1"),
        ("RLCD minus oracle", "colab/rlcd-s1", "colab/oracle-s1"),
    ]),
    ("Colab, seed 2 (A100), all RL arms from the Colab runs/q-warmup", [
        ("RLCD minus warmup", "colab/rlcd-s2", "colab/warmup"),
        ("RLCD minus supervised continuation", "colab/rlcd-s2", "colab/warmup-cont"),
        ("RLCD minus RLVR low lr", "colab/rlcd-s2", "colab/rlvr-lowlr-s2"),
        ("RLCD minus oracle", "colab/rlcd-s2", "colab/oracle-s2"),
    ]),
]

METRICS = [("acc", "accuracy"), ("ece", f"ECE ({N_BINS} bins)"), ("brier", "Brier loss"),
           ("mean_conf", "mean confidence"), ("conf_minus_acc", "confidence minus accuracy"),
           ("nota_rate", "NOTA recall"), ("nota_false_alarm", "NOTA false alarm")]

PROBE_COLS = [  # (group, key, header with the ideal value)
    ("dept_single", "mean_max_p", "one cue: mean max p (ideal 1.0)"),
    ("dept_double", "mean_max_p", "two cues: mean max p (ideal 0.5)"),
    ("dept_double", "mean_mass_on_cued", "two cues: mass on the cued pair (ideal 1.0)"),
    ("escalate", "acc", "escalate: accuracy (ideal 0.90)"),
    ("escalate", "mean_conf", "escalate: mean confidence (ideal 0.90)"),
]


# ---------- loading ----------

def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_log(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_rows(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_runs() -> dict[str, dict]:
    runs = {}
    for key, arm, hardware, seed, rel in RUNS:
        d = ROOT / rel
        preds = load_preds(d / "eval" / "preds.jsonl")
        runs[key] = {"key": key, "arm": arm, "hardware": hardware, "seed": seed, "dir": rel,
                     "meta": read_json(d / "meta.json"), "preds": preds,
                     "probe": read_json(d / "eval" / "probe.json"), "log": read_log(d / "train_log.jsonl"),
                     "eval_meta": read_json(d / "eval" / "metrics.json")["meta"]}
    ids = {key: [p["id"] for p in r["preds"]] for key, r in runs.items()}
    answers = {key: [p["answer"] for p in r["preds"]] for key, r in runs.items()}
    first = RUNS[0][0]
    for key in runs:
        if ids[key] != ids[first] or answers[key] != answers[first]:
            raise SystemExit(f"{key} was not evaluated on the same rows as {first}; refusing to consolidate")
        if runs[key]["eval_meta"]["split"] != runs[first]["eval_meta"]["split"]:
            raise SystemExit(f"{key} was evaluated on {runs[key]['eval_meta']['split']}, not the shared split")
    for key, r in runs.items():
        r["summary"] = summarize(r["preds"])
    return runs


# ---------- formatting ----------

def fmt(x, digits=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{digits}f}"


def fmt_signed(x, digits=3) -> str:
    return f"{x:+.{digits}f}"


def small(x: float) -> str:
    """An absolute difference to four decimals, or 'under 0.0001'."""
    return "under 0.0001" if abs(x) < 0.0001 else f"{abs(x):.4f}"


def agg(values: list[float], digits=3) -> str:
    """mean [min, max] over the seeds, or the single value."""
    if len(values) == 1:
        return fmt(values[0], digits)
    return f"{np.mean(values):.{digits}f} [{min(values):.{digits}f}, {max(values):.{digits}f}]"


def by_arm(runs: dict[str, dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for r in runs.values():
        out[r["arm"]].append(r)
    return out


def n_label(members: list[dict]) -> str:
    if len(members) == 1:
        return f"1 ({members[0]['hardware']})"
    return f"{len(members)} ({', '.join(m['hardware'] for m in members)})"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


# ---------- sections ----------

def run_table(runs: dict[str, dict]) -> str:
    header = ["key", "arm", "hardware", "seed", "run directory", "init", "steps", "rows x epochs",
              "batch (prompts/step)", "peak lr", "test accuracy"]
    rows = []
    for r in runs.values():
        m = r["meta"]
        steps = f"{m['steps']}" + (" (stopped)" if m.get("stopped") else "")
        rows.append([r["key"], r["arm"], r["hardware"], r["seed"], r["dir"], m["init"], steps,
                     f"{m['slice_rows']} x {m['epochs']}", str(m["micro"] * m["accum"]), f"{m['lr']:g}",
                     fmt(r["summary"]["overall"]["acc"])])
    return md_table(header, rows)


def main_table(arms: dict[str, list[dict]]) -> str:
    header = ["arm", "n runs (hardware)"] + [label for _, label in METRICS]
    rows = []
    for arm, (label, _) in ARMS.items():
        members = arms[arm]
        cells = [agg([m["summary"]["overall"][key] for m in members]) for key, _ in METRICS]
        rows.append([label, n_label(members)] + cells)
    return md_table(header, rows)


def per_run_table(runs: dict[str, dict]) -> str:
    header = ["run", "accuracy [95% CI]", "Brier loss [95% CI]", f"ECE ({N_BINS} bins) [95% CI]", "ECE bias",
              "mean confidence", "conf minus acc", "NOTA recall", "NOTA false alarm"]
    rows = []
    for r in runs.values():
        m = r["summary"]["overall"]

        def ci(key):
            lo, hi = m["ci"][key]
            return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append([r["key"], ci("acc"), ci("brier"), ci("ece"), fmt_signed(m["ece_bias"]), fmt(m["mean_conf"]),
                     fmt_signed(m["conf_minus_acc"]), fmt(m["nota_rate"]), fmt(m["nota_false_alarm"])])
    return md_table(header, rows)


def paired_block(runs: dict[str, dict]) -> tuple[str, dict]:
    parts = []
    result = {}
    for title, pairs in PAIRS:
        header = ["pair", "accuracy difference [95% CI]", f"ECE difference [95% CI]",
                  "Brier loss difference [95% CI]"]
        rows = []
        for label, a, b in pairs:
            d = paired_differences(runs[a]["preds"], runs[b]["preds"])  # refuses mismatched ids
            result[(title, label)] = d

            def cell(k):
                return f"{d[k]['diff']:+.3f} [{d[k]['lo']:+.3f}, {d[k]['hi']:+.3f}]"
            rows.append([f"{label} ({a} minus {b})", cell("acc"), cell("ece"), cell("brier")])
        parts += [f"**{title}**", "", md_table(header, rows), ""]
    return "\n".join(parts), result


def probe_table(arms: dict[str, list[dict]]) -> str:
    header = ["arm", "n runs"] + [h for _, _, h in PROBE_COLS]
    rows = []
    for arm, (label, _) in ARMS.items():
        members = arms[arm]
        cells = [agg([m["probe"]["report"][g][k] for m in members]) for g, k, _ in PROBE_COLS]
        rows.append([label, str(len(members))] + cells)
    return md_table(header, rows)


def source_tables(arms: dict[str, list[dict]], which: list[str]) -> str:
    sources = sorted({s for arm in which for m in arms[arm] for s in m["summary"]["by_source"]})
    parts = []
    for title, key in (("Accuracy by source", "acc"), (f"ECE by source ({N_BINS} bins)", "ece")):
        rows = []
        for arm in which:
            members = arms[arm]
            label = f"{ARMS[arm][0]} (n={len(members)})"
            rows.append([label] + [fmt(np.mean([m["summary"]["by_source"][s][key] for m in members]))
                                   for s in sources])
        parts += [f"**{title}, mean across runs**", "", md_table(["arm"] + sources, rows), ""]
    return "\n".join(parts)


# ---------- figures ----------

def plot_curves_with_seeds(local: dict[str, list[dict]], seeds: dict[str, list[list[dict]]], offsets: dict[str, int],
                           out_png: Path) -> None:
    """Local arms as in rlcd.eval.plot_curves, plus each Colab seed curve as a thin unmarked line in
    the color of its arm, so the seed spread sits behind the seed-0 curve."""
    plt = _pyplot()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for name, logs in seeds.items():
        style = run_style(name)
        for i, rows in enumerate(logs):
            for ax, key in ((top, "eval_acc"), (bottom, "eval_ece")):
                x, y = curve_points(rows, key, offsets.get(name, 0))
                ax.plot(x, y, lw=0.9, alpha=0.55, color=style["color"], ls="-",
                        label=f"{name}, Colab seeds 1 and 2" if (i == 0 and ax is top) else None)
    for name, rows in local.items():
        style = run_style(name)
        for ax, key in ((top, "eval_acc"), (bottom, "eval_ece")):
            x, y = curve_points(rows, key, offsets.get(name, 0))
            ax.plot(x, y, lw=1.6, ms=5, label=name if ax is top else None, zorder=3, **style)
            if x and stopped_step(rows) is not None:
                ax.plot(x[-1:], y[-1:], ls="none", marker="x", ms=11, mew=2.2, color="black", zorder=5)
                ax.annotate("stopped", (x[-1], y[-1]), xytext=(6, 6), textcoords="offset points", fontsize=8)
    top.set_ylabel("held-out accuracy (2000-row val sample)")
    bottom.set_ylabel(f"held-out ECE ({N_BINS} bins)")
    bottom.set_xlabel("optimizer step, counted from the start of the warmup")
    bottom.set_ylim(bottom=0)
    _finish(fig, [top, bottom], out_png, legend_ax=top)
    plt.close(fig)


def plot_probe(arms: dict[str, list[dict]], out_png: Path) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    names = list(ARMS)
    width = 0.38
    for i, arm in enumerate(names):
        members = arms[arm]
        color = run_style(ARMS[arm][1])["color"]
        for j, (group, hatch, alpha) in enumerate((("dept_single", None, 1.0), ("dept_double", "//", 0.45))):
            vals = [m["probe"]["report"][group]["mean_max_p"] for m in members]
            mean = float(np.mean(vals))
            err = np.array([[mean - min(vals)], [max(vals) - mean]])
            ax.bar(i + (j - 0.5) * width, mean, width, color=color, alpha=alpha, hatch=hatch,
                   edgecolor=color, linewidth=0.8, yerr=err if len(vals) > 1 else None,
                   error_kw={"ecolor": "black", "capsize": 2, "lw": 0.8})
    ax.axhline(1.0, color="black", ls="--", lw=1, label="ideal with one cue (1.0)")
    ax.axhline(0.5, color="black", ls=":", lw=1, label="ideal with two cues (0.5)")
    ax.bar([np.nan], [np.nan], color="#888888", label="one cued department")
    ax.bar([np.nan], [np.nan], color="#888888", alpha=0.45, hatch="//", edgecolor="#888888", label="two cued departments")
    short = {"warmup": "warmup", "warmup-cont": "sup. cont.", "rlcd": "RLCD", "rlvr": "RLVR\n(stopped)",
             "rlvr-lowlr": "RLVR\nlow lr", "oracle": "oracle", "sft32k": "32k SFT"}
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short[a] for a in names], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean max probability on the department question")
    ax.set_xlabel("arm (bar: mean across runs; whisker: min to max across seeds)")
    _finish(fig, [ax], out_png)
    plt.close(fig)


def make_figures(runs: dict[str, dict], arms: dict[str, list[dict]]) -> dict[str, str]:
    IMG.mkdir(parents=True, exist_ok=True)
    local = {"q-warmup": "local/warmup", "q-rlcd": "local/rlcd", "q-rlvr": "local/rlvr",
             "q-rlvr-lowlr": "local/rlvr-lowlr", "q-oracle": "local/oracle", "bake-qwen06": "local/sft32k"}
    preds = {name: runs[key]["preds"] for name, key in local.items()}
    plot_reliability(preds, IMG / "reliability.png")
    plot_coverage(preds, IMG / "coverage.png")

    warmup_steps = runs["local/warmup"]["meta"]["steps"]
    curves_local = {name: runs[key]["log"] for name, key in local.items() if name != "bake-qwen06"}
    seeds = {"q-rlcd": [runs["colab/rlcd-s1"]["log"], runs["colab/rlcd-s2"]["log"]],
             "q-rlvr-lowlr": [runs["colab/rlvr-lowlr-s1"]["log"], runs["colab/rlvr-lowlr-s2"]["log"]]}
    offsets = {name: warmup_steps for name in list(curves_local) + list(seeds) if name != "q-warmup"}
    plot_curves_with_seeds(curves_local, seeds, offsets, IMG / "curves.png")

    plot_probe(arms, IMG / "probe.png")
    shutil.copyfile(LATENCY_SRC, IMG / "latency.png")
    return {
        "reliability.png": "Reliability diagram of the five local seed-0 arms and the 32k-label SFT reference "
                           f"on the 8000-row test split ({N_BINS} equal-width bins; bins with fewer than "
                           f"{MIN_BIN_ROWS} rows are not drawn). ECE in the legend.",
        "curves.png": "Held-out accuracy and ECE on the 2000-row validation stride sample against optimizer "
                      f"step. The warmup is drawn from step 0 and every RL arm continues from step {warmup_steps}. "
                      "Thick lines with markers: the local seed-0 runs. Thin lines: the Colab seed 1 and seed 2 "
                      "runs of RLCD and RLVR low lr, in the same colors. The black x marks where the stop rule "
                      "ended the shared-settings RLVR run.",
        "coverage.png": "Selective prediction on the test split: error rate among the answered rows when the "
                        "most confident fraction is answered, for the local seed-0 arms and the SFT reference.",
        "probe.png": "Known-posterior probe: mean max probability on the department question of fresh synthetic "
                     "tickets with one cued department (solid, ideal 1.0) and two cued departments (hatched, "
                     "ideal 0.5), per arm; whiskers span the seeds.",
        "latency.png": "Latency of the decision API on an RTX 4080 (copied from runs/decide-bench/latency.png): "
                       "the cached-prefix ask path against the sequential training-time path, in fp32 and under "
                       "bf16 autocast, against question count, state length and option count.",
    }


# ---------- prose ----------

def setup_paragraph(runs: dict[str, dict]) -> str:
    w, r, lo, sc, sft = (runs[k]["meta"] for k in ("local/warmup", "local/rlcd", "local/rlvr-lowlr",
                                                   "colab/warmup-cont", "local/sft32k"))
    cw, cr = runs["colab/warmup"]["meta"], runs["colab/rlcd-s1"]["meta"]
    n_train, n_val, n_test = (count_rows(ROOT / "data" / f"{s}.jsonl") for s in ("train", "val", "test"))
    n_test_eval = runs["local/rlcd"]["eval_meta"]["n"]
    eval_n = r["eval_n"]
    stop = runs["local/rlvr"]["meta"]["stopped"]
    return (
        f"Base model `{w['init']}` (full fine-tune, fp32 master weights, bf16 autocast, AdamW with betas 0.9 "
        f"and 0.95 and weight decay 0.1, cosine schedule to a floor of 0.1 x peak after a linear warmup of 5 "
        f"percent of the steps). Data: `{w['data']}` has {n_train:,} rows, `data/val.jsonl` {n_val:,}, "
        f"`data/test.jsonl` {n_test:,}; every number below is on the full {n_test_eval:,}-row test split at "
        f"temperature 1 with prompts truncated to {r['max_len']} tokens. Warmup: supervised, rows "
        f"{w['slice_start']}..{w['slice_start'] + w['slice_rows'] - 1} ({w['slice_rows']:,} rows), "
        f"{w['epochs']} epoch, {w['micro'] * w['accum']} prompts per step, {w['steps']} steps, peak lr "
        f"{w['lr']:g}. RL arms: rows {r['slice_start']}..{r['slice_start'] + r['slice_rows'] - 1} "
        f"({r['slice_rows']:,} rows, disjoint from the warmup rows), {r['epochs']} epochs, "
        f"{r['micro'] * r['accum']} prompts per step with {r['group']} sampled actions per prompt (group "
        f"size {r['group']}, leave-one-out baseline), {r['steps']} steps, no KL term (kl {r['kl']}), peak lr "
        f"{r['lr']:g} for RLCD, the oracle and the shared-settings RLVR arm, and {lo['lr']:g} for the RLVR "
        f"low-lr arm. RLCD's reward is the outcome minus the stated probability of the taken action, RLVR's "
        f"the outcome alone; the oracle sees the label and minimizes its NLL on the same rows. Stop rule, "
        f"checked at every evaluation (every {r['eval_every']} steps on a {eval_n:,}-row stride sample of the "
        f"validation split): \"{stop}\"; it fired only for the shared-settings RLVR run. Supervised "
        f"continuation control: the Colab warmup trained for {sc['epochs']} more epochs on the same "
        f"{sc['slice_rows']:,} rows, {sc['micro'] * sc['accum']} prompts per step, {sc['steps']} steps, peak "
        f"lr {sc['lr']:g}. 32k-label SFT reference: rows {sft['slice_start']}..{sft['slice_start'] + sft['slice_rows'] - 1} "
        f"from the base model, {sft['epochs']} epoch, {sft['micro'] * sft['accum']} prompts per step, "
        f"{sft['steps']} steps. Hardware: the local runs on one RTX 4080 (micro-batch {w['micro']} x "
        f"{w['accum']} accumulation for SFT, {r['micro']} x {r['accum']} for RL, gradient checkpointing on); "
        f"the Colab runs on an A100 (micro-batch {cw['micro']} x {cw['accum']} for SFT, {cr['micro']} x "
        f"{cr['accum']} for RL, gradient checkpointing off; the run files do not record the GPU name, the "
        f"notebook asks for an A100 runtime), with the same effective batch. The Colab seed runs started from "
        f"the Colab warmup (seed 0, trained on the A100), not from the local warmup."
    )


def findings(runs: dict[str, dict], arms: dict[str, list[dict]], paired: dict) -> str:
    def ov(key, metric):
        return runs[key]["summary"]["overall"][metric]

    def arm_mean(arm, metric):
        return float(np.mean([m["summary"]["overall"][metric] for m in arms[arm]]))

    def arm_range(arm, metric):
        vals = [m["summary"]["overall"][metric] for m in arms[arm]]
        return min(vals), max(vals)

    def probe_mean(arm, group, key):
        return float(np.mean([m["probe"]["report"][group][key] for m in arms[arm]]))

    def diff(block_prefix, label, metric):
        for (title, lab), d in paired.items():
            if title.startswith(block_prefix) and lab == label:
                return d[metric]
        raise KeyError((block_prefix, label))

    def cell(d):
        return f"{d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

    def excludes_zero(d):
        return d["lo"] > 0 or d["hi"] < 0

    def verdict(ds):
        """One phrase for a metric across the three seeds' paired intervals."""
        n = sum(excludes_zero(d) for d in ds)
        return {0: "none of the three intervals excludes zero", 1: "one of the three intervals excludes zero",
                2: "two of the three intervals exclude zero", 3: "all three intervals exclude zero"}[n]

    seeds = [("Local", "seed 0"), ("Colab, seed 1", "seed 1"), ("Colab, seed 2", "seed 2")]
    rw = {m: [diff(p, "RLCD minus warmup", m) for p, _ in seeds] for m in ("acc", "ece", "brier")}
    rl = {m: [diff(p, "RLCD minus RLVR low lr", m) for p, _ in seeds] for m in ("acc", "ece", "brier")}
    ro = {m: [diff(p, "RLCD minus oracle", m) for p, _ in seeds] for m in ("acc", "ece", "brier")}
    rc = {m: [diff(p, "RLCD minus supervised continuation", m) for p, _ in seeds[1:]] for m in ("acc", "ece", "brier")}
    rs = {m: diff("Local", "RLCD minus 32k-label SFT", m) for m in ("acc", "ece", "brier")}
    os_ = {m: diff("Local", "oracle minus 32k-label SFT", m) for m in ("acc", "ece", "brier")}
    rv = {m: diff("Local", "RLCD minus RLVR shared settings (stopped at step 150)", m) for m in ("acc", "ece", "brier")}

    def rng(pairs):
        return ", ".join(f"{p} {cell(d)}" for (_, p), d in zip(seeds, pairs))

    acc_lo, acc_hi = arm_range("rlcd", "acc")
    ece_lo, ece_hi = arm_range("rlcd", "ece")
    conf_lo, conf_hi = arm_range("rlvr-lowlr", "mean_conf")
    largest_ece_shift = max(rw["ece"], key=lambda d: abs(d["diff"]))
    lowest = min(runs.values(), key=lambda r: r["summary"]["overall"]["ece"])
    lowest_ece_note = "" if lowest["key"] == "local/sft32k" else f" except {lowest['key']}"
    lines = [
        "1. RLCD learns from bandit feedback and keeps the warmup's calibration. Across the three seeds RLCD's test "
        f"accuracy is {arm_mean('rlcd', 'acc'):.3f} [{acc_lo:.3f}, {acc_hi:.3f}] against {arm_mean('warmup', 'acc'):.3f} "
        f"for the warmups it started from, with ECE {arm_mean('rlcd', 'ece'):.3f} [{ece_lo:.3f}, {ece_hi:.3f}] "
        f"(warmups {arm_mean('warmup', 'ece'):.3f}) and mean confidence {arm_mean('rlcd', 'mean_conf'):.3f} against "
        f"accuracy {arm_mean('rlcd', 'acc'):.3f}. Paired against its own warmup, RLCD's accuracy difference is "
        f"{rng(rw['acc'])} ({verdict(rw['acc'])}); Brier {rng(rw['brier'])} ({verdict(rw['brier'])}); ECE "
        f"{rng(rw['ece'])} ({verdict(rw['ece'])}). The accuracy and Brier gains are resolved on every seed. The ECE "
        f"change is at most {largest_ece_shift['diff']:+.3f} and no seed's interval excludes zero, though seed 2's "
        "interval only just includes it. 500 RL steps left calibration within about 0.01 ECE of where the warmup "
        "was; RLCD did not improve it.",

        "2. RLVR with the outcome-only reward does not keep it. At the shared learning rate it collapses inside 50 "
        f"steps and the stop rule ends it at step {runs['local/rlvr']['meta']['steps']}: test accuracy "
        f"{ov('local/rlvr', 'acc'):.3f}, mean confidence {ov('local/rlvr', 'mean_conf'):.3f}, ECE "
        f"{ov('local/rlvr', 'ece'):.3f}, Brier {ov('local/rlvr', 'brier'):.3f} (RLCD minus this run: accuracy "
        f"{cell(rv['acc'])}, ECE {cell(rv['ece'])}). At lr 4e-6 it keeps its accuracy "
        f"({arm_mean('rlvr-lowlr', 'acc'):.3f} mean over three seeds, above the warmups) but its mean confidence is "
        f"{arm_mean('rlvr-lowlr', 'mean_conf'):.3f} [{conf_lo:.3f}, {conf_hi:.3f}], ECE {arm_mean('rlvr-lowlr', 'ece'):.3f} "
        f"and Brier {arm_mean('rlvr-lowlr', 'brier'):.3f}, a worse Brier than the warmup it started from "
        f"({arm_mean('warmup', 'brier'):.3f}). RLCD minus RLVR low lr, per seed: accuracy {rng(rl['acc'])} "
        f"({verdict(rl['acc'])}); ECE {rng(rl['ece'])}; Brier {rng(rl['brier'])}. The two arms differ in the "
        "reward (RLCD subtracts p(taken action) from the outcome) and in the learning rate (RLVR low lr uses "
        f"{runs['local/rlvr-lowlr']['meta']['lr']:g} because at {runs['local/rlvr']['meta']['lr']:g} it collapses); "
        "RLCD was not run at the lower rate.",

        "3. Against the oracle, which sees the label on the same rows, RLCD gives up about one accuracy point and is "
        f"better calibrated. RLCD minus oracle, per seed: accuracy {rng(ro['acc'])} ({verdict(ro['acc'])}); ECE "
        f"{rng(ro['ece'])} ({verdict(ro['ece'])}); Brier {rng(ro['brier'])} ({verdict(ro['brier'])}). The oracle's "
        f"mean confidence is {arm_mean('oracle', 'mean_conf'):.3f} against accuracy {arm_mean('oracle', 'acc'):.3f}: "
        "two epochs of supervised training on the same rows make it overconfident, and bandit feedback with the "
        "RLCD reward does not.",

        "4. The gain is not just more steps. The supervised continuation trained the Colab warmup for 500 more steps "
        f"on its own 6,400 labeled rows and reached accuracy {ov('colab/warmup-cont', 'acc'):.3f} with ECE "
        f"{ov('colab/warmup-cont', 'ece'):.3f} and mean confidence {ov('colab/warmup-cont', 'mean_conf'):.3f}: it "
        "gains accuracy and loses calibration the way the low-lr RLVR arm does. RLCD minus the continuation, seeds 1 "
        f"and 2: accuracy {cell(rc['acc'][0])} and {cell(rc['acc'][1])}; ECE {cell(rc['ece'][0])} and "
        f"{cell(rc['ece'][1])}; Brier {cell(rc['brier'][0])} and {cell(rc['brier'][1])}. 32,000 bandit rows with "
        "the RLCD reward are worth more than five more passes over the 6,400 labels, on accuracy and on calibration.",

        "5. Against the 32k-label SFT reference (500 supervised steps on 32,000 labeled rows, one local run), RLCD "
        f"with 6,400 labels plus 32,000 bandit rows is close but behind: accuracy {cell(rs['acc'])}, ECE "
        f"{cell(rs['ece'])}, Brier {cell(rs['brier'])}. The oracle, which had the labels of the same 32,000 rows "
        f"plus the warmup's 6,400, matches the reference on accuracy ({cell(os_['acc'])}) and Brier "
        f"({cell(os_['brier'])}) and is worse calibrated ({cell(os_['ece'])}). The reference's ECE "
        f"({ov('local/sft32k', 'ece'):.3f}) is the lowest of any run here{lowest_ece_note}: one pass over 32,000 "
        "labels kept its calibration where two passes over the same labels (the oracle) and five passes over 6,400 "
        "(the continuation) lost it.",

        "6. On the known-posterior probe RLCD's confidence moves toward the posterior on the ambiguous tickets. On "
        f"tickets with one cued department (posterior 1.0) RLCD's mean max p is "
        f"{probe_mean('rlcd', 'dept_single', 'mean_max_p'):.3f} (warmups {probe_mean('warmup', 'dept_single', 'mean_max_p'):.3f}); "
        f"on tickets with two cued departments (posterior 0.5 each) it is {probe_mean('rlcd', 'dept_double', 'mean_max_p'):.3f}, "
        f"down from {probe_mean('warmup', 'dept_double', 'mean_max_p'):.3f} for the warmups. The oracle "
        f"({probe_mean('oracle', 'dept_double', 'mean_max_p'):.3f}) and the SFT reference "
        f"({probe_mean('sft32k', 'dept_double', 'mean_max_p'):.3f}) get there with the labels; RLVR low lr says "
        f"{probe_mean('rlvr-lowlr', 'dept_double', 'mean_max_p'):.3f} on the same ambiguous tickets, and the supervised "
        f"continuation moves the wrong way ({probe_mean('warmup-cont', 'dept_double', 'mean_max_p'):.3f}). One "
        f"difference from the labeled arms: RLCD keeps {probe_mean('rlcd', 'dept_double', 'mean_mass_on_cued'):.3f} of "
        f"the mass on the two cued departments against {probe_mean('oracle', 'dept_double', 'mean_mass_on_cued'):.3f} "
        f"for the oracle and {probe_mean('warmup', 'dept_double', 'mean_mass_on_cued'):.3f} for the warmups, so part "
        "of its lower max p is mass that leaked to the uncued departments rather than a fairer split between the "
        "cued two. "
        f"On the escalate question (posterior 0.90) RLCD's mean confidence is "
        f"{probe_mean('rlcd', 'escalate', 'mean_conf'):.3f}, the oracle's {probe_mean('oracle', 'escalate', 'mean_conf'):.3f}, "
        f"the warmups' {probe_mean('warmup', 'escalate', 'mean_conf'):.3f}, the SFT reference's "
        f"{probe_mean('sft32k', 'escalate', 'mean_conf'):.3f} and RLVR low lr's "
        f"{probe_mean('rlvr-lowlr', 'escalate', 'mean_conf'):.3f}.",
    ]
    return "\n\n".join(lines)


def fp32_drift(runs: dict[str, dict]) -> dict[str, float]:
    """runs/q-rlcd scored in fp32 by the decision API (runs/decide-bench/preds_ask.jsonl, rows grouped by
    context) against its bf16 evaluation, on the same rows: the size of the precision-only difference."""
    fp32 = {p["id"]: p for p in load_preds(FP32_PREDS)}
    bf16 = runs["local/rlcd"]["preds"]
    if set(fp32) != {p["id"] for p in bf16}:
        raise SystemExit("runs/decide-bench/preds_ask.jsonl does not cover the test rows of runs/q-rlcd")
    a = summarize([fp32[p["id"]] for p in bf16])["overall"]
    b = runs["local/rlcd"]["summary"]["overall"]
    return {k: abs(a[k] - b[k]) for k in ("acc", "ece", "brier")}


def rlvr_attempts(runs: dict[str, dict]) -> tuple[int, float, float]:
    """Validation accuracy of the two local shared-settings RLVR attempts (same seed and command) at
    their first evaluation after step 0."""
    first = [r for r in read_log(RLVR_FIRST_ATTEMPT) if "eval_step" in r and r["eval_step"] > 0]
    second = [r for r in runs["local/rlvr"]["log"] if "eval_step" in r and r["eval_step"] > 0]
    if first[0]["eval_step"] != second[0]["eval_step"]:
        raise SystemExit("the two RLVR attempts evaluated at different steps")
    return first[0]["eval_step"], first[0]["eval_acc"], second[0]["eval_acc"]


def caveats(runs: dict[str, dict], arms: dict[str, list[dict]]) -> str:
    rlvr = runs["local/rlvr"]["meta"]
    n_test = runs["local/rlcd"]["eval_meta"]["n"]
    max_len = runs["local/rlcd"]["eval_meta"]["max_len"]
    ece_bias = max(abs(r["summary"]["overall"]["ece_bias"]) for r in runs.values())
    esc = [m["probe"]["report"]["escalate"]["mean_conf"] for m in arms["rlcd"]]
    drift = fp32_drift(runs)
    step, acc_first, acc_second = rlvr_attempts(runs)
    peaks = []  # (peak val ECE, step, final val ECE, key) per RLCD run
    for m in arms["rlcd"]:
        evals = [r for r in m["log"] if "eval_step" in r and r["eval_step"] > 0]
        top = max(evals, key=lambda r: r["eval_ece"])
        peaks.append((top["eval_ece"], top["eval_step"], evals[-1]["eval_ece"], m["key"]))
    peak_text = "; ".join(f"{key} peaks at {p:.3f} at step {s} and ends at {e:.3f}" for p, s, e, key in peaks)
    lines = [
        "- One warmup per hardware. The local RL arms all started from the local warmup (seed 0, RTX 4080); the "
        "Colab seed 1 and seed 2 arms both started from the one Colab warmup (seed 0, A100). The warmup row of the "
        "main table therefore averages two runs, not three, and the two Colab seeds share their starting point.",
        "- \"Seed\" varies more than the sampling. Seeds 1 and 2 differ from seed 0 in the RL sampling seed, in "
        "the warmup checkpoint, in the GPU, in the micro-batch shape and in gradient checkpointing; the effective "
        "batch and every other setting are the same. The seed spread in the tables therefore includes hardware "
        "and warmup variation, and a difference between seed 0 and seeds 1 and 2 is not a pure seed effect.",
        "- bf16 is not deterministic. Training and the evaluations ran under bf16 autocast, and a bf16 kernel's "
        "rounding changes with the batch shape. Scoring runs/q-rlcd in fp32 through the decision API "
        "(runs/decide-bench/preds_ask.jsonl) instead of the bf16 evaluation moves accuracy by "
        f"{small(drift['acc'])}, ECE by {small(drift['ece'])} and Brier by {small(drift['brier'])} on the same "
        "8000 rows. "
        "Two local shared-settings RLVR attempts with the same seed and command took different trajectories "
        f"(validation accuracy {acc_first:.3f} and {acc_second:.3f} at step {step}, from runs/q-rlvr and "
        "runs/q-rlvr-b), so training is not bit-reproducible either. The third decimal is printed so the tables "
        "agree with the files; it is not stable.",
        f"- The shared-settings RLVR arm has one run, stopped at step {rlvr['steps']} by the rule "
        f"\"{rlvr['stopped']}\", and its numbers are those of the stopped checkpoint; it was not rerun on Colab. "
        "The three-seed RLVR comparison is the low-lr arm.",
        f"- ECE. Plug-in ECE is biased upward and its percentile bootstrap interval inherits the bias (the largest "
        f"bootstrap-mean-minus-point-estimate across these runs is {ece_bias:.3f}), so the per-run intervals of "
        "well calibrated runs sit above the point estimate and overlap of two runs' intervals says little. The "
        "paired differences resample the same rows for both runs, so the row-to-row agreement cancels; those are "
        "the intervals to read. They cover test-row sampling only, not training noise: a paired interval that "
        "excludes zero on one seed says that one pair of checkpoints differs on this split, and the three seeds "
        "are the only estimate of training noise here.",
        "- The supervised continuation is one run on one warmup at a batch of 64 prompts per step (the warmup's "
        "batch), not the RL arms' 128; it answers \"does 500 more supervised steps on the same 6,400 labels give "
        "what RLCD gives\", not \"does any supervised schedule\". The 32k-label SFT reference trained on rows "
        "0..31999 from the base model, while the RL arms saw 6,400 of those rows as labels plus rows 32000..63999 as "
        "bandit rows, so that comparison is by row count, not identical rows.",
        f"- Everything is in-distribution: the test split is a held-out slice of the same public classification "
        f"datasets and the same synthetic triage generator the model trained on, {n_test:,} rows, prompts of at most "
        f"{max_len} tokens. The known-posterior probe is fresh synthetic triage tickets whose true posterior the "
        "generator fixes; it says nothing about calibration on real tickets or on longer prompts. No out-of-"
        "distribution evaluation was run.",
        f"- RLCD's calibration during training is not flat. Validation ECE over the 500 RL steps (see curves.png): "
        f"{peak_text}. The final checkpoint is what is reported; no checkpoint selection was done, but a reader "
        "should not take the final ECE as a guarantee at every step. On the escalate probe "
        f"question RLCD's mean confidence is {', '.join(f'{v:.3f}' for v in esc)} across the seeds against a "
        "posterior of 0.90, so it sits a little above the posterior there.",
        "- The stopped RLVR arm and the low-lr arm are the same reward at two learning rates, and the low-lr arm "
        f"is compared with RLCD at {runs['local/rlcd']['meta']['lr']:g}, five times its rate; RLCD at "
        f"{runs['local/rlvr-lowlr']['meta']['lr']:g} was not run. No other RLVR regularization (a KL term, an "
        "entropy bonus, a temperature) was tried on this base, so \"RLVR miscalibrates\" means \"this "
        "REINFORCE-with-outcome-reward recipe miscalibrates at these settings\", not that no outcome-reward recipe "
        "could be made to work.",
    ]
    return "\n".join(lines)


# ---------- document ----------

def build_document(runs: dict[str, dict]) -> str:
    arms = by_arm(runs)
    figures = make_figures(runs, arms)
    paired_md, paired = paired_block(runs)
    parts = [
        "# Results: RLCD on Qwen3-0.6B-Base",
        "",
        "Generated by `scripts/consolidate_results.py` from the run directories under `runs/`; every number is "
        "computed from `preds.jsonl`, `probe.json`, `meta.json` and `train_log.jsonl` with `rlcd.eval` and "
        "`rlcd.metrics`. Do not edit by hand. Lower is better for ECE, Brier loss and NOTA false alarm; higher is "
        "better for accuracy and NOTA recall. Cells with three values are mean [min, max] across the runs of "
        "that arm.",
        "",
        "## Setup",
        "",
        setup_paragraph(runs),
        "",
        "## Runs",
        "",
        "One row per evaluated checkpoint. `local` is the RTX 4080, `colab` the A100. The `q-rlvr` row of the "
        "local report is `runs/q-rlvr-b`, the rerun that the built-in stop rule ended; the first attempt "
        "(`runs/q-rlvr`) was ended by an external zero-gradient watch at step 140 and has no evaluation.",
        "",
        run_table(runs),
        "",
        "## Main table",
        "",
        "Mean [min, max] across the runs of each arm on the 8000-row test split at temperature 1. The warmup "
        "row averages the local warmup and the Colab warmup; the two Colab seed runs of every RL arm share the "
        "Colab warmup, so it is one run evaluated once, not two.",
        "",
        main_table(arms),
        "",
        "## Paired comparisons",
        "",
        "Each difference is the first run minus the second on the same 8000 rows, with a 95 percent percentile "
        "interval of a paired bootstrap (1000 resamples, seed 0; every resample applies the same row indices to "
        "both runs, `rlcd.metrics.paired_bootstrap_diff`). Pairs are formed only within one warmup: local seed 0 "
        "against the local warmup, each Colab seed against the Colab warmup and against the other arms of that "
        "seed. The interval covers test-row sampling, not training noise. Negative ECE and Brier differences "
        "favor the first run; positive accuracy differences favor the first run.",
        "",
        paired_md,
        "## Known-posterior probe",
        "",
        "3000 fresh synthetic tickets (probe seed 12345, 2 questions dropped for overlap with training contexts), "
        "department and escalate questions, no NOTA option. The generator fixes the true posterior: 1.0 on the "
        "cued department when one department is cued, 0.5 each when two are cued, and 0.90 on the escalate answer "
        "the priority implies. Mean [min, max] across the runs of each arm; the ideal value is in the header.",
        "",
        probe_table(arms),
        "",
        "## Per-source accuracy and ECE",
        "",
        "Mean across the runs of the arm, from each run's `summary[\"by_source\"]`.",
        "",
        source_tables(arms, ["warmup", "rlcd", "oracle", "sft32k"]),
        "## Figures",
        "",
        "\n".join(f"- `docs/img/{name}`: {desc}" for name, desc in figures.items()),
        "",
        "## What the numbers show",
        "",
        findings(runs, arms, paired),
        "",
        "## Caveats",
        "",
        caveats(runs, arms),
        "",
        "## Per-run numbers with bootstrap intervals",
        "",
        "For reference: each run on its own, with 95 percent percentile bootstrap intervals over the test rows "
        "(1000 resamples, seed 0). ECE bias is the bootstrap mean minus the point estimate; see the caveat on ECE.",
        "",
        per_run_table(runs),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    runs = load_runs()
    text = build_document(runs)
    if any(ord(ch) > 127 for ch in text):
        bad = sorted({ch for ch in text if ord(ch) > 127})
        raise SystemExit(f"non-ASCII characters in the document: {bad!r}")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DOC} and {sorted(p.name for p in IMG.glob('*.png'))}")


if __name__ == "__main__":
    main()
