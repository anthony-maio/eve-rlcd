"""Predictions, metrics with bootstrap intervals, plots, and a known-posterior probe for one
checkpoint; comparison tables, plots, and training curves across checkpoints."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zlib
from pathlib import Path

import numpy as np

from rlcd.data import DEPARTMENTS, DEPT_CUES, PRIO_CUES, PRIORITIES, shuffle_choices, synthetic_triage
from rlcd.metrics import (bootstrap_ci, brier, coverage_error, ece, nota_false_alarm, nota_rate,
                          reliability_bins)
from rlcd.quick_eval import log_probs, predict_logits, stride_sample
from rlcd.schema import NOTA, Question, letter_token_ids, read_jsonl

N_BINS = 15


# ---------- json ----------

def _json_safe(obj):
    """Plain python types, with NaN and infinities turned into None so the output is strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def to_json(obj) -> str:
    return json.dumps(_json_safe(obj), indent=2, allow_nan=False)


def load_preds(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------- predictions and metrics ----------

def predict(model, tok, questions: list[Question], max_len: int = 512, batch_size: int = 32,
            device: str = "cuda", letters: list[int] | None = None) -> list[dict]:
    """One row per question with the first k decision logits. letters defaults to the
    tokenizer's letter token ids; tests pass their own."""
    letters = letter_token_ids(tok) if letters is None else letters
    rows = predict_logits(model, tok, letters, questions, max_len, batch_size, device)
    return [{"id": q.id, "source": q.source, "primitive": q.primitive, "k": q.k, "answer": q.answer,
             "logits": row, "nota_index": q.choices.index(NOTA) if NOTA in q.choices else -1}
            for q, row in zip(questions, rows)]


def _arrays(preds: list[dict], temperature: float):
    probs = np.exp(log_probs([p["logits"] for p in preds], temperature))
    answers = np.array([p["answer"] for p in preds])
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == answers).astype(float)
    nota_index = np.array([p["nota_index"] for p in preds])
    return probs, answers, pred, conf, correct, nota_index


def _row_brier(probs: np.ndarray, answers: np.ndarray) -> np.ndarray:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(answers)), answers] = 1.0
    return ((probs - onehot) ** 2).sum(1)


def _group_metrics(preds: list[dict], temperature: float, with_ci: bool = False) -> dict:
    probs, answers, pred, conf, correct, nota_index = _arrays(preds, temperature)
    acc, mean_conf = float(correct.mean()), float(conf.mean())
    out = {"n": len(preds), "acc": acc, "brier": brier(probs, answers), "ece": ece(conf, correct, N_BINS),
           "nota_rate": nota_rate(pred, answers, nota_index),
           "nota_false_alarm": nota_false_alarm(pred, answers, nota_index),
           "mean_conf": mean_conf, "conf_minus_acc": mean_conf - acc}
    if with_ci:
        out["ci"] = {"acc": list(bootstrap_ci(lambda c: float(c.mean()), (correct,))),
                     "ece": list(bootstrap_ci(lambda f, c: ece(f, c, N_BINS), (conf, correct))),
                     "brier": list(bootstrap_ci(lambda b: float(b.mean()), (_row_brier(probs, answers),)))}
    return out


def summarize(preds: list[dict], temperature: float = 1.0) -> dict:
    """Metrics overall, by primitive, and by source. Brier is a loss: lower is better. overall
    also carries 95 percent bootstrap intervals (1000 row resamples, seed 0) under "ci"."""
    result = {"overall": _group_metrics(preds, temperature, with_ci=True), "by_primitive": {}, "by_source": {}}
    for key, field in (("by_primitive", "primitive"), ("by_source", "source")):
        for name in sorted({p[field] for p in preds}):
            result[key][name] = _group_metrics([p for p in preds if p[field] == name], temperature)
    return result


# ---------- plots ----------

# One fixed look per run, used by every figure. Known runs are listed; any other name gets a
# stable palette slot from its crc32 and a dashed line, so it looks the same in every figure.
PALETTE = ["#2a78d6", "#e34948", "#008300", "#eda100", "#4a3aa7", "#eb6834", "#1baf7a", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
RUN_STYLE = {
    "rlcd": ("#2a78d6", "o", "-"),
    "rlvr": ("#e34948", "X", "-"),
    "rlvr-kl05": ("#eda100", "s", "-"),
    "rlvr-lr1e5": ("#e87ba4", "D", "-"),
    "oracle": ("#008300", "^", "-"),
    "warmup": ("#4a3aa7", "v", "-"),
    "diag-lr5e-5": ("#1baf7a", "P", "-"),
    "rlcd-kl05": ("#eb6834", "h", "-"),
    "zeroshot": ("#6f6d68", "<", ":"),
}


def run_style(name: str) -> dict:
    if name in RUN_STYLE:
        color, marker, linestyle = RUN_STYLE[name]
    else:
        h = zlib.crc32(name.encode("utf-8"))
        color, marker, linestyle = PALETTE[h % len(PALETTE)], MARKERS[(h // 8) % len(MARKERS)], "--"
    return {"color": color, "marker": marker, "linestyle": linestyle}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(fig, axes, out_png, legend_ax=None):
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    (legend_ax or axes[0]).legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False)
    fig.patch.set_facecolor("white")
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")


def plot_reliability(preds_by_name: dict[str, list[dict]], out_png, temperature_by_name=None):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], color="black", ls="--", lw=1, label="perfect calibration")
    for name, preds in preds_by_name.items():
        t = (temperature_by_name or {}).get(name, 1.0)
        _, _, _, conf, correct, _ = _arrays(preds, t)
        bc, ba, bn = reliability_bins(conf, correct, N_BINS)
        m = bn > 0
        ax.plot(bc[m], ba[m], lw=1.6, ms=5, label=f"{name} (ECE {ece(conf, correct, N_BINS):.3f})",
                **run_style(name))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel(f"stated confidence (mean of each of {N_BINS} equal-width bins)")
    ax.set_ylabel("fraction correct in the bin")
    _finish(fig, [ax], out_png)
    plt.close(fig)


def plot_coverage(preds_by_name: dict[str, list[dict]], out_png, temperature_by_name=None):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, preds in preds_by_name.items():
        t = (temperature_by_name or {}).get(name, 1.0)
        _, _, _, conf, correct, _ = _arrays(preds, t)
        cov, err = coverage_error(conf, correct)
        style = run_style(name)
        ax.plot(cov, err, lw=1.6, ms=5, markevery=max(len(cov) // 10, 1), label=name, **style)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage (fraction answered, most confident first)")
    ax.set_ylabel("error rate among answered")
    _finish(fig, [ax], out_png)
    plt.close(fig)


def plot_curves(evals_by_name: dict[str, list[dict]], out_png):
    plt = _pyplot()
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for name, rows in evals_by_name.items():
        steps = [r["eval_step"] for r in rows]
        style = run_style(name)
        top.plot(steps, [r["eval_acc"] for r in rows], lw=1.6, ms=5, label=name, **style)
        with_ece = [r for r in rows if "eval_ece" in r]  # logs older than the ECE column have none
        bottom.plot([r["eval_step"] for r in with_ece], [r["eval_ece"] for r in with_ece], lw=1.6, ms=5,
                    label=name, **style)
    top.set_ylabel("held-out accuracy")
    bottom.set_ylabel(f"held-out ECE ({N_BINS} bins)")
    bottom.set_xlabel("optimizer step")
    bottom.set_ylim(bottom=0)
    _finish(fig, [top, bottom], out_png, legend_ax=top)
    plt.close(fig)


# ---------- known-posterior probe ----------

PROBE_IDEAL = {"dept_single": {"acc": 1.0, "mean_p_cued": 1.0},
               "dept_double": {"acc": 0.5, "mean_max_p": 0.5, "mean_mass_on_cued": 1.0,
                               "mean_abs_dev_first_from_half": 0.0},
               "escalate": {"acc": 0.9, "mean_conf": 0.9, "mean_p_implied": 0.9}}


def _cued_departments(context: str) -> list[str]:
    """Departments whose cue phrases appear in the ticket's report text, in text order."""
    report = context.split("Report: ", 1)[1].rsplit(". Note: ", 1)[0]
    found = sorted((report.index(phrase), dept) for dept, phrases in DEPT_CUES.items()
                   for phrase in phrases if phrase in report)
    return [dept for _, dept in found]


def _implied_escalate(context: str) -> int:
    """The escalate answer the note text implies before label noise: 0 (true) for P1 and P0."""
    note = context.rsplit(". Note: ", 1)[1]
    hits = [i for i, prio in enumerate(PRIORITIES) if any(note == phrase + "." for phrase in PRIO_CUES[prio])]
    if len(hits) != 1:
        raise AssertionError(f"note does not name exactly one priority: {note!r}")
    return 0 if hits[0] >= 2 else 1


def build_probe(n_tickets: int, seed: int, train_contexts: set[str]) -> tuple[list[tuple[Question, dict]], int]:
    """Department and escalate questions of fresh synthetic tickets, without NOTA and with the
    department options shuffled, each paired with what is known about its true posterior.
    Questions whose context occurs in train_contexts are dropped; the count is returned."""
    rng = random.Random(seed)
    tickets = synthetic_triage(n_tickets, rng)
    items: list[tuple[Question, dict]] = []
    dropped = 0
    for dept_q, prio_q, esc_q in zip(tickets[0::3], tickets[1::3], tickets[2::3]):
        if not (dept_q.id.endswith("-dept") and prio_q.id.endswith("-prio") and esc_q.id.endswith("-esc")):
            raise AssertionError("synthetic_triage no longer yields dept, prio, esc triples")
        cued = _cued_departments(dept_q.context)
        if len(cued) not in (1, 2) or len(set(cued)) != len(cued) or DEPARTMENTS[dept_q.answer] not in cued:
            raise AssertionError(f"cue recovery failed for {dept_q.id}: {cued}")
        implied = _implied_escalate(esc_q.context)
        if implied != (0 if prio_q.answer >= 2 else 1):
            raise AssertionError(f"note text and priority label disagree for {esc_q.id}")
        if dept_q.context in train_contexts:
            dropped += 2
            continue
        shuffled = shuffle_choices(dept_q, rng)
        items.append((shuffled, {"kind": "dept", "cued": [shuffled.choices.index(d) for d in cued]}))
        items.append((esc_q, {"kind": "escalate", "implied": implied}))
    return items, dropped


def _mean(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def probe_report(rows: list[dict]) -> dict:
    """Calibration against known posteriors. Rows are plain dicts: kind ("dept" or "escalate"),
    probs over the options, label, and for dept the option indices of the cued departments in
    text order, for escalate the option index implied by the priority."""
    single = [r for r in rows if r["kind"] == "dept" and len(r["cued"]) == 1]
    double = [r for r in rows if r["kind"] == "dept" and len(r["cued"]) == 2]
    esc = [r for r in rows if r["kind"] == "escalate"]

    def acc(group):
        return _mean(float(int(np.argmax(r["probs"])) == r["label"]) for r in group)

    return {
        "dept_single": {"n": len(single), "acc": acc(single),
                        "mean_p_cued": _mean(r["probs"][r["cued"][0]] for r in single)},
        "dept_double": {"n": len(double), "acc": acc(double),
                        "mean_max_p": _mean(max(r["probs"]) for r in double),
                        "mean_mass_on_cued": _mean(r["probs"][r["cued"][0]] + r["probs"][r["cued"][1]]
                                                   for r in double),
                        "mean_abs_dev_first_from_half": _mean(abs(r["probs"][r["cued"][0]] - 0.5)
                                                              for r in double)},
        "escalate": {"n": len(esc), "acc": acc(esc), "mean_conf": _mean(max(r["probs"]) for r in esc),
                     "mean_p_implied": _mean(r["probs"][r["implied"]] for r in esc)},
    }


def _fmt(x, digits: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def probe_tables(reports: dict[str, dict]) -> str:
    """Three markdown tables, one per probe group, with the ideal values in the header row."""
    specs = [
        ("Department question, one cued department (true posterior 1.0 on the cued department)", "dept_single",
         [("acc", "accuracy (ideal 1.0)"), ("mean_p_cued", "mean p(cued department) (ideal 1.0)")]),
        ("Department question, two cued departments (true posterior 0.5 on each)", "dept_double",
         [("acc", "accuracy (ideal 0.5)"), ("mean_max_p", "mean max probability (ideal 0.5)"),
          ("mean_mass_on_cued", "mean mass on the two cued (ideal 1.0)"),
          ("mean_abs_dev_first_from_half", "mean abs(p(first cued) - 0.5) (ideal 0.0)")]),
        ("Escalate question (true posterior 0.90 on the answer implied by the priority)", "escalate",
         [("acc", "accuracy (ideal 0.90)"), ("mean_conf", "mean confidence (ideal 0.90)"),
          ("mean_p_implied", "mean p(implied answer) (ideal 0.90)")]),
    ]
    lines = []
    for title, group, cols in specs:
        lines += [f"**{title}**", "", "| run | n | " + " | ".join(label for _, label in cols) + " |",
                  "|---" * (2 + len(cols)) + "|"]
        for name, report in reports.items():
            g = report[group]
            lines.append(f"| {name} | {g['n']} | " + " | ".join(_fmt(g[key]) for key, _ in cols) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------- commands ----------

def _write_preds(path, preds: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")


def cmd_run(args):
    from rlcd.policy import load_eve
    model, tok = load_eve(args.model, device=args.device)
    qs = read_jsonl(args.split)
    if args.limit:
        qs = stride_sample(qs, args.limit)
    preds = predict(model, tok, qs, args.max_len, args.batch_size, args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_preds(out / "preds.jsonl", preds)
    summary = summarize(preds, args.temperature)
    summary["meta"] = {"model": str(args.model), "split": str(args.split), "n": len(preds),
                       "temperature": args.temperature, "max_len": args.max_len, "n_bins": N_BINS,
                       "note": "brier is a loss, lower is better; ci is a 95 percent bootstrap interval"}
    (out / "metrics.json").write_text(to_json(summary) + "\n")
    name = args.name or Path(args.model).name
    plot_reliability({name: preds}, out / "reliability.png", {name: args.temperature})
    plot_coverage({name: preds}, out / "coverage.png", {name: args.temperature})
    print(to_json(summary["overall"]))


def cmd_probe(args):
    from rlcd.policy import load_eve
    train_contexts = {q.context for q in read_jsonl(args.train)}
    items, dropped = build_probe(args.n_tickets, args.seed, train_contexts)
    model, tok = load_eve(args.model, device=args.device)
    preds = predict(model, tok, [q for q, _ in items], args.max_len, args.batch_size, args.device)
    probs = np.exp(log_probs([p["logits"] for p in preds]))
    rows = [{**meta, "id": q.id, "label": q.answer, "probs": probs[i, : q.k].tolist()}
            for i, (q, meta) in enumerate(items)]
    report = probe_report(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_preds(out / "probe_rows.jsonl", rows)
    (out / "probe.json").write_text(to_json({
        "model": str(args.model), "n_tickets": args.n_tickets, "seed": args.seed,
        "questions_dropped_for_train_overlap": dropped, "ideal": PROBE_IDEAL, "report": report}) + "\n")
    print(f"dropped {dropped} probe questions whose context occurs in {args.train}")
    print(probe_tables({args.name or Path(args.model).name: report}))


def _named_paths(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"expected name=path, got {item!r}")
        out[name] = Path(path)
    return out


def _run_meta(eval_dir: Path) -> dict:
    """steps and stopped from the run's meta.json when the eval dir sits inside a run dir."""
    meta_path = eval_dir.parent / "meta.json"
    if not meta_path.is_file():
        return {"steps": None, "stopped": ""}
    meta = json.loads(meta_path.read_text())
    return {"steps": meta.get("steps"), "stopped": meta.get("stopped") or ""}


def _with_ci(m: dict, key: str) -> str:
    lo, hi = m["ci"][key]
    return f"{m[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def cmd_compare(args):
    runs = _named_paths(args.runs)
    preds_by_name = {name: load_preds(path / "preds.jsonl") for name, path in runs.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = {name: [p["id"] for p in preds] for name, preds in preds_by_name.items()}
    first = next(iter(ids))
    for name in ids:
        if ids[name] != ids[first]:
            print(f"WARNING: {name} and {first} were not evaluated on the same rows", file=sys.stderr)

    summaries = {name: summarize(preds) for name, preds in preds_by_name.items()}
    rows = ["| run | steps | n | acc [95% CI] | Brier loss, lower is better [95% CI] | ECE [95% CI] | mean conf "
            "| conf minus acc | NOTA recall | NOTA false alarm |", "|---" * 10 + "|"]
    table = {}
    for name, summary in summaries.items():
        m = summary["overall"]
        meta = _run_meta(runs[name])
        steps = "n/a" if meta["steps"] is None else str(meta["steps"]) + (" (stopped)" if meta["stopped"] else "")
        table[name] = {**m, **meta, "by_source": summary["by_source"], "by_primitive": summary["by_primitive"]}
        rows.append(f"| {name} | {steps} | {m['n']} | {_with_ci(m, 'acc')} | {_with_ci(m, 'brier')} "
                    f"| {_with_ci(m, 'ece')} | {m['mean_conf']:.3f} | {m['conf_minus_acc']:+.3f} "
                    f"| {_fmt(m['nota_rate'])} | {_fmt(m['nota_false_alarm'])} |")
    (out / "table.md").write_text("\n".join(rows) + "\n")
    (out / "metrics.json").write_text(to_json(table) + "\n")

    sources = sorted({s for summary in summaries.values() for s in summary["by_source"]})
    lines = []
    for title, key in (("Accuracy by source", "acc"), (f"ECE by source ({N_BINS} bins)", "ece")):
        lines += [f"**{title}**", "", "| run | " + " | ".join(sources) + " |", "|---" * (1 + len(sources)) + "|"]
        for name, summary in summaries.items():
            cells = [_fmt(summary["by_source"].get(s, {}).get(key)) for s in sources]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "by_source.md").write_text("\n".join(lines))

    plot_reliability(preds_by_name, out / "reliability.png")
    plot_coverage(preds_by_name, out / "coverage.png")

    probes = {name: json.loads((path / "probe.json").read_text())["report"]
              for name, path in runs.items() if (path / "probe.json").is_file()}
    if probes:
        (out / "probe.md").write_text(probe_tables(probes))
    print("\n".join(rows))
    print()
    print("\n".join(lines))
    if probes:
        print(probe_tables(probes))


def cmd_curves(args):
    evals = {}
    for name, path in _named_paths(args.runs).items():
        with open(path / "train_log.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        evals[name] = [r for r in rows if "eval_step" in r]
        if not evals[name]:
            raise SystemExit(f"{path / 'train_log.jsonl'} has no eval rows")
        if not any("eval_ece" in r for r in evals[name]):
            print(f"note: {name} logged no eval_ece, so it appears in the accuracy panel only")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(evals, out / args.name)
    print(f"wrote {out / args.name}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_args(p):
        p.add_argument("--model", required=True, help="checkpoint directory or hub id")
        p.add_argument("--out", required=True)
        p.add_argument("--name", default=None, help="run name for plots and tables (default: model dir name)")
        p.add_argument("--max-len", type=int, default=512)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--device", default="cuda")

    r = sub.add_parser("run", help="predict a split, write preds.jsonl, metrics.json, and two plots")
    model_args(r)
    r.add_argument("--split", default="data/test.jsonl")
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--limit", type=int, default=0, help="evaluate an even stride sample of this many rows")
    r.set_defaults(fn=cmd_run)

    p = sub.add_parser("probe", help="calibration against the known posteriors of synthetic triage")
    model_args(p)
    p.add_argument("--n-tickets", type=int, default=3000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--train", default="data/train.jsonl", help="probe contexts found in this file are dropped")
    p.set_defaults(fn=cmd_probe)

    c = sub.add_parser("compare", help="tables and overlaid plots across eval directories")
    c.add_argument("--runs", nargs="+", required=True, help="name=path/to/eval-dir")
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_compare)

    v = sub.add_parser("curves", help="held-out accuracy and ECE against optimizer step")
    v.add_argument("--runs", nargs="+", required=True, help="name=path/to/run-dir")
    v.add_argument("--out", required=True)
    v.add_argument("--name", default="curves.png", help="file name of the figure inside --out")
    v.set_defaults(fn=cmd_curves)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
