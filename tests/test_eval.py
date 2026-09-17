import json
import math
import random

import numpy as np
import torch

from rlcd.calibrate import fit_temperature
from rlcd.compat import eve_config
from rlcd.data import DEPARTMENTS, synthetic_triage
from rlcd.eval import (ECE_NOTE, FALLBACK_PALETTE, MIN_BIN_ROWS, RUN_STYLE, build_probe, curve_points, main,
                       overall_text, predict, probe_report, probe_tables, reliability_points, run_style,
                       stopped_step, summarize, to_json, training_label, write_rows)
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policies import EvePolicy
from rlcd.schema import NOTA, Question
from rlcd.schema import write_jsonl as write_questions

GROUP_KEYS = {"n", "acc", "brier", "ece", "nota_rate", "nota_false_alarm", "mean_conf", "conf_minus_acc"}


def _pred(logits, answer, primitive="choice", source="s", nota_index=-1):
    return {"id": "x", "source": source, "primitive": primitive, "k": len(logits),
            "answer": answer, "logits": logits, "nota_index": nota_index}


def test_summarize_basic():
    preds = [_pred([2.0, 0.0], 0), _pred([0.0, 2.0], 0), _pred([0.0, 0.0, 3.0], 2, "score", "t")]
    s = summarize(preds)
    assert s["overall"]["n"] == 3
    assert abs(s["overall"]["acc"] - 2 / 3) < 1e-9
    assert set(s["by_primitive"]) == {"choice", "score"}
    assert set(s["by_source"]) == {"s", "t"}
    assert 0.0 <= s["overall"]["ece"] <= 1.0
    assert math.isnan(s["overall"]["nota_rate"])


def test_summarize_group_keys_and_confidence_gap():
    preds = [_pred([2.0, 0.0], 0), _pred([0.0, 2.0], 0), _pred([0.0, 0.0, 3.0], 2, "score", "t")]
    s = summarize(preds)
    assert set(s["overall"]) == GROUP_KEYS | {"ci", "ece_bias", "reliability"}
    for group in list(s["by_primitive"].values()) + list(s["by_source"].values()):
        assert set(group) == GROUP_KEYS
    o = s["overall"]
    assert abs(o["conf_minus_acc"] - (o["mean_conf"] - o["acc"])) < 1e-12


def test_summarize_overall_has_bootstrap_intervals():
    rng = np.random.default_rng(0)
    preds = [_pred(rng.normal(size=3).tolist(), int(rng.integers(3))) for _ in range(300)]
    o = summarize(preds)["overall"]
    assert set(o["ci"]) == {"acc", "ece", "brier"}
    for key in ("acc", "brier"):
        lo, hi = o["ci"][key]
        assert lo < o[key] < hi
    lo, hi = o["ci"]["ece"]
    assert 0.0 <= lo < hi <= 1.0
    assert summarize(preds)["overall"]["ci"] == o["ci"]


def test_summarize_reports_nota_recall_and_false_alarm():
    preds = [_pred([0.0, 0.0, 5.0], 2, nota_index=2),   # NOTA is the answer and is predicted
             _pred([5.0, 0.0, 0.0], 2, nota_index=2),   # NOTA is the answer and is missed
             _pred([0.0, 0.0, 5.0], 0, nota_index=2),   # NOTA offered as a distractor and predicted
             _pred([5.0, 0.0, 0.0], 0, nota_index=2),
             _pred([5.0, 0.0, 0.0], 0, nota_index=2),
             _pred([5.0, 0.0, 0.0], 0, nota_index=2)]
    o = summarize(preds)["overall"]
    assert o["nota_rate"] == 0.5
    assert o["nota_false_alarm"] == 0.25


def test_summarize_temperature_flattens_confidence():
    preds = [_pred([4.0, 0.0], 0)]
    assert summarize(preds, temperature=1.0)["overall"]["mean_conf"] > summarize(preds, temperature=4.0)["overall"]["mean_conf"]


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(2000):
        true_logits = rng.normal(size=4)
        answer = int(rng.choice(4, p=np.exp(true_logits) / np.exp(true_logits).sum()))
        preds.append(_pred((true_logits * 3.0).tolist(), answer))  # overconfident by 3x
    t = fit_temperature(preds)
    assert 2.4 < t < 3.6


def test_nan_metrics_serialize_as_json_null():
    s = summarize([_pred([2.0, 0.0], 0), _pred([0.0, 2.0], 0)])
    assert math.isnan(s["overall"]["nota_rate"])
    text = to_json(s)
    assert "NaN" not in text

    def reject(token):
        raise AssertionError(f"non-standard JSON constant {token}")

    back = json.loads(text, parse_constant=reject)
    assert back["overall"]["nota_rate"] is None
    assert back["overall"]["nota_false_alarm"] is None
    assert back["overall"]["n"] == 2


class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def test_predict_rows_carry_ids_logits_and_nota_index():
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=128,
                     num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    model = EveMoEForCausalLM(cfg)
    qs = [Question("choice", "ctx", "pick", ["x", "y", NOTA], answer=2, source="alpha", id="a-1"),
          Question("noul", "ctx", "is it", ["true", "false"], answer=1, source="beta", id="b-1")]
    policy = EvePolicy(model, FakeTok(), list(range(100, 126)))
    preds = predict(policy, qs, max_len=64, batch_size=1, device="cpu")
    assert [set(p) for p in preds] == [{"id", "source", "primitive", "k", "answer", "logits", "nota_index"}] * 2
    assert [p["id"] for p in preds] == ["a-1", "b-1"]
    assert [len(p["logits"]) for p in preds] == [3, 2]
    assert [p["nota_index"] for p in preds] == [2, -1]
    assert [p["answer"] for p in preds] == [2, 1]
    assert summarize(preds)["overall"]["n"] == 2


# ---------- known-posterior probe ----------

def test_probe_report_on_hand_made_rows():
    rows = [
        {"kind": "dept", "probs": [0.7, 0.1, 0.1, 0.1], "cued": [0], "label": 0},
        {"kind": "dept", "probs": [0.2, 0.6, 0.1, 0.1], "cued": [0], "label": 0},
        # Two cues; the first one in the text is option 1. Argmax 0 equals the label.
        {"kind": "dept", "probs": [0.5, 0.3, 0.1, 0.1], "cued": [1, 0], "label": 0},
        # Argmax is option 2 (first of the tie), the label is 3.
        {"kind": "dept", "probs": [0.1, 0.1, 0.4, 0.4], "cued": [2, 3], "label": 3},
        {"kind": "escalate", "probs": [0.9, 0.1], "implied": 0, "label": 0},
        {"kind": "escalate", "probs": [0.8, 0.2], "implied": 0, "label": 1},  # flipped label
        {"kind": "escalate", "probs": [0.3, 0.7], "implied": 1, "label": 1},
    ]
    r = probe_report(rows)
    single, double, esc = r["dept_single"], r["dept_double"], r["escalate"]
    assert single["n"] == 2
    assert abs(single["acc"] - 0.5) < 1e-12
    assert abs(single["mean_p_cued"] - 0.45) < 1e-12
    assert abs(single["mean_max_p"] - 0.65) < 1e-12   # max of row one is 0.7, of row two 0.6
    assert double["n"] == 2
    assert abs(double["acc"] - 0.5) < 1e-12
    assert abs(double["mean_max_p"] - 0.45) < 1e-12
    assert abs(double["mean_mass_on_cued"] - 0.8) < 1e-12
    assert abs(double["mean_abs_dev_first_from_half"] - 0.15) < 1e-12
    assert esc["n"] == 3
    assert abs(esc["acc"] - 2 / 3) < 1e-12
    assert abs(esc["mean_conf"] - 0.8) < 1e-12
    assert abs(esc["mean_p_implied"] - 0.8) < 1e-12


def test_probe_report_empty_group_is_nan_not_an_error():
    r = probe_report([{"kind": "escalate", "probs": [0.9, 0.1], "implied": 0, "label": 0}])
    assert r["dept_single"]["n"] == 0
    assert math.isnan(r["dept_single"]["acc"])
    assert r["dept_double"]["n"] == 0


def test_build_probe_recovers_cues_and_drops_training_contexts():
    items, dropped = build_probe(400, 12345, train_contexts=set())
    assert dropped == 0
    assert len(items) == 800
    kinds = [meta["kind"] for _, meta in items]
    assert kinds.count("dept") == 400 and kinds.count("escalate") == 400
    orders = set()
    n_double = 0
    for q, meta in items:
        assert NOTA not in q.choices
        if meta["kind"] == "dept":
            assert sorted(q.choices) == sorted(DEPARTMENTS)
            orders.add(tuple(q.choices))
            assert len(meta["cued"]) in (1, 2)
            assert q.answer in meta["cued"]
            n_double += len(meta["cued"]) == 2
        else:
            assert q.primitive == "noul"
            assert meta["implied"] in (0, 1)
    assert len(orders) > 1  # department options are shuffled per question
    assert 0.2 < n_double / 400 < 0.4
    flipped = sum(q.answer != meta["implied"] for q, meta in items if meta["kind"] == "escalate")
    assert 0.04 < flipped / 400 < 0.17

    seen = {q.context for q in synthetic_triage(400, random.Random(12345))[:30]}  # the first 10 tickets
    kept, dropped = build_probe(400, 12345, train_contexts=seen)
    assert dropped == 20
    assert len(kept) == 780
    assert not any(q.context in seen for q, _ in kept)


# ---------- compare and curves on fake directories ----------

def _fake_preds(rng, n, sharpness, with_nota):
    preds = []
    for i in range(n):
        k = int(rng.integers(2, 6))
        true_logits = rng.normal(size=k)
        p = np.exp(true_logits) / np.exp(true_logits).sum()
        answer = int(rng.choice(k, p=p))
        nota_index = k - 1 if (with_nota and k > 2 and i % 2 == 0) else -1
        preds.append({"id": f"row-{i}", "source": "alpha" if i % 3 else "beta",
                      "primitive": "choice", "k": k, "answer": answer,
                      "logits": (true_logits * sharpness).tolist(), "nota_index": nota_index})
    return preds


def _fake_eval_dir(root, name, sharpness, meta, probe):
    run = root / name
    ev = run / "eval"
    ev.mkdir(parents=True)
    preds = _fake_preds(np.random.default_rng(0), 240, sharpness, with_nota=True)
    (ev / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds), encoding="utf-8")
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta))
    if probe is not None:
        (ev / "probe.json").write_text(to_json({"report": probe}))
    return ev


def _probe_fixture():
    return probe_report([
        {"kind": "dept", "probs": [0.7, 0.1, 0.1, 0.1], "cued": [0], "label": 0},
        {"kind": "dept", "probs": [0.5, 0.3, 0.1, 0.1], "cued": [1, 0], "label": 0},
        {"kind": "escalate", "probs": [0.9, 0.1], "implied": 0, "label": 0},
    ])


def _table_cells(path):
    lines = path.read_text().splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    cells = {}
    for line in lines[2:]:
        if not line.startswith("|"):
            break
        row = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(row) == len(header)
        cells[row[0]] = row
    return header, cells, lines


def test_compare_writes_table_plots_and_probe(tmp_path):
    probe = _probe_fixture()
    _fake_eval_dir(tmp_path, "warmup", 1.0, {"steps": 100, "init": "some/hub-id"}, None)
    a = _fake_eval_dir(tmp_path, "rlcd", 1.0, {"steps": 500, "stopped": "", "arm": "rlcd",
                                               "init": str(tmp_path / "warmup")}, probe)
    b = _fake_eval_dir(tmp_path, "rlvr", 4.0, {"steps": 150, "stopped": "eval_acc collapsed", "arm": "rlvr",
                                               "init": str(tmp_path / "warmup")}, probe)
    c = _fake_eval_dir(tmp_path, "zeroshot", 0.2, None, None)
    out = tmp_path / "compare"
    main(["compare", "--runs", f"rlcd={a}", f"rlvr={b}", f"zeroshot={c}", "--out", str(out)])

    header, cells, lines = _table_cells(out / "table.md")
    assert header == ["run", "training", "n", "acc [95% CI]", "Brier loss, lower is better [95% CI]",
                      "ECE [95% CI]", "ECE bias", "mean conf", "conf minus acc", "NOTA recall",
                      "NOTA false alarm"]
    assert set(cells) == {"rlcd", "rlvr", "zeroshot"}
    assert cells["rlcd"][1] == "100 SFT + 500 RL"
    assert cells["rlvr"][1] == "100 SFT + 150 RL (stopped)"
    assert cells["zeroshot"][1] == "n/a"
    assert cells["rlcd"][2] == "240"
    assert "[" in cells["rlcd"][3] and "]" in cells["rlcd"][3]
    # The 4x sharpened copy is overconfident, so its ECE must be the larger one.
    ece = lambda cell: float(cell.split()[0])  # noqa: E731
    assert ece(cells["rlvr"][5]) > ece(cells["rlcd"][5])
    # The honesty footnote sits under the table, after a blank line.
    assert lines[5] == "" and lines[6] == ECE_NOTE
    assert "biased upward" in ECE_NOTE and "paired" in ECE_NOTE

    for name in ("reliability.png", "coverage.png"):
        assert (out / name).stat().st_size > 1000
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics) == {"runs", "ece_note"}
    assert metrics["ece_note"] == ECE_NOTE
    assert set(metrics["runs"]) == {"rlcd", "rlvr", "zeroshot"}
    assert metrics["runs"]["rlvr"]["stopped"] == "eval_acc collapsed"
    assert metrics["runs"]["rlvr"]["training"] == "100 SFT + 150 RL (stopped)"
    m = metrics["runs"]["rlcd"]
    assert float(cells["rlcd"][6]) == round(m["ece_bias"], 3)
    assert len(m["reliability"]["bin_count"]) == 15 and sum(m["reliability"]["bin_count"]) == 240
    assert not (out / "paired.md").exists()
    by_source = (out / "by_source.md").read_text()
    assert "alpha" in by_source and "beta" in by_source and "rlvr" in by_source
    probe_md = (out / "probe.md").read_text()
    assert "ideal 0.5" in probe_md and "ideal 0.90" in probe_md
    assert "rlcd" in probe_md and "rlvr" in probe_md and "zeroshot" not in probe_md


def test_compare_pairs_writes_paired_differences(tmp_path):
    a = _fake_eval_dir(tmp_path, "rlcd", 1.0, None, None)
    b = _fake_eval_dir(tmp_path, "rlvr", 4.0, None, None)
    c = _fake_eval_dir(tmp_path, "twin", 1.0, None, None)
    out = tmp_path / "compare"
    main(["compare", "--runs", f"rlcd={a}", f"rlvr={b}", f"twin={c}", "--out", str(out),
          "--pairs", "rlcd:rlvr", "rlcd:twin"])
    paired = json.loads((out / "metrics.json").read_text())["paired"]
    assert set(paired) == {"rlcd:rlvr", "rlcd:twin"}
    for block in paired.values():
        assert set(block) >= {"acc", "ece", "brier"}
        for key in ("acc", "ece", "brier"):
            assert set(block[key]) == {"diff", "lo", "hi"}
    # Same argmax, sharper confidence: equal accuracy, and rlcd has the lower ECE and Brier loss.
    sharp = paired["rlcd:rlvr"]
    assert sharp["acc"] == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    assert sharp["ece"]["hi"] < 0.0 and sharp["brier"]["hi"] < 0.0
    assert all(paired["rlcd:twin"][key] == {"diff": 0.0, "lo": 0.0, "hi": 0.0} for key in ("acc", "ece", "brier"))
    text = (out / "paired.md").read_text()
    assert "rlcd minus rlvr" in text and "rlcd minus twin" in text
    assert "+0.000 [+0.000, +0.000]" in text


def test_compare_pairs_refuses_runs_evaluated_on_different_rows(tmp_path):
    import pytest
    a = _fake_eval_dir(tmp_path, "rlcd", 1.0, None, None)
    b = _fake_eval_dir(tmp_path, "rlvr", 4.0, None, None)
    preds = [json.loads(line) for line in (b / "preds.jsonl").read_text().splitlines()]
    preds[3]["id"] = "some-other-row"
    (b / "preds.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    out = tmp_path / "compare"
    with pytest.raises(SystemExit) as err:
        main(["compare", "--runs", f"rlcd={a}", f"rlvr={b}", "--out", str(out), "--pairs", "rlcd:rlvr"])
    assert "same rows" in str(err.value) and "rlcd" in str(err.value) and "rlvr" in str(err.value)
    assert not (out / "paired.md").exists()
    with pytest.raises(SystemExit) as err:
        main(["compare", "--runs", f"rlcd={a}", "--out", str(out), "--pairs", "rlcd:nobody"])
    assert "nobody" in str(err.value)


def test_curves_writes_a_png_from_train_logs(tmp_path):
    runs = []
    for name, slope in (("rlcd", 0.001), ("rlvr", -0.001)):
        run = tmp_path / name
        run.mkdir()
        rows = [{"arm": name, "total_steps": 100}]
        for step in range(0, 101, 50):
            rows.append({"step": step + 1, "loss": 0.1})
            rows.append({"eval_step": step, "eval_acc": 0.5 + slope * step, "eval_ece": 0.05 + abs(slope) * step})
        (run / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        runs.append(f"{name}={run}")
    out = tmp_path / "compare"
    main(["curves", "--runs", *runs, "--out", str(out)])
    assert (out / "curves.png").stat().st_size > 1000


def test_calibrate_cli_fits_per_primitive_and_applies_to_held_out_preds(tmp_path):
    from rlcd.calibrate import main as calibrate_main
    val = _fake_preds(np.random.default_rng(1), 600, 3.0, with_nota=False)
    for i, p in enumerate(val):
        p["primitive"] = "choice" if i % 2 else "score"
    test = _fake_preds(np.random.default_rng(2), 600, 3.0, with_nota=False)
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(p) + "\n" for p in val))
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(p) + "\n" for p in test))
    out = tmp_path / "sub" / "temperature.json"
    calibrate_main(["--preds", str(tmp_path / "val.jsonl"), "--out", str(out),
                    "--apply-to", str(tmp_path / "test.jsonl")])
    result = json.loads(out.read_text())
    assert {"overall", "choice", "score", "applied"} <= set(result)
    assert 2.0 < result["overall"] < 4.5
    applied = result["applied"]
    # Logits sharpened 3x are overconfident; the fitted temperature must shrink held-out ECE.
    assert applied["temperature_fitted"]["ece"] < applied["temperature_1"]["ece"]
    assert applied["temperature_fitted"]["acc"] == applied["temperature_1"]["acc"]


def test_curves_accepts_an_older_log_without_ece(tmp_path, capsys):
    new, old = tmp_path / "rlcd", tmp_path / "diag"
    new.mkdir()
    old.mkdir()
    (new / "train_log.jsonl").write_text("".join(
        json.dumps({"eval_step": s, "eval_acc": 0.5, "eval_ece": 0.05}) + "\n" for s in (0, 50)))
    (old / "train_log.jsonl").write_text("".join(
        json.dumps({"eval_step": s, "eval_acc": 0.4}) + "\n" for s in (100, 200)))
    out = tmp_path / "compare"
    main(["curves", "--runs", f"rlcd={new}", f"diag={old}", "--out", str(out), "--name", "c.png"])
    assert (out / "c.png").stat().st_size > 1000
    assert "diag" in capsys.readouterr().out  # the missing ECE curve is reported, not hidden


# ---------- pinned numbers, labels, styles, writers ----------

def test_summarize_pins_brier_ece_and_mean_conf_on_a_hand_computed_mixed_k_example():
    ln = math.log
    preds = [_pred([ln(3.0), 0.0], 0),                  # p = .75 .25, right, Brier .125, bin 11
             _pred([0.0, 0.0, ln(2.0)], 0),             # p = .25 .25 .5, wrong, Brier .875, bin 7
             _pred([ln(7.0), 0.0, 0.0, 0.0], 0),        # p = .7 .1 .1 .1, right, Brier .12, bin 10
             _pred([ln(0.72 / 0.28), 0.0], 1)]          # p = .72 .28, wrong, Brier 1.0368, bin 10
    o = summarize(preds)["overall"]
    assert abs(o["acc"] - 0.5) < 1e-12
    assert abs(o["mean_conf"] - (0.75 + 0.5 + 0.7 + 0.72) / 4) < 1e-12
    assert abs(o["brier"] - (0.125 + 0.875 + 0.12 + 1.0368) / 4) < 1e-12
    # Bins 11 and 7 hold one row each; bin 10 holds two with mean conf .71 and accuracy .5.
    assert abs(o["ece"] - (0.25 + 0.5 + 2 * 0.21) / 4) < 1e-12
    assert o["reliability"]["bin_count"][7] == 1 and o["reliability"]["bin_count"][10] == 2
    assert abs(o["reliability"]["bin_conf"][10] - 0.71) < 1e-12
    assert o["reliability"]["bin_acc"][10] == 0.5


def test_summarize_ece_bias_is_bootstrap_mean_minus_point_estimate():
    from rlcd.metrics import bootstrap_stats, ece
    rng = np.random.default_rng(3)
    preds = _fake_preds(rng, 400, 1.0, with_nota=False)   # calibrated by construction
    o = summarize(preds)["overall"]
    probs = [np.exp(np.array(p["logits"])) / np.exp(np.array(p["logits"])).sum() for p in preds]
    conf = np.array([p.max() for p in probs])
    correct = np.array([float(p.argmax() == q["answer"]) for p, q in zip(probs, preds)])
    boot = bootstrap_stats(lambda f, c: ece(f, c, 15), (conf, correct))
    assert abs(o["ece_bias"] - (boot.mean() - o["ece"])) < 1e-9
    assert o["ece_bias"] > 0.0   # plug-in ECE of a calibrated model is biased upward under resampling


def test_training_label_cases(tmp_path):
    def run(name, meta):
        d = tmp_path / name
        d.mkdir()
        if meta is not None:
            (d / "meta.json").write_text(json.dumps(meta))
        return d

    warm = run("warm", {"steps": 100, "init": "Qwen/Qwen3-0.6B-Base"})
    assert training_label(warm) == "100 SFT"
    assert training_label(run("sft", {"steps": 500, "init": "org/model"})) == "500 SFT"
    rl = run("rl", {"steps": 500, "arm": "rlcd", "stopped": "", "init": str(warm)})
    assert training_label(rl) == "100 SFT + 500 RL"
    stopped = run("stopped", {"steps": 150, "arm": "rlvr", "stopped": "eval_acc fell", "init": str(warm)})
    assert training_label(stopped) == "100 SFT + 150 RL (stopped)"
    # An init recorded relative to another working directory resolves to the sibling run directory.
    moved = run("moved", {"steps": 500, "arm": "oracle", "stopped": "", "init": "runs/warm"})
    assert training_label(moved) == "100 SFT + 500 RL"
    # An RL run whose init is a hub id, or is gone, still says what it ran itself.
    assert training_label(run("hub-rl", {"steps": 40, "arm": "rlcd", "init": "org/model"})) == "40 RL"
    assert training_label(run("partial", {"steps": 300, "arm": "rlcd", "init": str(warm),
                                          "in_progress": True})) == "100 SFT + 300 RL (unfinished)"
    assert training_label(run("no-meta", None)) == "n/a"
    assert training_label(run("no-steps", {"init": "x"})) == "n/a"
    assert training_label(tmp_path / "does-not-exist") == "n/a"


def test_fallback_palette_is_disjoint_from_registered_run_colors():
    registered = {color.lower() for color, _, _ in RUN_STYLE.values()}
    assert len(FALLBACK_PALETTE) >= 6
    assert registered.isdisjoint(c.lower() for c in FALLBACK_PALETTE)
    assert len({c.lower() for c in FALLBACK_PALETTE}) == len(FALLBACK_PALETTE)
    names = ("q-warmup", "q-rlcd", "q-rlvr", "q-rlvr-lowlr", "q-oracle", "bake-qwen06")
    for name in names:
        assert name in RUN_STYLE
    assert len({RUN_STYLE[n][0] for n in names}) == 6
    unknown = run_style("some-run-nobody-registered")
    assert unknown["color"] in FALLBACK_PALETTE and unknown["linestyle"] == "--"
    assert run_style("q-rlcd")["color"] == RUN_STYLE["q-rlcd"][0]


def test_reliability_points_skip_sparse_bins():
    assert MIN_BIN_ROWS == 30
    conf = np.concatenate([np.full(40, 0.95), np.full(29, 0.55), np.full(30, 0.35)])
    correct = np.concatenate([np.ones(40), np.zeros(29), np.ones(30)])
    x, y = reliability_points(conf, correct)
    assert np.allclose(x, [0.35, 0.95]) and np.allclose(y, [1.0, 1.0])   # the 29-row bin is not drawn


def test_curve_points_offset_and_stopped_step():
    rows = [{"eval_step": 0, "eval_acc": 0.7, "eval_ece": 0.05}, {"step": 10, "loss": 0.1},
            {"eval_step": 50, "eval_acc": 0.6}, {"stopped": "eval_acc fell", "at_step": 50}]
    assert curve_points(rows, "eval_acc", 100) == ([100, 150], [0.7, 0.6])
    assert curve_points(rows, "eval_ece", 0) == ([0], [0.05])
    assert stopped_step(rows) == 50
    assert stopped_step(rows[:3]) is None


def test_curves_offset_flag_and_stopped_marker(tmp_path, capsys):
    import pytest
    runs = []
    for name, stopped in (("q-rlcd", False), ("q-rlvr", True)):
        run = tmp_path / name
        run.mkdir()
        rows = [{"eval_step": s, "eval_acc": 0.7 - 0.001 * s * stopped, "eval_ece": 0.05} for s in (0, 50, 100)]
        if stopped:
            rows.append({"stopped": "eval_acc fell", "at_step": 100})
        (run / "train_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        runs.append(f"{name}={run}")
    out = tmp_path / "compare"
    main(["curves", "--runs", *runs, "--out", str(out), "--offset", "q-rlcd=100", "q-rlvr=100"])
    assert (out / "curves.png").stat().st_size > 1000
    text = capsys.readouterr().out
    assert "q-rlvr" in text and "stopped" in text
    with pytest.raises(SystemExit) as err:
        main(["curves", "--runs", *runs, "--out", str(out), "--offset", "nobody=100"])
    assert "nobody" in str(err.value)


def test_probe_tables_show_single_cue_max_probability_and_tolerate_old_reports():
    report = _probe_fixture()
    text = probe_tables({"new": report})
    assert "mean max probability (ideal 1.0)" in text and "mean max probability (ideal 0.5)" in text
    assert "| new | 1 | 1.000 | 0.700 | 0.700 |" in text
    old = json.loads(to_json(report))
    del old["dept_single"]["mean_max_p"]
    assert "| old | 1 | 1.000 | 0.700 | n/a |" in probe_tables({"old": old})


def test_write_rows_is_strict_json_with_nan_as_null(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_rows(path, [{"id": "a", "logits": [1.0, float("nan")], "probs": np.array([0.5, 0.5])},
                       {"id": "b", "logits": [float("inf")]}])

    def reject(token):
        raise AssertionError(f"non-standard JSON constant {token}")

    rows = [json.loads(line, parse_constant=reject) for line in path.read_text().splitlines()]
    assert rows == [{"id": "a", "logits": [1.0, None], "probs": [0.5, 0.5]}, {"id": "b", "logits": [None]}]


def test_overall_text_names_brier_as_a_loss():
    text = overall_text(summarize([_pred([2.0, 0.0], 0), _pred([0.0, 2.0], 0)])["overall"])
    assert "Brier loss (lower is better)" in text
    assert "acc" in text and "ECE" in text and "n/a" in text   # NOTA recall is undefined here


def test_run_command_end_to_end_on_a_tiny_model(tmp_path, monkeypatch, capsys):
    import rlcd.policies
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=128,
                     num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    policy = EvePolicy(EveMoEForCausalLM(cfg), FakeTok(), list(range(100, 126)))
    monkeypatch.setattr(rlcd.policies, "load_policy", lambda *a, **kw: policy)
    qs = [Question("choice", "ctx", "pick", ["x", "y", NOTA], answer=2, source="alpha", id="a-1"),
          Question("noul", "ctx", "is it", ["true", "false"], answer=1, source="beta", id="b-1")]
    split = tmp_path / "split.jsonl"
    write_questions(str(split), qs)
    out = tmp_path / "eval"
    main(["run", "--model", "unused", "--out", str(out), "--split", str(split), "--device", "cpu",
          "--max-len", "64"])
    assert "Brier loss (lower is better)" in capsys.readouterr().out
    assert [json.loads(line)["id"] for line in (out / "preds.jsonl").read_text().splitlines()] == ["a-1", "b-1"]
    assert json.loads((out / "metrics.json").read_text())["overall"]["n"] == 2


def test_calibrate_applies_the_val_fitted_temperature_and_prints_brier_as_a_loss(tmp_path, capsys):
    from rlcd.calibrate import main as calibrate_main
    val = _fake_preds(np.random.default_rng(1), 600, 3.0, with_nota=False)
    test = _fake_preds(np.random.default_rng(2), 600, 1.5, with_nota=False)
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(p) + "\n" for p in val))
    (tmp_path / "test.jsonl").write_text("".join(json.dumps(p) + "\n" for p in test))
    out = tmp_path / "temperature.json"
    calibrate_main(["--preds", str(tmp_path / "val.jsonl"), "--out", str(out),
                    "--apply-to", str(tmp_path / "test.jsonl")])
    result = json.loads(out.read_text())
    fitted = fit_temperature(val)
    assert result["overall"] == fitted
    assert abs(fitted - fit_temperature(test)) > 0.5   # a test-fitted temperature would be a different number
    assert result["applied"]["temperature_fitted"] == json.loads(to_json(summarize(test, temperature=fitted)["overall"]))
    assert result["applied"]["temperature_1"] == json.loads(to_json(summarize(test, temperature=1.0)["overall"]))
    assert "Brier loss (lower is better)" in capsys.readouterr().out
