import json
import math
import random

import numpy as np
import torch

from rlcd.calibrate import fit_temperature
from rlcd.compat import eve_config
from rlcd.data import DEPARTMENTS, synthetic_triage
from rlcd.eval import build_probe, main, predict, probe_report, summarize, to_json
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policies import EvePolicy
from rlcd.schema import NOTA, Question

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
    assert set(s["overall"]) == GROUP_KEYS | {"ci"}
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


def test_compare_writes_table_plots_and_probe(tmp_path):
    probe = probe_report([
        {"kind": "dept", "probs": [0.7, 0.1, 0.1, 0.1], "cued": [0], "label": 0},
        {"kind": "dept", "probs": [0.5, 0.3, 0.1, 0.1], "cued": [1, 0], "label": 0},
        {"kind": "escalate", "probs": [0.9, 0.1], "implied": 0, "label": 0},
    ])
    a = _fake_eval_dir(tmp_path, "rlcd", 1.0, {"steps": 500, "stopped": ""}, probe)
    b = _fake_eval_dir(tmp_path, "rlvr", 4.0, {"steps": 150, "stopped": "eval_acc collapsed"}, probe)
    c = _fake_eval_dir(tmp_path, "zeroshot", 0.2, None, None)
    out = tmp_path / "compare"
    main(["compare", "--runs", f"rlcd={a}", f"rlvr={b}", f"zeroshot={c}", "--out", str(out)])

    lines = (out / "table.md").read_text().splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    assert header == ["run", "steps", "n", "acc [95% CI]", "Brier loss, lower is better [95% CI]",
                      "ECE [95% CI]", "mean conf", "conf minus acc", "NOTA recall", "NOTA false alarm"]
    assert len(lines) == 2 + 3
    cells = {}
    for line in lines[2:]:
        row = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(row) == len(header)
        cells[row[0]] = row
    assert cells["rlcd"][1] == "500"
    assert cells["rlvr"][1] == "150 (stopped)"
    assert cells["zeroshot"][1] == "n/a"
    assert cells["rlcd"][2] == "240"
    assert "[" in cells["rlcd"][3] and "]" in cells["rlcd"][3]
    # The 4x sharpened copy is overconfident, so its ECE must be the larger one.
    ece = lambda cell: float(cell.split()[0])  # noqa: E731
    assert ece(cells["rlvr"][5]) > ece(cells["rlcd"][5])

    for name in ("reliability.png", "coverage.png"):
        assert (out / name).stat().st_size > 1000
    metrics = json.loads((out / "metrics.json").read_text())
    assert set(metrics) == {"rlcd", "rlvr", "zeroshot"}
    assert metrics["rlvr"]["stopped"] == "eval_acc collapsed"
    by_source = (out / "by_source.md").read_text()
    assert "alpha" in by_source and "beta" in by_source and "rlvr" in by_source
    probe_md = (out / "probe.md").read_text()
    assert "ideal 0.5" in probe_md and "ideal 0.90" in probe_md
    assert "rlcd" in probe_md and "rlvr" in probe_md and "zeroshot" not in probe_md


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
