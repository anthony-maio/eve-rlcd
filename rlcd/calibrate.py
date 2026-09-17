"""Post-hoc temperature scaling on saved validation logits. Reference only: headline metrics
stay at temperature 1."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rlcd.eval import load_preds, summarize, to_json
from rlcd.schema import NEG


def fit_temperature(preds: list[dict]) -> float:
    """The temperature that minimizes the NLL of the answers, found over log T with LBFGS.
    Rows have variable k; padding holds NEG so it carries no mass at any temperature."""
    kmax = max(p["k"] for p in preds)
    logits = torch.full((len(preds), kmax), NEG, dtype=torch.float64)
    for i, p in enumerate(preds):
        logits[i, : p["k"]] = torch.tensor(p["logits"], dtype=torch.float64)
    answers = torch.tensor([p["answer"] for p in preds])
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), answers)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="validation preds.jsonl to fit on")
    ap.add_argument("--out", required=True, help="where to write the fitted temperatures as JSON")
    ap.add_argument("--apply-to", default=None,
                    help="optional held-out preds.jsonl; its overall metrics are reported at "
                         "temperature 1 and at the fitted overall temperature")
    args = ap.parse_args(argv)
    preds = load_preds(args.preds)
    result = {"fit_on": str(args.preds), "n": len(preds), "overall": fit_temperature(preds)}
    for prim in sorted({p["primitive"] for p in preds}):
        result[prim] = fit_temperature([p for p in preds if p["primitive"] == prim])
    if args.apply_to:
        held_out = load_preds(args.apply_to)
        result["applied"] = {"preds": str(args.apply_to),
                             "temperature_1": summarize(held_out, 1.0)["overall"],
                             "temperature_fitted": summarize(held_out, result["overall"])["overall"]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(to_json(result) + "\n")
    print(to_json(result))


if __name__ == "__main__":
    main()
