"""One IT ticket, three typed questions, one prefill, typed answers as JSON.

    uv run python scripts/demo_decide.py --model runs/q-rlcd/decision
"""
from __future__ import annotations

import argparse
import json
import time

from rlcd.decide import ChoiceQ, Decider, NoulQ, ScoreQ

STATE = ("Ticket #4242 from an enterprise, tier 1 customer. Report: API latency spiked to 4 seconds; "
         "load balancer returning 502s. Note: all customers affected.")
QUESTIONS = [
    ChoiceQ("Which department should handle this ticket?",
            ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]),
    ScoreQ("What is the priority of this ticket?", ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]),
    NoulQ("Should an on-call engineer be paged immediately?"),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/q-rlcd/decision")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    decider = Decider.load(args.model, args.device)
    decider.ask(STATE, QUESTIONS)  # warm up the kernels so the timing below is the steady state
    t0 = time.perf_counter()
    answers = decider.ask(STATE, QUESTIONS)
    ms = (time.perf_counter() - t0) * 1000
    print(json.dumps({"state": STATE, "questions": [q.__dict__ | {"kind": type(q).__name__} for q in QUESTIONS],
                      "answers": answers, "ms": round(ms, 1)}, indent=2))


if __name__ == "__main__":
    main()
