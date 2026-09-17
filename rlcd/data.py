"""Dataset converters, NOTA injection, synthetic triage, and the build CLI."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rlcd.schema import MAX_CHOICES, NOTA, Question, write_jsonl


# ---------- helpers ----------

def subset_choices(all_choices: list[str], truth: str, n: int, rng: random.Random) -> tuple[list[str], int]:
    """n-1 random distractors plus the truth at a random position."""
    distractors = [c for c in all_choices if c != truth]
    rng.shuffle(distractors)
    chosen = distractors[: max(n - 1, 0)]
    pos = rng.randrange(len(chosen) + 1)
    chosen.insert(pos, truth)
    return chosen, pos


def inject_nota(q: Question, rng: random.Random, p_remove: float = 0.15, p_add: float = 0.35) -> Question:
    """With p_remove: drop the true choice and make NOTA correct.
    With p_add: append NOTA as a distractor. Otherwise unchanged. Never for noul."""
    if q.primitive == "noul" or NOTA in q.choices:
        return q
    r = rng.random()
    if r < p_remove:
        choices = [c for i, c in enumerate(q.choices) if i != q.answer]
        if len(choices) >= MAX_CHOICES:
            choices = choices[: MAX_CHOICES - 1]
        choices.append(NOTA)
        return Question(q.primitive, q.context, q.question, choices, q.ordered,
                        len(choices) - 1, q.source, q.id)
    if r < p_remove + p_add:
        choices = list(q.choices)
        answer = q.answer
        if len(choices) >= MAX_CHOICES:
            drop = rng.choice([i for i in range(len(choices)) if i != answer])
            choices.pop(drop)
            if drop < answer:
                answer -= 1
        choices.append(NOTA)
        return Question(q.primitive, q.context, q.question, choices, q.ordered, answer, q.source, q.id)
    return q


def clean_labels(names: list[str], source: str) -> list[str]:
    """Make dataset label names schema-safe: stripped and single-line. Duplicates after
    cleaning would make two options indistinguishable, so they are an error."""
    out = [" ".join(str(n).splitlines()).strip() for n in names]
    if any(not n for n in out):
        raise ValueError(f"{source}: empty label name in {names!r}")
    dupes = sorted({n for n in out if out.count(n) > 1})
    if dupes:
        raise ValueError(f"{source}: duplicate label names after cleaning: {dupes!r}")
    return out


def drop_empty_contexts(qs: list[Question]) -> tuple[list[Question], int]:
    """Rows with an empty or whitespace-only context carry nothing to decide on."""
    kept = [q for q in qs if q.context.strip()]
    return kept, len(qs) - len(kept)


def split_source(qs: list[Question], per_source: int, n_val: int, n_test: int, rng: random.Random):
    qs = list(qs)
    rng.shuffle(qs)
    test = qs[:n_test]
    val = qs[n_test:n_test + n_val]
    train = qs[n_test + n_val:n_test + n_val + per_source]
    return train, val, test


# ---------- synthetic triage ----------

DEPARTMENTS = ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]
PRIORITIES = ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]  # ascending severity

DEPT_CUES = {
    "BILLING": ["invoice total is wrong", "charged twice this month", "refund has not arrived",
                "credit card declined at renewal", "tax line looks incorrect"],
    "INFRASTRUCTURE": ["database CPU pinned at 100 percent", "API latency spiked to 4 seconds",
                       "pods crash looping in prod", "disk on the primary node is full",
                       "load balancer returning 502s"],
    "SECURITY": ["login from an unknown country", "suspected leaked API key",
                 "phishing email hit the whole team", "MFA bypass reported", "unexpected admin role grant"],
    "PRODUCT_SUPPORT": ["export button does nothing", "cannot find the settings page",
                        "how do I invite a teammate", "dark mode resets on reload", "search ignores filters"],
}
PRIO_CUES = {
    "P3_LOW": ["no rush", "cosmetic", "whenever you get a chance"],
    "P2_NORMAL": ["please look into this", "affecting one user", "not blocking"],
    "P1_HIGH": ["blocking our team", "several customers affected", "need this today"],
    "P0_CRITICAL": ["production is down", "all customers affected", "revenue impact right now"],
}
CUSTOMER_TIERS = ["free tier", "startup plan", "enterprise, tier 1", "enterprise, tier 2"]


def synthetic_triage(n_records: int, rng: random.Random) -> list[Question]:
    out: list[Question] = []
    for i in range(n_records):
        dept = rng.choice(DEPARTMENTS)
        prio_idx = rng.randrange(4)
        cues = [rng.choice(DEPT_CUES[dept])]
        label_dept = dept
        if rng.random() < 0.3:  # genuinely ambiguous record
            other = rng.choice([d for d in DEPARTMENTS if d != dept])
            cues.append(rng.choice(DEPT_CUES[other]))
            if rng.random() < 0.4:
                label_dept = other
        rng.shuffle(cues)
        text = (f"Ticket #{1000 + i} from a {rng.choice(CUSTOMER_TIERS)} customer. "
                f"Report: {'; '.join(cues)}. Note: {rng.choice(PRIO_CUES[PRIORITIES[prio_idx]])}.")
        escalate = prio_idx >= 2
        if rng.random() < 0.1:
            escalate = not escalate
        rid = f"triage-{i}"
        out.append(Question("choice", text, "Which department should handle this ticket?",
                            list(DEPARTMENTS), False, DEPARTMENTS.index(label_dept), "triage", rid + "-dept"))
        out.append(Question("score", text, "What is the priority of this ticket?",
                            list(PRIORITIES), True, prio_idx, "triage", rid + "-prio"))
        out.append(Question("noul", text, "Should an on-call engineer be paged immediately?",
                            ["true", "false"], False, 0 if escalate else 1, "triage", rid + "-esc"))
    return out


# ---------- public dataset converters (network) ----------

def _ds(name: str, split: str, **kw):
    from datasets import load_dataset
    return load_dataset(name, split=split, **kw)


def load_bitext(rng: random.Random) -> list[Question]:
    ds = _ds("bitext/Bitext-customer-support-llm-chatbot-training-dataset", "train")
    raw = sorted(set(ds["intent"]))
    clean = dict(zip(raw, clean_labels(raw, "bitext")))
    intents = list(clean.values())
    out = []
    for i, row in enumerate(ds):
        choices, ans = subset_choices(intents, clean[row["intent"]], 25, rng)
        out.append(Question("choice", row["instruction"], "What is the customer's intent?",
                            choices, False, ans, "bitext", f"bitext-{i}"))
    return out


def load_banking77(rng: random.Random) -> list[Question]:
    # PolyAI/banking77 ships only a loading script, which datasets >= 4 refuses to run.
    # legacy-datasets/banking77 is the hub's parquet copy of the same dataset.
    ds = _ds("legacy-datasets/banking77", "train+test")
    names = clean_labels(ds.features["label"].names, "banking77")
    out = []
    for i, row in enumerate(ds):
        truth = names[row["label"]]
        choices, ans = subset_choices(names, truth, 25, rng)
        out.append(Question("choice", row["text"], "Which banking intent does the message express?",
                            choices, False, ans, "banking77", f"banking77-{i}"))
    return out


def load_ag_news(rng: random.Random) -> list[Question]:
    ds = _ds("fancyzhx/ag_news", "train[:12000]+test[:3000]")
    names = ["World", "Sports", "Business", "Sci/Tech"]
    return [Question("choice", row["text"], "What is the topic of this article?",
                     list(names), False, int(row["label"]), "ag_news", f"ag_news-{i}")
            for i, row in enumerate(ds)]


def load_mnli(rng: random.Random) -> list[Question]:
    ds = _ds("nyu-mll/multi_nli", "train[:12000]+validation_matched[:3000]")
    names = ["entailment", "neutral", "contradiction"]
    out = []
    for i, row in enumerate(ds):
        if row["label"] < 0:
            continue
        out.append(Question("choice", row["premise"],
                            f'What is the relation of the context to the hypothesis: "{row["hypothesis"]}"?',
                            list(names), False, int(row["label"]), "mnli", f"mnli-{i}"))
    return out


def load_sst5(rng: random.Random) -> list[Question]:
    ds = _ds("SetFit/sst5", "train+validation+test")
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    return [Question("score", row["text"], "What is the sentiment of the text?",
                     list(levels), True, int(row["label"]), "sst5", f"sst5-{i}")
            for i, row in enumerate(ds)]


def load_yelp(rng: random.Random) -> list[Question]:
    ds = _ds("Yelp/yelp_review_full", "train[:12000]+test[:3000]")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    return [Question("score", row["text"][:1500], "How many stars did the reviewer give?",
                     list(levels), True, int(row["label"]), "yelp", f"yelp-{i}")
            for i, row in enumerate(ds)]


def load_boolq(rng: random.Random) -> list[Question]:
    ds = _ds("google/boolq", "train+validation")
    out = []
    for i, row in enumerate(ds):
        proposition = row["question"].strip().rstrip("?") + "?"
        out.append(Question("noul", row["passage"][:1500], f"Is the answer to this question yes: {proposition}",
                            ["true", "false"], False, 0 if row["answer"] else 1, "boolq", f"boolq-{i}"))
    return out


def load_triage(rng: random.Random) -> list[Question]:
    return synthetic_triage(4000, rng)


SOURCES = {
    "bitext": load_bitext,
    "banking77": load_banking77,
    "ag_news": load_ag_news,
    "mnli": load_mnli,
    "sst5": load_sst5,
    "yelp": load_yelp,
    "boolq": load_boolq,
    "triage": load_triage,
}


# ---------- build ----------

def build(out_dir: str, per_source: int = 8000, n_val: int = 1000, n_test: int = 1000,
          seed: int = 0, sources: list[str] | None = None) -> dict:
    rng = random.Random(seed)
    train, val, test = [], [], []
    stats = {}
    skipped = {}
    for name in sources or list(SOURCES):
        qs, skipped[name] = drop_empty_contexts(SOURCES[name](rng))
        tr, va, te = split_source(qs, per_source, n_val, n_test, rng)
        tr = [inject_nota(q, rng) for q in tr]
        va = [inject_nota(q, rng) for q in va]
        te = [inject_nota(q, rng) for q in te]
        train += tr
        val += va
        test += te
        stats[name] = {"train": len(tr), "val": len(va), "test": len(te)}
    rng.shuffle(train)
    out = Path(out_dir)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "test.jsonl", test)
    stats["total"] = {"train": len(train), "val": len(val), "test": len(test)}
    stats["skipped"] = skipped
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data")
    b.add_argument("--per-source", type=int, default=8000)
    b.add_argument("--n-val", type=int, default=1000)
    b.add_argument("--n-test", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sources", nargs="*", default=None)
    args = ap.parse_args()
    print(json.dumps(build(args.out, args.per_source, args.n_val, args.n_test, args.seed, args.sources), indent=2))


if __name__ == "__main__":
    main()
