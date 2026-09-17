# eve-rlcd: a toy recreation of RLCD on Eve-2

Date: 2026-09-17
Status: approved design, not yet implemented

## Goal

Build a small open-source model in the spirit of TypeSafe's Jev: a model that returns
typed decisions with probabilities and cannot emit text. Train it with a faithful
recreation of Reinforcement Learning for Calibrated Decisions (RLCD), and show with an
ablation why a calibration reward behaves differently from a verifiable-outcome reward.

Base model: `anthonym21/Eve-2-MoE-IT-272M`. Hardware: one RTX 4080 (16 GB) on Windows.

## Background

Two sources motivated this project.

The Substack post "Jev: The Language Model That Won't Talk" (2026-09-16) describes
TypeSafe's product: three primitives (Choice, Score, Noul), typed distributions as the
native output, questions evaluated in parallel against a shared state, and a training
objective called RLCD that is supposed to make stated probabilities match how often the
decision turns out right. TypeSafe has published no reward function, architecture, or
training procedure.

The Hugging Face repo `harshatheg/Qwen-2.5-1B-RLCD` is, despite its name, an
inference-only engine: it prefills context and schema once, broadcasts the KV cache
across schema fields, and takes a softmax over the candidate tokens at each field's
decision position. The model is stock Qwen2.5-1.5B-Instruct. Nothing is trained and
nothing is calibrated. It supplies the inference pattern; this project supplies the
training.

## What makes this RL rather than supervised learning

If the trainer sees the correct label for every question, a proper scoring rule such as
log score or Brier score is a differentiable loss, and "RLCD" collapses into supervised
training. That is not the setting a deployed decision model lives in. When a support
system routes a ticket to Billing, it learns whether Billing was right. It never learns
what the other 25 departments would have done.

This project therefore trains under bandit feedback. The policy samples one decision per
question, the environment reveals only whether that decision was correct, and the full
label stays hidden from the learner. Supervised training is impossible here and
reinforcement learning is required.

## Policy and output

A question is rendered in Eve-2-IT's own prompt style:

```
User: Context:
<shared state>

Question: <question text>
Options:
A) <choice 1>
B) <choice 2>
...
Answer with the letter only.
Assistant: The answer is
```

One forward pass. The logits at the final position are sliced to the letter tokens
(" A" through " Z" are single GPT-2 tokens) and softmaxed. That K-way distribution is
both the policy and the API output.

The three primitives map onto it as follows.

- Choice: K declared options. Returns the distribution, the argmax, and a confidence
  value. Two confidence statistics are reported, max probability and one minus normalized
  entropy, because TypeSafe has not said which it uses.
- Score: the same rendering over ordered levels. Returns the distribution plus a
  continuous value, the expected level index scaled to the range 0 to 1.
- Noul: Choice over `true` and `false`. Returns P(true).

Cardinality is capped at 26 in v1. Every question may declare a "None of the above"
option. Training data injects it deliberately: in a fraction of examples the true label
is removed from the choices and "None of the above" becomes the correct answer, so the
model learns an abstain route instead of forcing probability onto a wrong option.

Several questions about one context run as a single right-padded batch. Eve-2 ignores
attention masks and uses pure causal attention, so right padding leaves every real
position's logits untouched. Logits are gathered at each row's last real token.

## The RLCD reward

For a prompt x the policy p(. | x) samples an action a with probability p_a. The
environment returns c = 1 if a was correct and c = 0 otherwise. The reward is

```
r = c - p_a
```

Stated 90 percent and was right: reward 0.1. Stated 90 percent and was wrong: reward
-0.9. Stated 30 percent and was right: reward 0.7. The reward measures how far the stated
probability sat from the outcome, which is the definition of calibration the article
gives.

This reward is not a heuristic. Let y be the hidden correct action and let the Brier
score of the full distribution be J(p) = -sum_a (p_a - 1[a = y])^2. Its gradient is
2 grad p_y - 2 sum_a p_a grad p_a. A REINFORCE update with reward r(a) = 2(1[a = y] - p_a)
has expected gradient sum_a p_a r(a) grad log p_a = 2 grad p_y - 2 sum_a p_a grad p_a,
which is exactly grad J. So REINFORCE with r = c - p_a (the constant 2 folds into the
learning rate) is an unbiased estimator of the Brier score gradient using only the
outcome of the taken action. Its fixed point is a calibrated policy. A unit test checks
this identity numerically on a toy softmax.

The update uses G sampled actions per prompt with a group-mean baseline (GRPO style) for
variance reduction, and a KL penalty toward the warmup model so the policy does not
collapse or drift off the letter format. No token generation is involved: the action is
a single sampled letter from one forward pass, so each RL step is one forward and one
backward.

## The ablation

Same model, same data, same RL loop, three training signals, matching the three
objectives the article separates.

| Arm | Signal | Feedback | Expected behavior |
|-----|--------|----------|-------------------|
| RLVR | r = c | bandit | accuracy rises, reliability curve bends into overconfidence |
| RLCD | r = c - p_a | bandit | similar accuracy, reliability curve stays near the diagonal |
| Supervised oracle | log score on the full label | full label | the ceiling RLCD with partial feedback should approach |

All three are compared on the same held-out test set with reliability diagrams,
expected calibration error, Brier score, accuracy, and coverage-versus-error curves at
confidence thresholds. The last of these answers the article's operational question:
what error rate remains at a given level of autonomous coverage.

## Pipeline

1. Warmup SFT. About 2,000 labeled examples for one epoch so the letter format works at
   all. This mirrors the InstructGPT pattern of SFT before RL and is itself ablated.
2. RL loop. Plain PyTorch. The policy is a single categorical, so TRL and other
   sequence-oriented RL libraries do not fit. Full fine-tune in bf16, no LoRA, following
   the Eve-2 recipe. Eve's router auxiliary loss is added to every update.
3. Post-hoc temperature scaling on validation, reported for reference. The claim under
   test is that the RLCD reward produces calibration on its own, so temperature is not
   applied to the headline numbers.
4. Amputation. Replace the 50,304-row LM head with the 26 selected rows and save a
   decision model that cannot produce text. The input embedding stays full-size.

## Data

Public labeled sets converted into one JSONL format with fields for primitive, context,
question, choices, ordered flag, and answer index. Capped at about 8k train, 1k
validation, and 1k test per source.

- Choice: Bitext customer support intents, Banking77 (each example gets a sampled
  26-choice subset that always includes the truth), AG News, MultiNLI with the premise as
  context and the hypothesis as the question.
- Score: SST-5, Yelp five-star reviews.
- Noul: BoolQ.
- Synthetic multi-field triage records (priority, department, escalate) generated from
  templates with injected ambiguity, so some examples are genuinely uncertain and a
  calibrated model should say so.
- None-of-the-above injection across all sources.

The RL environment is a simulator over these labels. It receives a sampled action and
returns only whether that action was correct.

## Repository layout

```
eve-rlcd/
  pyproject.toml
  README.md
  rlcd/
    schema.py       question types, validation, prompt rendering, letter token ids
    data.py         dataset converters, NOTA injection, synthetic triage generator
    policy.py       model adapter: forward, right-padded gather, sliced softmax
    env.py          bandit environment over labeled data
    rewards.py      rlvr, rlcd, and supervised objectives
    train_sft.py    warmup
    train_rl.py     RL loop with group sampling and KL penalty
    calibrate.py    temperature fitting
    eval.py         metrics, reliability diagrams, coverage-error curves
    amputate.py     build and save the decision-only model
    infer.py        Choice / Score / Noul API over a saved model
  tests/
  data/             generated, gitignored
  runs/             checkpoints and plots, gitignored
  docs/plans/
```

## Environment

Python 3.12 managed by uv, CUDA build of PyTorch, transformers, datasets, numpy,
matplotlib, pytest. Eve-2 loads through `trust_remote_code=True`.

## Testing

- Reward math: numerical check that the RLCD REINFORCE estimator matches the direct
  Brier gradient on a toy softmax; check that the RLVR estimator matches the accuracy
  gradient.
- Prompt rendering and letter token ids for every cardinality 2 through 26.
- Right-padded batch gather returns the same logits as unpadded single-row forwards.
- Metrics (ECE, Brier, coverage-error) against hand-computed values.
- NOTA injection produces the intended mix and never leaves the truth absent without
  NOTA present.
- Schema validation rejects more than 26 choices, duplicate choices, and Score questions
  without ordered levels.
- A 200-example smoke run of SFT, RL, and eval before any full run.

## Out of scope for v1

Cardinality above 26, KV-cache broadcasting (Eve-2 has no cache), verbalized confidence
tokens, an RLHF-style preference arm, and any comparison model other than the optional
Qwen2.5-0.5B-Instruct zero-shot reference.

## References

- Anthony Maio, "Jev: The Language Model That Won't Talk", Substack, 2026-09-16.
- harshatheg/Qwen-2.5-1B-RLCD on Hugging Face (parallel constrained decoding engine).
- anthonym21/Eve-2-MoE-IT-272M on Hugging Face.
- Stangel et al., "Rewarding Doubt", 2025 (RL with a proper scoring rule reward on
  verbalized confidence).
- Guo et al., "On Calibration of Modern Neural Networks", 2017 (temperature scaling).
