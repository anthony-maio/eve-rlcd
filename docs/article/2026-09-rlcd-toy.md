# I built a small model that won't talk, to see if the objective holds up

Two days ago I ended a post about TypeSafe's Jev on a shrug: whether reinforcement learning for calibrated decisions delivers is unproven, because TypeSafe has published no reward, no architecture, no training procedure. That is still true. So I did the thing you can do when a company describes an objective and won't show it: take the description literally and build the smallest version that could be wrong.

This is not a reproduction of Jev. I have no idea what Jev is inside, and nothing below is a claim about it. It is a 0.6B-parameter model that answers typed questions with probabilities instead of text, trained with the only reward I could think of that actually means "your probabilities should match how often you turn out to be right." Code, data, and every number are in a repo linked at the end.

## The one subtraction

Start with the setting, because the setting is what makes this reinforcement learning and not a fancy name for a classifier. A support system routes a ticket to Billing. Later it finds out whether Billing was right. It never finds out what the other twenty-five departments would have done. That is bandit feedback: you learn the outcome of the action you took and nothing else. With full labels you would train a classifier and be done. You don't have them.

Reinforcement learning with verifiable rewards, RLVR, uses the obvious reward: sample a decision, get 1 if it was correct and 0 if not. The reward I used instead is

    r = c - p_a

where c is that same 1 or 0 and p_a is the probability the model had put on the option it sampled. Said 90 percent and was right: reward 0.1. Said 90 percent and was wrong: reward minus 0.9. Said 30 percent and was right: reward 0.7. The reward is literally how far your stated probability sat from what happened.

It is not a heuristic. A REINFORCE update with that reward is an unbiased estimate of the gradient of the Brier score, using only the outcome of the action you took, and the fixed point of the Brier score is a calibrated model. There is a numerical test of that identity in the repo because I did not trust myself; it holds to floating-point roundoff. The whole difference between the two training signals is that one subtraction. Same sampled actions, same seed, same batches, same optimizer.

## The setup

The model is Qwen3-0.6B-Base. The prompt lists a context, a question, and up to 26 lettered options, and the answer is the softmax over the 26 letter tokens at the last position. One forward pass, no decoding. The data is seven public classification sets plus a synthetic IT-triage generator I'll come back to, 64,000 training rows. Every arm starts from the same warmup: 6,400 labeled rows, one epoch, 0.748 accuracy, expected calibration error 0.022. Then 32,000 rows the warmup never saw, two passes, 500 optimizer steps, where the arms differ only in what they learn from:

- RLCD: outcome-only feedback, reward c - p_a.
- RLVR: outcome-only feedback, reward c. Run at the shared learning rate and at a fifth of it, because at the shared rate it destroys itself.
- Oracle: full labels on the same rows. The ceiling.
- A control that keeps training on the warmup's own 6,400 labels for the same number of steps, to separate "more steps" from "new outcomes."

Three seeds per arm, one on my 4080 and two on a Colab A100, evaluated on 8,000 held-out rows.

## What happened

| arm | accuracy | calibration error | mean confidence |
|---|---|---|---|
| warmup start | 0.748 | 0.022 | 0.76 |
| RLCD, outcomes only | 0.808 [0.806, 0.811] | 0.023 [0.020, 0.029] | 0.83 |
| RLVR, low learning rate | 0.778 [0.775, 0.780] | 0.213 [0.209, 0.216] | 0.99 |
| RLVR, shared rate (stopped at step 150) | 0.576 | 0.415 | 0.99 |
| supervised continuation, same 6,400 labels | 0.778 | 0.192 | 0.97 |
| oracle, full labels | 0.819 [0.817, 0.822] | 0.059 [0.046, 0.072] | 0.88 |

RLCD gained six points of accuracy from right-or-wrong feedback alone and ended with confidence 0.83 against accuracy 0.81. Its calibration error moved by less than a hundredth on every seed. It ends about one point behind the oracle, which had the actual labels, and better calibrated than it: the oracle's second pass over the labels pushed its confidence to 0.88.

RLVR at the shared settings collapsed in under 50 steps: confidence 0.99, accuracy below the warmup, stop rule. I expected that, and I expected the objection that nobody runs RLVR like that. So the fair arm runs at a fifth of the learning rate. It keeps its accuracy, gains three points over the warmup, and is still at 0.99 confidence with a calibration error of 0.21 on all three seeds, within a range of 0.007. That is not an optimizer accident. Under r = c, a perfectly calibrated policy still has a gradient pointing toward sharper. Under r = c - p_a, that gradient is zero.

The control is the row I care about most. Keep training on the labels you already have for the same number of steps and you get accuracy 0.778 with confidence 0.97, which is the RLVR row with a different label on it. The gain in the RLCD row did not come from optimizer steps. It came from 32,000 new outcomes and a reward that does not pay you for being sure.

## Where I can check the answer

Calibration error is a population statistic and it can hide a lot. So one dataset is built so that I know the true probability of every answer. Synthetic triage tickets carry cue phrases for a department. Thirty percent carry cues for two, and the generator picks the label between them with a coin flip, in shuffled order, so nothing in the text breaks the tie: the right answer there is 0.5 and 0.5, and on single-cue tickets it is 1.0. The escalate question has a 10 percent random flip in its labels, so the right confidence is 0.90.

On fresh tickets the model never saw:

| arm | one cue (ideal 1.0) | two cues (ideal 0.5) | escalate confidence (ideal 0.90) |
|---|---|---|---|
| warmup | 0.98 | 0.80 | 0.82 |
| RLCD | 0.97 | 0.59 | 0.93 |
| RLVR, low rate | 1.00 | 0.99 | 1.00 |
| oracle | 1.00 | 0.63 | 0.88 |

RLCD says 0.97 when the answer is certain and 0.59 when it is a coin flip. The RLVR model says 0.99 on the coin flips too. It has no idea it doesn't know. One caveat that I would rather say than have you find: part of RLCD's lower number on the ambiguous tickets is probability leaking to the two departments that weren't cued at all, about 8 percent of it. It got less sure, and also a bit less sharp. The oracle, with labels, splits the pair more cleanly.

## What this does not show

RLCD did not improve calibration. It held the warmup's calibration while learning. The best-calibrated model in the whole study is plain supervised training on 32,000 labels, one pass, at 0.014. If you have labels, use them. The point of the reward is the case where you don't, which is most of what happens after a model is deployed.

The RLVR comparison crosses learning rates, because the reward-only arm cannot survive the rate the others use, and I never ran RLCD at the low one. No KL term or entropy bonus was tried on the RLVR side, so "this recipe miscalibrates at these settings" is the honest scope. Everything is in-distribution, on public data, with prompts under 512 tokens. Calibration had a mid-run bump on every seed before it settled. And three seeds is three seeds.

I should also admit the first attempt. I started on my own 272M mixture-of-experts model, Eve-2. The contrast showed up there too, but the model could not read: it never learned entailment or BoolQ even with full labels, and the same reward gave it a confidence of 0.69 whether a ticket was certain or a coin flip. A bake-off put Qwen3-0.6B at 0.817 after a warmup where Eve managed 0.526, and that was that.

## The part where it won't talk

The last step was the interface the previous post argued for. You give the model one state and a list of typed questions. It runs the state once, caches it, and evaluates every question in one batched pass against the cache. Because attention is causal this is the same computation as asking each question alone, to within 1e-5 of the training-time path; reordering questions changes nothing, and adding a question cannot change the answer to another one.

On my 4080, eight questions against an 800-token state take 105 milliseconds, four times faster than one at a time. Sixty-four take 472 milliseconds against 3.7 seconds. Below about four questions the extra prefill costs more than it saves.

Here it is on a ticket: "API latency spiked to 4 seconds; load balancer returning 502s. Note: all customers affected." Department: INFRASTRUCTURE at 0.87. Priority: P0 at 0.99, score 0.995 on the ordered scale. Page the on-call engineer: 0.89. Sixty-six milliseconds. No sentence anywhere.

The exported checkpoint has the vocabulary projection removed and a 26-row decision head in its place. One honesty note: Qwen ties that projection to its input embedding, so a determined person could rebuild it. The export removes generation from the API. It does not make it physically impossible.

## So

None of this tells you what Jev is. It tells you that the objective TypeSafe described is coherent, that the one-subtraction version of it does what the description says under the feedback a deployed system actually gets, and that the obvious alternative reward reliably produces a model that is sure of everything. A day and a half on a gaming card, plus some Colab credit for the extra seeds, was enough to find that out. The bigger question from the last post, whether AI gets more dependable when we stop making every component talk, is still open. The calibrated-probability half of it is not vaporware.

Code and results: https://github.com/anthony-maio/eve-rlcd. The model card and the decision-only checkpoint are going up on Hugging Face under anthonym21.
