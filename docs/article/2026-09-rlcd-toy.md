# The model that won't talk, built to test the thesis

Everyone has fixated on the wrong half of Jev. The won't-talk half is a decoding trick, and there are eight repos on the awesome-jev list doing it already. The half that matters is RLCD, the training objective that supposedly makes the probabilities mean something, and nobody had run it. So I did.

Reinforcement learning for calibrated decisions turns out to fit in one subtraction. A 0.6B model trained with it says 0.60 when the true answer is a coin flip. Trained without it, same data, same loop, the model says 0.99. The weights are on Hugging Face and the rest of this post is the receipt.

## The problem the reward has to solve

A deployed support system routes a ticket to Billing. Later it finds out whether Billing was right. It never finds out what the other twenty-five departments would have done. That is bandit feedback, one outcome per action taken, and it is what most systems live on after they ship. With full labels you would train a classifier and be done. The question is what you do when you don't have them.

RLVR, reinforcement learning with verifiable rewards, answers with the obvious reward, r = c: 1 if the decision was right, 0 if not. The reward I used instead is

    r = c - p_a

where c is that same 1 or 0 and p_a is the probability the model placed on the action it took. Said 90 percent and was right: reward 0.1. Said 90 percent and was wrong: reward -0.9. Said 30 percent and was right: reward 0.7. The reward is the distance between what you said you believed and what turned out to be true.

A REINFORCE update with that reward is an unbiased estimate of the gradient of the Brier score, under bandit feedback, and the fixed point of the Brier score is a calibrated model. There is a numerical test of that identity in the repo because I did not want to trust it on paper. It holds to floating-point roundoff. The whole difference between the two training signals is the subtraction. Same sampled actions, same seed, same batches, same optimizer.

## Setup

The model is Qwen3-0.6B-Base. The prompt is a context, a question, and up to 26 lettered options; the answer is the softmax over the 26 letter tokens at the last position. One forward pass, no generation loop. Seven public classification datasets plus a synthetic IT-triage generator, 64,000 rows.

Every arm starts from the same warmup: 6,400 labeled rows, one epoch, 0.748 accuracy, calibration error 0.022. Then 32,000 rows the warmup never touched, two passes, 500 optimizer steps, and the arms differ only in what they learn from:

- RLCD: outcomes only, reward c - p_a.
- RLVR: outcomes only, reward c. Run at the shared learning rate and at a fifth of it, because the shared rate destroys it.
- Oracle: full labels on the same rows. The ceiling.
- Control: keep training on the warmup's own labels for the same step count. It separates "more training" from "new outcomes."

Three seeds each, one on my 4080 and two on a Colab A100, evaluated on 8,000 held-out rows.

## What happened

| arm | accuracy | calibration error | mean confidence |
|---|---|---|---|
| warmup start | 0.748 | 0.022 | 0.76 |
| RLCD, outcomes only | 0.808 [0.806, 0.811] | 0.023 [0.020, 0.029] | 0.83 |
| RLVR, low learning rate | 0.778 [0.775, 0.780] | 0.213 [0.209, 0.216] | 0.99 |
| RLVR, shared rate (stopped at step 150) | 0.576 | 0.415 | 0.99 |
| supervised continuation, warmup labels | 0.778 | 0.192 | 0.97 |
| oracle, full labels | 0.819 [0.817, 0.822] | 0.059 [0.046, 0.072] | 0.88 |

RLCD gained six accuracy points from right-or-wrong feedback, its calibration error moved less than a hundredth on every seed, and it ends one point behind the oracle. It is also better calibrated than the oracle, whose second pass over the same labels pushed its confidence to 0.88 without a matching accuracy gain.

RLVR collapsed at the shared learning rate in under 50 steps. The low-rate arm is the fair comparison. It gains three accuracy points and ends at 0.99 confidence with 0.21 calibration error on all three seeds, inside a range of 0.007, which is not optimizer noise. Under r = c a perfectly calibrated model still has a gradient pushing it sharper, because confident correct answers keep earning reward. Under r = c - p_a that gradient is zero at calibration.

The control row is the one I would argue with hardest if I were reviewing this. Continue training on the warmup labels for the same number of steps and you get 0.778 accuracy at 0.97 confidence, which is the RLVR row under a different name. The RLCD gain did not come from more steps. It came from 32,000 new outcomes and a reward that does not pay for certainty.

## Checking the ground truth directly

Calibration error is a population statistic and it can average over a lot of structure. So I built a dataset where I know the true probability of every answer: synthetic triage tickets where 30 percent carry cues for two departments and the generator picks between them with a coin flip. Nothing in the text resolves the ambiguity, so the right answer there is 0.5 and 0.5, and on single-cue tickets it is 1.0. The escalate label has a 10 percent random flip, so the right confidence there is 0.90.

| arm | one cue (ideal 1.0) | two cues (ideal 0.5) | escalate (ideal 0.90) |
|---|---|---|---|
| warmup | 0.98 | 0.80 | 0.82 |
| RLCD | 0.97 | 0.59 | 0.93 |
| RLVR, low rate | 1.00 | 0.99 | 1.00 |
| oracle | 1.00 | 0.63 | 0.88 |

RLCD says 0.97 on the certain tickets and 0.59 on the coin flips. RLVR says 0.99 on both. It doesn't know that it doesn't know, and 0.99 is a very confident way to not know. One caveat: about 8 percent of RLCD's probability on the ambiguous tickets leaks to departments that weren't cued at all. It got less sure and a little less sharp. The oracle, working from labels, splits the pair more cleanly.

## What this doesn't show

The best-calibrated model in the whole study is plain supervised training on 32,000 labeled rows, one pass, calibration error 0.014. If you have the labels, use them. The reward is for the case where you don't, which is most of what happens after a model ships.

The RLVR comparison runs at two learning rates because the reward-only arm cannot survive the rate the others use. No KL or entropy term was tried on the RLVR side, so the accurate scope is "this recipe miscalibrates at these settings," not "RLVR is broken." Everything is in-distribution, public data, prompts under 512 tokens, and three seeds is three seeds.

The first attempt was on my own 272M mixture-of-experts model, Eve-2. The contrast showed up there too, but Eve could not read. It never learned entailment or BoolQ even with full labels, and the same reward gave it a confidence of 0.69 whether the ticket was certain or a coin flip. A bake-off put Qwen3-0.6B at 0.817 warmup accuracy where Eve managed 0.526. That was that.

## The interface

The last step was the one the previous post argued mattered: a model that does not generate text at all. You give it one state and a list of typed questions. It runs the state once, caches the key-value pairs, and evaluates every question in one batched pass against the cache. Because attention is causal, this is the same computation as asking each question alone, to within 1e-5 of the training-time path. Question order does not matter, and adding a question cannot change the answer to another one.

On my 4080, eight questions against an 800-token state take 105 milliseconds, four times faster than one at a time. Sixty-four questions take 472 milliseconds against 3.7 seconds. Below four questions the prefill costs more than it saves.

On an actual ticket, "API latency spiked to 4 seconds; load balancer returning 502s. Note: all customers affected." Department: INFRASTRUCTURE, 0.87. Priority: P0, 0.99. Page the on-call engineer: 0.89. Sixty-six milliseconds. No sentence produced.

The exported checkpoint has the vocabulary projection replaced with a 26-row decision head. Qwen ties that projection to its input embedding, so a determined person could rebuild it. The export removes generation from the API rather than making it physically impossible.

## So

None of this tells you what Jev is. It tells you that the objective TypeSafe described is coherent, that the one-subtraction version does what the description says under bandit feedback, and that the obvious alternative produces a model that is 0.99 confident about coin flips. A day and a half on a gaming card and some Colab credit was enough to find that out. Whether AI systems get more dependable when you stop making every component generate language is still open. The calibrated-probability half of that argument isn't.

Code and results: https://github.com/anthony-maio/eve-rlcd. Checkpoint and model card: https://huggingface.co/anthonym21/qwen3-0.6b-rlcd-decision.
