# Inference: the decision API

This describes `rlcd/decide.py`, `rlcd/export.py` and the scripts under `scripts/` that check them. It
describes only what this code does.

## What the API does

A `Decider` holds a trained hf-decoder policy (here `runs/q-rlcd`, Qwen3-0.6B-Base after supervised
warmup and RLCD). `ask(state, questions)` takes one state string and a list of typed questions and
returns one typed answer per question, in order, without generating any text:

```python
from rlcd.decide import ChoiceQ, Decider, NoulQ, ScoreQ

d = Decider.load("runs/q-rlcd/decision")
answers = d.ask(
    "Ticket #4242 from an enterprise, tier 1 customer. Report: API latency spiked to 4 seconds; "
    "load balancer returning 502s. Note: all customers affected.",
    [ChoiceQ("Which department should handle this ticket?",
             ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]),
     ScoreQ("What is the priority of this ticket?", ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]),
     NoulQ("Should an on-call engineer be paged immediately?")])
```

The primitives:

- `ChoiceQ(question, options)`: 2..26 unordered options. Answer: `kind`, `value` (the most probable
  option), `probs` (option -> probability), `confidence`, `entropy_confidence`.
- `ScoreQ(question, levels)`: 2..26 ordered levels, low to high. Answer: the choice fields plus `score`.
- `NoulQ(question)`: a yes/no question, rendered verbatim with the options `true` and `false`, exactly
  as the noul rows of the training data are rendered (their questions are literal yes/no questions
  such as `Should an on-call engineer be paged immediately?`). Answer: `kind`, `p_true`, `confidence`.
  Phrase it as a question; a bare proposition is a prompt the model was not trained on.

`ask` runs the model body in fp32 by default, which makes it agree with the training-time path to
about 1e-6 (see Equivalence). `fast=True`, on the call or on `Decider.load`, runs the body under bf16
autocast on CUDA: cheaper at large question counts (see Latency), at the cost of the bf16 rounding noise
that the training-time path itself has. The decision head runs in fp32 either way.

The trained and evaluated range is prompts of up to 512 tokens (the training `max_len`; the longest
test prompt is 473 tokens). The API accepts states up to `max_state_tokens` (default 1536) and question
suffixes up to `max_question_tokens` (default 448), but a prompt beyond about 500 tokens is outside
the range the model was trained and evaluated on, and its calibration there is unmeasured.

The confidence fields are our own definitions, not standard ones:

- `confidence` is the largest probability over the declared options (the probability of `value`).
  For a noul question it is `max(p_true, 1 - p_true)`.
- `entropy_confidence` is `1 - H(p) / ln K`, with `H` the entropy in nats of the distribution over the
  `K` declared options: 1 for a point mass, 0 for a uniform distribution.
- `score` is the probability-weighted mean level index divided by `K - 1`, so it lies in `[0, 1]` with 0
  the lowest level and 1 the highest.

Every question is rendered exactly as in training (`rlcd.schema.render_prompt`), the readout is the
softmax over the 26 letter rows of the output embedding at the last prompt token, and letters beyond the
option count are masked. The prompt format is unchanged; the API only changes how the prompt is run.

## The prefix-sharing argument

The training prompt is

```
User: Context:
<state>

Question: <question>
Options:
A) ...
Answer with the letter only.
Assistant: The answer is
```

`render_prefix(state)` is everything up to and including the blank line, and `render_suffix(q)` is
everything from `Question:` onward; their concatenation is `render_prompt(q)` character for character
(`tests/test_schema.py`). The model is a causal decoder, so the hidden states at the prefix positions
depend on the prefix only. `ask` therefore:

1. tokenizes the prefix once, runs the model body over it once with `use_cache=True`, and keeps the
   key/value cache;
2. tokenizes every suffix separately, right-pads them into a batch, builds the attention mask as
   `[ones(prefix_len) | suffix_mask]` and sets `cache_position` and `position_ids` to continue from
   `prefix_len`;
3. runs one body forward over the batch, reading the cache through `SharedPrefixCache` (the prefix
   expanded across the batch as a view, so it is neither copied per question nor kept after the
   forward), gathers the last real position of each suffix, and applies the 26-row fp32 head with
   autocast disabled (`HFDecoderPolicy.logits_from_hidden`);
4. batches large question lists in chunks of 32; the prefix is computed once regardless.

Each question attends to the state and to its own suffix and nothing else, so the questions are
evaluated in isolation, and the logits are the ones the single full-prompt pass gives.

Two conditions have to hold for this to be exact, and both are checked:

- Tokenizing the prefix and the suffix separately must give the tokens of the whole prompt. With the
  Qwen3 tokenizer the blank line before `Question:` is a token of its own and a letter never merges
  with a preceding newline, so the split holds. `scripts/check_decide.py` and
  `tests/test_decide.py` check it, and `Decider` refuses a tokenizer for which it fails (gpt2's
  regex is one: `\n\n` followed by a letter tokenizes as two newlines, but `\n\n` alone as one token,
  so the fallback split at the last newline of the prefix fails on every row).
- Truncation must not change the rendered text between the two paths. The state is left-truncated to
  `max_state_tokens` (default 1536) before rendering, so the header always survives, and a question is
  left-truncated from its question text only, never from its options block, until its suffix fits
  `max_question_tokens` (default 448). Both `ask` and `ask_sequential` render from the same truncated
  pieces.

`Decider.ask_sequential` renders every full prompt and runs it through the training-time path
(`Policy.decision_logits`, right-padded batches of 32). It exists for the equivalence proof and the
latency comparison.

## Equivalence and independence

`scripts/check_decide.py --model runs/q-rlcd` (results in `runs/decide-bench/equivalence.json`, log in
`runs/decide-bench/check.log`). 300 test rows in 205 context groups (56 groups with 2 to 37 questions
sharing a context, the rest single), every group asked together, and again with 1, 5 and 20 unrelated
questions from other rows mixed in. Every probability over the declared options is compared. Dataset
rows go through the primitives that render them verbatim (`NoulQ` included). Tokenization split first:
over all 8000 test rows, tokenizing the prefix and the suffix separately gives exactly the tokens of
the whole prompt on every row (0 failures); the fallback split before the last newline of the prefix
fails on all 8000 rows, because the Qwen3 tokenizer keeps `\n\n` as one token.

With the body in fp32, the default (tolerance 1e-5):

| check | max abs diff | mean abs diff | probabilities |
|---|---|---|---|
| ask vs sequential | 3.7e-6 | 9.8e-8 | 1493 |
| ask with 1, 5, 20 unrelated questions added vs sequential | 1.08e-5 | 1.3e-7 | 4479 |
| independence: adding unrelated questions | 4.4e-6 | 1.0e-7 | 4479 |
| independence: asking one question alone | 1.9e-6 | 1.2e-7 | 194 |
| independence: reversing the order | 0 | 0 | 486 |
| sequential at batch size 1 vs batch size 32 (the reference against itself) | 6.6e-6 | 6.1e-8 | 1493 |

Every check but one is under 1e-5; the one is a single probability out of 4479 at 1.08e-5, in the
check that mixes in 20 unrelated questions, and the reference path disagrees with itself by 6.6e-6 over
the same rows when only its batch size changes. So the script reports the fp32 check as failed at the
stated tolerance and the numbers are given as measured, rather than the tolerance being moved; the
agreement is at the fp32 rounding floor of the reference itself (`torch.get_float32_matmul_precision()`
is `highest`, TF32 off). The same check on the real checkpoint with eager attention instead of sdpa
(120 rows): ask vs sequential 1.4e-6, and the eager and sdpa results agree with each other at 2.2e-6.

Under bf16 autocast, `fast=True` (tolerance 1e-4): does not pass, and the numbers say why.

| check | max abs diff | mean abs diff |
|---|---|---|
| ask vs sequential | 2.2e-2 | 6.1e-4 |
| ask with unrelated questions added vs sequential | 4.3e-2 | 5.3e-4 |
| independence: adding unrelated questions | 4.1e-2 | 3.4e-4 |
| independence: asking one question alone | 1.0e-2 | 3.6e-4 |
| independence: reversing the order | 0 | 0 |
| sequential at batch size 1 vs batch size 32 (the reference against itself) | 2.3e-2 | 3.1e-4 |
| ask (bf16) vs sequential (fp32) | 1.9e-2 | 6.2e-4 |
| sequential (bf16) vs sequential (fp32) | 2.9e-2 | 6.8e-4 |

The fp32 agreement shows that the split, the positions and the mask are exact. The bf16 differences
are not from the cached path: the training-time path disagrees with itself by 2.3e-2 when only the
batch size changes, and the bf16 cached path sits closer to the fp32 single pass (1.9e-2) than the bf16
training-time path does (2.9e-2). What moves the numbers is that a bf16 matmul or attention call gives
slightly different roundings when its shape changes (the batch and sequence dimensions pick the kernel
and the reduction order), and those roundings compound over 28 layers; reordering questions keeps
every shape and gives bit-identical results, adding a question changes the shapes and does not. No
two computations with different shapes agree to 1e-4 in bf16 on this model, which is why fp32 is the
default and bf16 is an explicit `fast=True`. The bf16 maxima sit at near-tie decisions; the mean
absolute difference is 6e-4.

## Latency

`scripts/bench_decide.py`, RTX 4080, medians of 7 runs after 2 warmups, in milliseconds. `ask` is one
prefill of the state and one batched forward over the question suffixes; `sequential` is every full
prompt through the training-time path, right-padded in batches of 32. `fp32` is the body in fp32 (the
default); `fast` is the body under bf16 autocast (`fast=True`); the decision head is fp32 in every
column. The state is a synthetic ticket log cut to the stated token count; the questions are 4-option
(or k-option) choice questions in the triage style. Peak memory 4.8 GB (the fp32 weights are 2.4 GB).
Speedups are sequential over ask at the same precision.

(a) question count, one 800-token state, 4 options each

| questions | ask fp32 | ask fast | sequential fp32 | sequential fast | speedup fp32 | speedup fast |
|---|---|---|---|---|---|---|
| 1 | 95.0 | 101.2 | 69.4 | 55.7 | 0.73x | 0.55x |
| 2 | 97.0 | 83.0 | 109.3 | 55.2 | 1.13x | 0.67x |
| 4 | 92.2 | 88.4 | 216.3 | 102.7 | 2.35x | 1.16x |
| 8 | 105.3 | 84.8 | 440.9 | 222.2 | 4.19x | 2.62x |
| 16 | 148.5 | 110.4 | 910.0 | 491.3 | 6.13x | 4.45x |
| 32 | 261.2 | 162.8 | 1832.2 | 995.3 | 7.01x | 6.11x |
| 64 | 472.2 | 302.3 | 3679.1 | 1983.5 | 7.79x | 6.56x |

(b) state length, 8 questions, 4 options each

| state tokens | ask fp32 | ask fast | sequential fp32 | sequential fast | speedup fp32 | speedup fast |
|---|---|---|---|---|---|---|
| 200 | 65.7 | 84.7 | 117.8 | 61.5 | 1.79x | 0.73x |
| 800 | 106.9 | 79.1 | 444.2 | 222.4 | 4.16x | 2.81x |
| 1500 | 188.3 | 111.6 | 989.5 | 486.9 | 5.25x | 4.36x |

(c) option count, 8 questions, one 800-token state

| options | ask fp32 | ask fast | sequential fp32 | sequential fast | speedup fp32 | speedup fast |
|---|---|---|---|---|---|---|
| 2 | 102.2 | 86.3 | 437.9 | 219.7 | 4.28x | 2.54x |
| 5 | 115.4 | 79.9 | 446.3 | 223.9 | 3.87x | 2.80x |
| 10 | 129.7 | 87.1 | 473.7 | 236.0 | 3.65x | 2.71x |
| 26 | 179.6 | 106.7 | 532.4 | 267.8 | 2.96x | 2.51x |

How to read it. The sequential path scales with questions x (state + suffix) tokens, the cached path
with state + questions x suffix tokens, so `ask` is nearly flat in the state length and grows slowly in
the question count. The fp32 body costs `ask` little at small sizes (0.94x at 1 question, where the
autocast bookkeeping of `fast` costs more than the bf16 kernels save) and more as the work grows: 1.24x
at 8 questions, 1.60x at 32, 1.56x at 64, 1.69x at a 1500-token state. Below about 2 questions on an
800-token state the sequential path is faster in either precision: a single forward of this 28-layer
model in eager mode costs about 45 ms of launch overhead regardless of length (a 40-token forward and
an 840-token forward take the same time), and `ask` is two forwards. Above that the shared prefix pays
for itself, up to 7.8x (fp32) or 6.6x (fast) at 64 questions.

One implementation detail found by the benchmark: the sdpa attention integration of transformers 4.57
passes `enable_gqa=True` to torch whenever there is no attention mask, and on this torch build (no
flash attention compiled in) grouped-query attention without a mask falls back to the unfused math
kernel, about 8x slower per layer than the fused kernel that runs when the key/value heads are
repeated. A batch-1 prefill has no padding and so no mask, which made the prefill of a 1500-token
state cost about 140 ms. `Decider.prefill` therefore passes an explicit causal mask, which routes the
prefill to the fused kernel (52 ms for 1500 tokens). The mask is an additive 4D float mask in the dtype
of the attention scores (bf16 under `fast`, else the weights' dtype), because sdpa refuses a float
mask of another dtype and eager attention adds whatever mask it is given to the scores, so a boolean
mask would have made an eager-attention prefill acausal; `tests/test_decide.py` asserts that `ask`
equals `ask_sequential` under both `sdpa` and `eager` and that the two implementations agree with each
other. The explicit mask leaves the fp32 results unchanged (the fp32 equivalence above was measured
with it) and changes the bf16 rounding, since a different kernel runs.

## Accuracy sanity

`ask` with the fp32 body over the whole test split, grouped by context (8000 rows in 6936 groups; the
1333 noul rows go through `NoulQ`, which renders their questions verbatim), scored with
`rlcd.eval.summarize` like `runs/q-rlcd/eval/metrics.json` (which was computed under bf16 autocast):

| | ask (fp32) | eval (bf16) | diff |
|---|---|---|---|
| accuracy | 0.8080 | 0.8074 | +0.0006 |
| ECE (15 bins) | 0.0210 | 0.0214 | -0.0004 |
| Brier loss | 0.2683 | 0.2682 | +0.0001 |

Both within the 0.002 bound. The residual is the bf16 rounding of the eval itself (see above): an
earlier run of this check under bf16 gave 0.8071 and 0.0219, so the third and fourth digits move with
the kernels, which is also why an evaluation number should be quoted with its bootstrap interval
rather than to four digits.

## The decision-only export

`python -m rlcd.export --src runs/q-rlcd --out runs/q-rlcd/decision` writes the model body (every
weight except the language-model head), the tokenizer, `decision_head.safetensors` with the 26 letter
rows of the output head, `decision.json` (letters and their token ids, the prompt template pieces, the
primitive and confidence definitions, the base model id, the training summary from `meta.json`, and
the note below) and a `README.md` stub. `Decider.load` recognises the directory by its `decision.json`
and loads it strictly: `decision.json` records the sha256 of `model.safetensors` and
`decision_head.safetensors` and both are verified before anything runs, every tensor must match, the
tokenizer must still give the recorded letter ids and BOS behaviour, and the loaded model must be a
body without an output head. Loading the tokenizer prints a transformers 4.57 warning about an
"incorrect regex pattern" and `fix_mistral_regex`; it is spurious for this tokenizer (it is not a
Mistral tokenizer), and the encodings are unaffected and are the ones the model was trained on. The
README stub says the same.

The export of `runs/q-rlcd` is 2400 MB: `model.safetensors` 2384.2 MB (the body in fp32; the
embedding, which is also the tied head, stays because the body needs it as its input embedding),
`tokenizer.json` 11.4 MB, `vocab.json` 2.8 MB, `merges.txt` 1.7 MB, `decision_head.safetensors`
0.1 MB (26 x 1024 fp32), plus `config.json`, `decision.json`, `README.md`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json` and the `chat_template.jinja` the tokenizer carries.
`tests/test_export.py` exports a tiny untied model and checks that the export's `ask` output equals
the original policy's, that the head rows equal the original `lm_head` rows, that no `lm_head` tensor
is saved, that a LoRA adapter is refused, and that a tampered `decision.json` or a flipped byte in
either weight file is refused on load.

`scripts/demo_decide.py` runs one ticket and three questions (department choice, priority score,
page-on-call noul) on the export and prints the JSON answers.

Note: Qwen3-0.6B ties its output head to the input embedding, so the vocabulary projection is
recoverable from the embedding; the export removes the generation path from the API, it does not
make generation physically impossible.
