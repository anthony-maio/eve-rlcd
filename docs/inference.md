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
     NoulQ("an on-call engineer should be paged immediately")])
```

The primitives:

- `ChoiceQ(question, options)`: 2..26 unordered options. Answer: `kind`, `value` (the most probable
  option), `probs` (option -> probability), `confidence`, `entropy_confidence`.
- `ScoreQ(question, levels)`: 2..26 ordered levels, low to high. Answer: the choice fields plus `score`.
- `NoulQ(proposition)`: rendered as the question `Is this true: <proposition>` with the options `true`
  and `false`. Answer: `kind`, `p_true`, `confidence`.

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

`scripts/check_decide.py --model runs/q-rlcd` (results in `runs/decide-bench/equivalence.json`).
300 test rows in 205 context groups (56 groups with 2 to 37 questions sharing a context, the rest
single), every group asked together, and again with 1, 5 and 20 unrelated questions from other rows
mixed in. Every probability over the declared options is compared. Tokenization split first: over all
8000 test rows, tokenizing the prefix and the suffix separately gives exactly the tokens of the whole
prompt on every row (0 failures); the fallback split before the last newline of the prefix fails on all
8000 rows, because the Qwen3 tokenizer keeps `\n\n` as one token.

With the body in fp32 (tolerance 1e-5): pass.

| check | max abs diff | mean abs diff |
|---|---|---|
| ask vs sequential | 2.3e-6 | 9.4e-8 |
| ask with 1, 5, 20 unrelated questions added vs sequential | 3.3e-6 | 1.3e-7 |
| independence: adding unrelated questions | 4.4e-6 | 1.1e-7 |
| independence: asking one question alone | 1.8e-6 | 1.4e-7 |
| independence: reversing the order | 0 | 0 |
| sequential at batch size 1 vs batch size 32 (the reference against itself) | 2.4e-6 | 4.6e-8 |

Under bf16 autocast (tolerance 1e-4): does not pass, and the numbers say why.

| check | max abs diff | mean abs diff |
|---|---|---|
| ask vs sequential | 2.7e-2 | 6.0e-4 |
| ask with unrelated questions added vs sequential | 3.0e-2 | 5.4e-4 |
| independence: adding unrelated questions | 2.6e-2 | 3.3e-4 |
| independence: asking one question alone | 1.0e-2 | 3.7e-4 |
| independence: reversing the order | 0 | 0 |
| sequential at batch size 1 vs batch size 32 (the reference against itself) | 3.4e-2 | 3.2e-4 |
| ask (bf16) vs sequential (fp32) | 1.9e-2 | 6.2e-4 |
| sequential (bf16) vs sequential (fp32) | 3.7e-2 | 6.7e-4 |

The fp32 agreement at 2e-6 shows that the split, the positions and the mask are exact. The bf16
differences are not from the cached path: the training-time path disagrees with itself by 3.4e-2 when
only the batch size changes, and the bf16 cached path sits closer to the fp32 single pass (1.9e-2) than
the bf16 training-time path does (3.7e-2). What moves the numbers is that a bf16 matmul or attention
call gives slightly different roundings when its shape changes (the batch and sequence dimensions pick
the kernel and the reduction order), and those roundings compound over 28 layers; reordering questions
keeps every shape and gives bit-identical results, adding a question changes the shapes and does not.
No two computations with different shapes agree to 1e-4 in bf16 on this model, so the bf16 bound is
not something the cached path can meet against a reference that does not meet it either. The maxima
sit at near-tie decisions; the mean absolute difference is 6e-4.

## Latency

`scripts/bench_decide.py`, RTX 4080, bf16 autocast with the fp32 head, medians of 7 runs after 2
warmups, in milliseconds. `ask` is one prefill of the state and one batched forward over the question
suffixes; `sequential` is every full prompt through the training-time path, right-padded in batches of
32. The state is a synthetic ticket log cut to the stated token count; the questions are 4-option (or
k-option) choice questions in the triage style. Peak memory 4.8 GB (the fp32 weights are 2.4 GB).

(a) question count, one 800-token state, 4 options each

| questions | ask (ms) | sequential (ms) | speedup |
|---|---|---|---|
| 1 | 99.7 | 55.9 | 0.56x |
| 2 | 99.3 | 65.0 | 0.65x |
| 4 | 94.2 | 107.1 | 1.14x |
| 8 | 89.5 | 229.7 | 2.57x |
| 16 | 114.7 | 508.1 | 4.43x |
| 32 | 174.5 | 1023.7 | 5.87x |
| 64 | 305.5 | 2044.4 | 6.69x |

(b) state length, 8 questions, 4 options each

| state tokens | ask (ms) | sequential (ms) | speedup |
|---|---|---|---|
| 200 | 93.4 | 60.7 | 0.65x |
| 800 | 86.8 | 228.6 | 2.63x |
| 1500 | 109.5 | 503.0 | 4.59x |

(c) option count, 8 questions, one 800-token state

| options | ask (ms) | sequential (ms) | speedup |
|---|---|---|---|
| 2 | 93.1 | 226.8 | 2.44x |
| 5 | 103.1 | 231.2 | 2.24x |
| 10 | 98.2 | 242.1 | 2.47x |
| 26 | 114.0 | 275.7 | 2.42x |

How to read it. The sequential path scales with questions x (state + suffix) tokens, the cached path
with state + questions x suffix tokens, so `ask` is nearly flat in the state length and grows slowly in
the question count. Below about 4 questions on an 800-token state, or on a 200-token state, the
sequential path is faster: a single forward of this 28-layer model in eager mode costs about 45 ms of
launch overhead regardless of length (a 40-token forward and an 840-token forward take the same time),
and `ask` is two forwards. Above that the shared prefix pays for itself, up to 6.7x at 64 questions.

One implementation detail found by the benchmark: the sdpa attention integration of transformers 4.57
passes `enable_gqa=True` to torch whenever there is no attention mask, and on this torch build (no
flash attention compiled in) grouped-query attention without a mask falls back to the unfused math
kernel, about 8x slower per layer than the fused kernel that runs when the key/value heads are
repeated. A batch-1 prefill has no padding and so no mask, which made the prefill of a 1500-token
state cost about 140 ms. `Decider.prefill` therefore passes an explicit 4D causal mask, which routes
the prefill to the fused kernel (52 ms for 1500 tokens) and leaves the result unchanged (the fp32
equivalence above was measured with it).

## Accuracy sanity

`ask` over the whole test split, grouped by context (8000 rows in 6936 groups, dataset noul rows asked
with their own question text so the prompts are the training ones), scored with `rlcd.eval.summarize`
like `runs/q-rlcd/eval/metrics.json`, bf16 autocast in both:

| | ask | eval | diff |
|---|---|---|---|
| accuracy | 0.8071 | 0.8074 | -0.0002 |
| ECE (15 bins) | 0.0219 | 0.0214 | +0.0005 |
| Brier loss | 0.2682 | 0.2682 | 0.0000 |

Both within the 0.002 bound. The residual is the bf16 noise described above: the same check run before
the prefill switched attention kernels (see Latency) gave 0.8080 and 0.0209, so the third and fourth
digits move with the kernels, which is also why an evaluation number should be quoted with its
bootstrap interval rather than to four digits.

## The decision-only export

`python -m rlcd.export --src runs/q-rlcd --out runs/q-rlcd/decision` writes the model body (every
weight except the language-model head), the tokenizer, `decision_head.safetensors` with the 26 letter
rows of the output head, `decision.json` (letters and their token ids, the prompt template pieces, the
primitive and confidence definitions, the base model id, the training summary from `meta.json`, and
the note below) and a `README.md` stub. `Decider.load` recognises the directory by its `decision.json`
and loads it strictly: every tensor must match, the tokenizer must still give the recorded letter ids
and BOS behaviour, and the loaded model must be a body without an output head.

The export of `runs/q-rlcd` is 2400 MB: `model.safetensors` 2384.2 MB (the body in fp32; the
embedding, which is also the tied head, stays because the body needs it as its input embedding),
`tokenizer.json` 11.4 MB, `vocab.json` 2.8 MB, `merges.txt` 1.7 MB, `decision_head.safetensors`
0.1 MB (26 x 1024 fp32), plus `config.json`, `decision.json`, `README.md`, `tokenizer_config.json`,
`special_tokens_map.json`, `added_tokens.json` and the `chat_template.jinja` the tokenizer carries.
`tests/test_export.py` exports a tiny untied model and checks that the export's `ask` output equals
the original policy's, that the head rows equal the original `lm_head` rows, that no `lm_head` tensor
is saved, that a LoRA adapter is refused, and that a tampered `decision.json` is refused on load.

`scripts/demo_decide.py` runs one ticket and three questions (department choice, priority score,
page-on-call noul) on the export and prints the JSON answers.

Note: Qwen3-0.6B ties its output head to the input embedding, so the vocabulary projection is
recoverable from the embedding; the export removes the generation path from the API, it does not
make generation physically impossible.
