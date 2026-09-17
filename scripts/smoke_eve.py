"""Load the real Eve-2-IT, verify the tie, and check zero-shot letter behavior."""
import torch

from rlcd.policy import decision_logits, load_eve, questions_to_batch
from rlcd.schema import Question, letter_token_ids

EXPECTED_LETTER_IDS = [317, 347, 327, 360, 412, 376, 402, 367, 314, 449, 509, 406, 337, 399, 440,
                       350, 1195, 371, 311, 309, 471, 569, 370, 1395, 575, 1168]

model, tok = load_eve(device="cuda")
model.eval()
letters = letter_token_ids(tok)

# 0. Routing and tokenizer sanity.
print("config.top_k", model.config.top_k)
print("h[0].mlp.top_k", model.transformer.h[0].mlp.top_k)
print("letter ids match expected", letters == EXPECTED_LETTER_IDS)
print("tied", model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())
print("param dtype", next(model.parameters()).dtype, "freqs_cis dtype", model.freqs_cis.dtype)

# 1. Plain generation still works (proves weights loaded sensibly).
# The HF wrapper has no GenerationMixin on current transformers, so decode greedily by hand.
ids = tok.encode("User: What is the capital of France?\nAssistant:", return_tensors="pt").cuda()
start = ids.shape[1]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(20):
        nxt = model(input_ids=ids).logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
print("GEN:", repr(tok.decode(ids[0][start:])))

# 2. Zero-shot decisions on three obvious questions.
qs = [
    Question("choice", "The customer writes: my card was charged twice for one order.",
             "What is the customer's intent?", ["cancel order", "refund request", "track shipment", "change password"], answer=1),
    Question("noul", "Paris is the capital of France.", "Is the statement true?", ["true", "false"], answer=0),
    Question("score", "This movie was an absolute masterpiece, I cried.", "What is the sentiment?",
             ["very negative", "negative", "neutral", "positive", "very positive"], ordered=True, answer=4),
]
ids, last, k = questions_to_batch(tok, qs, device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits, aux = decision_logits(model, ids, last, letters, k)
probs = torch.softmax(logits, -1)
for q, p in zip(qs, probs):
    top = p[: q.k].tolist()
    print(q.primitive, "answer", q.answer, "probs", [round(x, 3) for x in top])
print("aux", float(aux))
print("VRAM MB", torch.cuda.max_memory_allocated() // 2**20)
