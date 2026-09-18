import json
import math

import pytest
import torch

from rlcd.decide import (ChoiceQ, Decider, NoulQ, ScoreQ, SharedPrefixCache, check_split, to_question)
from rlcd.policies import HFDecoderPolicy, load_policy
from rlcd.schema import NEG, Question, render_prefix, render_prompt, render_suffix

QWEN = "Qwen/Qwen3-0.6B-Base"


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN)


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=16, max_position_embeddings=4096,
                      tie_word_embeddings=True)
    return Qwen3ForCausalLM(cfg).eval()


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096)
    return LlamaForCausalLM(cfg).eval()


STATE = ("Ticket #4111 from a startup plan customer. Report: cannot find the settings page; "
         "API latency spiked to 4 seconds. Note: blocking our team.")
QUESTIONS = [ChoiceQ("Which department should handle this ticket?",
                     ["BILLING", "INFRASTRUCTURE", "SECURITY", "PRODUCT_SUPPORT"]),
             ScoreQ("What is the priority of this ticket?", ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]),
             NoulQ("an on-call engineer should be paged immediately"),
             ChoiceQ("Pick a letter", [f"option {j}" for j in range(26)])]


@pytest.fixture(scope="module")
def decider(qwen_tok):
    return Decider(HFDecoderPolicy(tiny_qwen3(), qwen_tok), device="cpu")


# ---------- typed primitives ----------

def test_primitives_become_validated_questions():
    q = to_question("  s ", QUESTIONS[0])
    assert q == Question("choice", "  s ", QUESTIONS[0].question, QUESTIONS[0].options)
    q = to_question("s", QUESTIONS[1])
    assert q.primitive == "score" and q.ordered is True and q.choices == QUESTIONS[1].levels
    q = to_question("s", QUESTIONS[2])
    assert q.primitive == "noul" and q.choices == ["true", "false"]
    assert q.question == "Is this true: an on-call engineer should be paged immediately"
    with pytest.raises(ValueError, match="need 2"):
        to_question("s", ChoiceQ("q", ["only"]))
    with pytest.raises(ValueError, match="need 2"):
        to_question("s", ScoreQ("q", [str(i) for i in range(27)]))
    with pytest.raises(ValueError, match="duplicate"):
        to_question("s", ChoiceQ("q", ["a", "a"]))
    with pytest.raises(TypeError, match="ChoiceQ, ScoreQ or NoulQ"):
        to_question("s", "not a question")


# ---------- the tokenization split ----------

def test_split_tokenization_holds_for_the_qwen_tokenizer_and_not_for_gpt2(qwen_tok):
    check_split(qwen_tok)
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    with pytest.raises(ValueError, match="prefix"):
        check_split(gpt2)


def test_split_tokenization_holds_across_random_prompts(qwen_tok):
    import random
    rng = random.Random(1)
    pieces = ["word", " tail.", "\n", "  ", "42", "!", "café", "(x)", "end:", "中文"]
    for _ in range(300):
        context = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 10)))
        q = Question("choice", context, "".join(rng.choice(pieces) for _ in range(3)), ["a", "b", "c"])
        whole = qwen_tok.encode(render_prompt(q), add_special_tokens=False)
        split = (qwen_tok.encode(render_prefix(q.context), add_special_tokens=False)
                 + qwen_tok.encode(render_suffix(q), add_special_tokens=False))
        assert whole == split, repr(context)


# ---------- truncation ----------

def test_state_is_left_truncated_and_the_header_survives(decider, qwen_tok):
    long_state = " ".join(f"w{i}" for i in range(400))
    prefix, qs = decider.render(long_state, QUESTIONS[:1], max_state_tokens=50, max_question_tokens=448)
    assert prefix.startswith("User: Context:\n") and prefix.endswith("\n\n")
    state = prefix[len("User: Context:\n"):-2]
    assert state.endswith("w399") and not state.startswith("w0")
    assert len(qwen_tok.encode(state, add_special_tokens=False)) <= 50
    assert qs[0].context == state
    # Short states pass through untouched apart from the strip that render_prompt applies anyway.
    prefix, _ = decider.render("  short  ", QUESTIONS[:1], 50, 448)
    assert prefix == "User: Context:\nshort\n\n"


def test_question_text_is_left_truncated_and_the_options_survive(decider, qwen_tok):
    long_q = ChoiceQ(" ".join(f"q{i}" for i in range(300)), ["alpha", "beta", "gamma"])
    _, qs = decider.render("s", [long_q], max_state_tokens=1536, max_question_tokens=40)
    suffix = render_suffix(qs[0])
    assert len(qwen_tok.encode(suffix, add_special_tokens=False)) <= 40
    assert qs[0].choices == ["alpha", "beta", "gamma"]
    assert qs[0].question.endswith("q299") and not qs[0].question.startswith("q0")
    assert suffix.endswith("A) alpha\nB) beta\nC) gamma\nAnswer with the letter only.\nAssistant: The answer is")
    with pytest.raises(ValueError, match="options"):
        decider.render("s", [ChoiceQ("q", [f"option {j}" for j in range(26)])], 1536, 20)


# ---------- cache expansion ----------

def test_shared_prefix_cache_expands_views_and_leaves_the_prefix_alone(decider):
    body = decider.policy._body()
    prefix_ids = torch.tensor([decider.tok.encode(render_prefix(STATE), add_special_tokens=False)])
    with torch.no_grad():
        prefix = body(input_ids=prefix_ids, attention_mask=torch.ones_like(prefix_ids), use_cache=True).past_key_values
    P = prefix_ids.shape[1]
    assert prefix.get_seq_length() == P
    before = [(layer.keys.clone(), layer.values.clone()) for layer in prefix.layers]
    cache = SharedPrefixCache(prefix)
    assert cache.get_seq_length() == P
    assert cache.get_mask_sizes(torch.arange(P, P + 5), 0) == (P + 5, 0)
    M, S = 3, 5
    layer0 = prefix.layers[0]
    new_k = torch.randn(M, layer0.keys.shape[1], S, layer0.keys.shape[3])
    k, v = cache.update(new_k, new_k + 1, 0)
    assert k.shape == (M, layer0.keys.shape[1], P + S, layer0.keys.shape[3]) and v.shape == k.shape
    assert torch.equal(k[:, :, :P], layer0.keys.expand(M, -1, -1, -1))
    assert torch.equal(k[:, :, P:], new_k)
    # The cache returned the expanded tensors but stored nothing: the prefix is reusable.
    assert cache.get_seq_length() == P
    for layer, (kk, vv) in zip(prefix.layers, before):
        assert torch.equal(layer.keys, kk) and torch.equal(layer.values, vv)
    # And a full ask leaves it alone too.
    with torch.no_grad():
        decider.probs_from_prefix(prefix, decider.render(STATE, QUESTIONS, 1536, 448)[1])
    for layer, (kk, vv) in zip(prefix.layers, before):
        assert torch.equal(layer.keys, kk) and layer.keys.shape[0] == 1


# ---------- ask ----------

def test_ask_returns_typed_dicts(decider):
    out = decider.ask(STATE, QUESTIONS)
    assert len(out) == 4
    choice, score, noul, wide = out
    assert choice["kind"] == "choice" and set(choice) == {"kind", "value", "probs", "confidence", "entropy_confidence"}
    assert list(choice["probs"]) == QUESTIONS[0].options
    assert abs(sum(choice["probs"].values()) - 1) < 1e-6
    assert choice["value"] == max(choice["probs"], key=choice["probs"].get)
    assert choice["confidence"] == pytest.approx(max(choice["probs"].values()))
    p = list(choice["probs"].values())
    h = -sum(x * math.log(x) for x in p)
    assert choice["entropy_confidence"] == pytest.approx(1 - h / math.log(4), abs=1e-6)
    assert score["kind"] == "score" and set(score) == set(choice) | {"score"}
    ps = list(score["probs"].values())
    assert score["score"] == pytest.approx(sum(i * x for i, x in enumerate(ps)) / 3, abs=1e-6)
    assert 0 <= score["score"] <= 1
    assert noul["kind"] == "noul" and set(noul) == {"kind", "p_true", "confidence"}
    assert noul["confidence"] == pytest.approx(max(noul["p_true"], 1 - noul["p_true"]))
    assert len(wide["probs"]) == 26 and abs(sum(wide["probs"].values()) - 1) < 1e-6
    json.dumps(out)  # plain floats and strings only
    assert all(isinstance(v, float) for v in choice["probs"].values())


def test_ask_matches_the_single_pass_path_and_is_independent_of_the_batch(decider):
    ref = decider.ask_sequential(STATE, QUESTIONS)
    got = decider.ask(STATE, QUESTIONS)
    _assert_same(got, ref, 1e-5)
    # Chunking changes nothing.
    _assert_same(decider.ask(STATE, QUESTIONS, batch_size=2), ref, 1e-5)
    # Independence: adding, removing and reordering other questions leaves a question alone.
    alone = decider.ask(STATE, QUESTIONS[:1])
    _assert_same(alone, ref[:1], 1e-5)
    reordered = decider.ask(STATE, QUESTIONS[::-1])
    _assert_same(reordered[::-1], ref, 1e-5)
    extra = decider.ask(STATE, QUESTIONS + [ChoiceQ("unrelated?", ["yes", "no", "maybe"])] * 5)
    _assert_same(extra[:4], ref, 1e-5)


def test_ask_on_a_llama_body_and_with_bos(qwen_tok):
    d = Decider(HFDecoderPolicy(tiny_llama(), qwen_tok), device="cpu")
    _assert_same(d.ask(STATE, QUESTIONS), d.ask_sequential(STATE, QUESTIONS), 1e-5)
    # A tokenizer that prepends BOS: both paths keep it at the front of the prefix.
    class Bos:
        bos_token_id = qwen_tok.eos_token_id

        def __getattr__(self, name):
            return getattr(qwen_tok, name)

    d.tok = d.policy.tok = Bos()
    d.policy.prepend_bos = True
    _assert_same(d.ask(STATE, QUESTIONS), d.ask_sequential(STATE, QUESTIONS), 1e-5)
    ids = d.tok.encode(render_prefix(STATE), add_special_tokens=False)
    assert d.prefill(render_prefix(STATE)).get_seq_length() == len(ids) + 1


def test_probs_are_masked_beyond_k_and_sum_to_one(decider):
    probs = decider.probs(STATE, QUESTIONS[:3])
    assert probs.shape == (3, 26) and probs.dtype == torch.float32
    assert torch.all(probs[0, 4:] == 0) and torch.all(probs[2, 2:] == 0)
    assert torch.allclose(probs.sum(1), torch.ones(3))
    logits = decider.logits(STATE, QUESTIONS[:3])
    assert torch.all(logits[2, 2:] == NEG)


def test_ask_of_no_questions_is_empty(decider):
    assert decider.ask(STATE, []) == []
    assert decider.ask_sequential(STATE, []) == []


def test_load_from_a_saved_checkpoint(tmp_path, qwen_tok):
    policy = HFDecoderPolicy(tiny_qwen3(), qwen_tok, origin=QWEN)
    policy.save(str(tmp_path / "ckpt"), {"steps": 1})
    d = Decider.load(str(tmp_path / "ckpt"), device="cpu")
    assert isinstance(d.policy, HFDecoderPolicy) and not d.policy.model.training
    assert d.autocast is False  # cpu never autocasts
    ref = Decider(policy, device="cpu").ask(STATE, QUESTIONS)
    _assert_same(d.ask(STATE, QUESTIONS), ref, 1e-6)


def _assert_same(got: list[dict], want: list[dict], tol: float) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g["kind"] == w["kind"]
        if g["kind"] == "noul":
            assert abs(g["p_true"] - w["p_true"]) <= tol
        else:
            assert list(g["probs"]) == list(w["probs"])
            for key in g["probs"]:
                assert abs(g["probs"][key] - w["probs"][key]) <= tol, key
