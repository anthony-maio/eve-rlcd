import json

import pytest
import torch
from safetensors.torch import load_file

from rlcd.decide import ChoiceQ, Decider, NoulQ, ScoreQ
from rlcd.export import DECISION_FILE, HEAD_FILE, DecisionOnlyPolicy, export, load_decision_only, main
from rlcd.policies import HFDecoderPolicy, load_policy
from rlcd.schema import LETTERS

QWEN = "Qwen/Qwen3-0.6B-Base"
STATE = "Ticket #77 from an enterprise, tier 1 customer. Report: pods crash looping in prod. Note: production is down."
QUESTIONS = [ChoiceQ("Which department should handle this ticket?", ["BILLING", "INFRASTRUCTURE", "SECURITY"]),
             ScoreQ("What is the priority of this ticket?", ["P3_LOW", "P2_NORMAL", "P1_HIGH", "P0_CRITICAL"]),
             NoulQ("Should an on-call engineer be paged immediately?")]


@pytest.fixture(scope="module")
def qwen_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(QWEN)


def tiny_llama(tie: bool = False):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=4096,
                      tie_word_embeddings=tie)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture()
def checkpoint(tmp_path, qwen_tok):
    policy = HFDecoderPolicy(tiny_llama(), qwen_tok, origin=QWEN)
    policy.save(str(tmp_path / "ckpt"), {"steps": 500, "arm": "rlcd"})
    return str(tmp_path / "ckpt")


def _same(a: list[dict], b: list[dict], tol: float = 1e-6) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["kind"] == y["kind"]
        if x["kind"] == "noul":
            assert abs(x["p_true"] - y["p_true"]) <= tol
        else:
            assert list(x["probs"]) == list(y["probs"])
            assert all(abs(x["probs"][k] - y["probs"][k]) <= tol for k in x["probs"])


def test_export_writes_the_body_the_head_rows_and_the_record(checkpoint, tmp_path, qwen_tok):
    out = tmp_path / "decision"
    record = export(checkpoint, str(out))
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", HEAD_FILE,
                 DECISION_FILE, "README.md"):
        assert (out / name).is_file(), name
    assert not (out / "policy.json").exists()  # not a training checkpoint any more
    weights = load_file(str(out / "model.safetensors"))
    assert not any(k.startswith("lm_head") for k in weights)
    assert "embed_tokens.weight" in weights and "layers.0.self_attn.q_proj.weight" in weights
    head = load_file(str(out / HEAD_FILE))
    original = load_policy(checkpoint, device="cpu")
    assert head["weight"].shape == (26, 32)
    assert torch.equal(head["weight"], original.model.lm_head.weight[torch.tensor(original.letters)].detach())
    assert "bias" not in head
    assert json.loads((out / DECISION_FILE).read_text()) == record
    assert record["letters"] == LETTERS and record["letter_ids"] == original.letters
    assert record["base_model"] == QWEN and record["source"] == checkpoint
    assert record["training"] == {"steps": 500, "arm": "rlcd"}
    assert record["prepend_bos"] is False and record["pad_id"] == qwen_tok.pad_token_id
    assert record["tied_embeddings"] is False
    assert set(record["primitives"]) == {"choice", "score", "noul"}
    assert "entropy_confidence" in record["confidence"]
    assert record["prompt"]["prefix"] == "User: Context:\n{state}\n\n"
    assert record["prompt"]["suffix"].startswith("Question: {question}\nOptions:\n")
    assert "removes the generation path from the API" in record["note"]
    readme = (out / "README.md").read_text()
    assert "removes the generation path from the API" in readme and "Decider.load" in readme


def test_the_export_answers_exactly_like_the_original_policy(checkpoint, tmp_path):
    out = tmp_path / "decision"
    export(checkpoint, str(out))
    original = Decider.load(checkpoint, device="cpu")
    exported = Decider.load(str(out), device="cpu")
    assert isinstance(exported.policy, DecisionOnlyPolicy)
    assert not exported.policy.model.training
    assert exported.policy.model.get_output_embeddings() is None  # a body, not a language model
    _same(exported.ask(STATE, QUESTIONS), original.ask(STATE, QUESTIONS))
    _same(exported.ask_sequential(STATE, QUESTIONS), original.ask_sequential(STATE, QUESTIONS))
    # The training-time entry point works on it too, with the same fp32 masked logits.
    from rlcd.schema import Question
    qs = [Question("choice", STATE, q.question, q.options) for q in QUESTIONS[:1]]
    with torch.no_grad():
        a, _ = original.policy.decision_logits(qs, 512, "cpu")
        b, _ = exported.policy.decision_logits(qs, 512, "cpu")
    assert torch.allclose(a, b, atol=1e-6)


def test_a_tied_model_says_so_in_the_note(tmp_path, qwen_tok):
    policy = HFDecoderPolicy(tiny_llama(tie=True), qwen_tok, origin="Qwen/Qwen3-0.6B-Base")
    policy.save(str(tmp_path / "tied"), {})
    record = export(str(tmp_path / "tied"), str(tmp_path / "decision"))
    assert record["tied_embeddings"] is True
    assert record["note"] == ("Qwen3-0.6B ties its output head to the input embedding, so the vocabulary projection "
                              "is recoverable from the embedding; the export removes the generation path from the "
                              "API, it does not make generation physically impossible.")
    exported = Decider.load(str(tmp_path / "decision"), device="cpu")
    _same(exported.ask(STATE, QUESTIONS), Decider(policy, device="cpu").ask(STATE, QUESTIONS))


def test_export_records_hashes_and_refuses_tampered_weights(checkpoint, tmp_path):
    import hashlib
    out = tmp_path / "decision"
    record = export(checkpoint, str(out))
    for name in ("model.safetensors", HEAD_FILE):
        assert record["sha256"][name] == hashlib.sha256((out / name).read_bytes()).hexdigest()
    load_decision_only(str(out), "cpu")
    readme = (out / "README.md").read_text()
    assert "fix_mistral_regex" in readme and "unaffected" in readme
    # A flipped byte in the head is caught before anything runs.
    head = bytearray((out / HEAD_FILE).read_bytes())
    head[-1] ^= 1
    (out / HEAD_FILE).write_bytes(head)
    with pytest.raises(RuntimeError, match=HEAD_FILE):
        load_decision_only(str(out), "cpu")
    export(checkpoint, str(out))
    body = bytearray((out / "model.safetensors").read_bytes())
    body[-1] ^= 1
    (out / "model.safetensors").write_bytes(body)
    with pytest.raises(RuntimeError, match="model.safetensors"):
        load_decision_only(str(out), "cpu")


def test_export_refuses_an_adapter_and_a_tampered_record(checkpoint, tmp_path):
    adapter = load_policy(checkpoint, device="cpu", lora=True)
    adapter.save(str(tmp_path / "adapter"), {})
    with pytest.raises(ValueError, match="LoRA"):
        export(str(tmp_path / "adapter"), str(tmp_path / "decision"))
    out = tmp_path / "decision"
    export(checkpoint, str(out))
    record = json.loads((out / DECISION_FILE).read_text())
    (out / DECISION_FILE).write_text(json.dumps(record | {"letter_ids": list(range(26))}))
    with pytest.raises(RuntimeError, match="letter"):
        load_decision_only(str(out), "cpu")


def test_export_cli_and_resave(checkpoint, tmp_path):
    out = tmp_path / "decision"
    main(["--src", checkpoint, "--out", str(out)])
    policy = load_decision_only(str(out), "cpu")
    policy.save(str(tmp_path / "again"), {"resaved": True})
    again = json.loads((tmp_path / "again" / DECISION_FILE).read_text())
    assert again["training"] == {"steps": 500, "arm": "rlcd"} and again["resaved"] is True
    _same(Decider.load(str(tmp_path / "again"), "cpu").ask(STATE, QUESTIONS),
          Decider(policy, "cpu").ask(STATE, QUESTIONS))
