import json
import os

import pytest
import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policies import (EvePolicy, HFDecoderPolicy, HFMaskedLMPolicy, load_policy, resolve_backend)
from rlcd.policy import EVE_ID, decision_logits, questions_to_batch
from rlcd.schema import NEG, Question, letter_token_ids, render_prompt


@pytest.fixture(scope="module")
def gpt2_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")


def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=50257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256)
    return LlamaForCausalLM(cfg).eval()


def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=50257, n_embd=32, n_layer=2, n_head=2, n_positions=256,
                     resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    return GPT2LMHeadModel(cfg).eval()


def _questions() -> list[Question]:
    return [Question("choice", "a short context", "pick one", ["x", "y", "z"], answer=1, source="alpha"),
            Question("noul", "a considerably longer context that pads the other row " * 3, "is it so",
                     ["true", "false"], answer=0, source="beta"),
            Question("choice", "ctx", "pick", [f"option {j}" for j in range(26)], answer=25, source="alpha")]


def _full_logit_slice(policy, q: Question) -> torch.Tensor:
    """The model's own full-vocabulary logits at the last prompt token, sliced to the letters."""
    ids = torch.tensor([policy.tok.encode(render_prompt(q), add_special_tokens=False)])
    full = policy.model(input_ids=ids).logits
    return full[0, -1, policy.letters]


@pytest.mark.parametrize("build", [tiny_llama, tiny_gpt2])
def test_hf_decoder_logits_equal_the_models_own_full_logit_slice(build, gpt2_tok):
    policy = HFDecoderPolicy(build(), gpt2_tok)
    qs = _questions()
    with torch.no_grad():
        got, aux = policy.decision_logits(qs, max_len=512, device="cpu")
        for i, q in enumerate(qs):
            want = _full_logit_slice(policy, q)
            assert torch.allclose(got[i, : q.k], want[: q.k], atol=1e-4)
    assert got.shape == (3, 26)
    assert got.dtype == torch.float32
    assert aux.ndim == 0 and aux.item() == 0.0


def test_hf_decoder_padded_batch_matches_single_rows(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    qs = _questions()
    with torch.no_grad():
        batched, _ = policy.decision_logits(qs, max_len=512, device="cpu")
        for i, q in enumerate(qs):
            single, _ = policy.decision_logits([q], max_len=512, device="cpu")
            assert torch.allclose(batched[i], single[0], atol=1e-4)


def test_hf_decoder_masks_beyond_k(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    qs = _questions()[:2]
    with torch.no_grad():
        logits, _ = policy.decision_logits(qs, max_len=512, device="cpu")
    assert torch.all(logits[0, 3:] == NEG)
    assert torch.all(logits[1, 2:] == NEG)
    assert torch.isfinite(logits).all()
    assert torch.softmax(logits, -1)[0, 3:].sum() == 0


def test_hf_decoder_truncates_from_the_left_and_keeps_the_tail(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    q = _questions()[1]
    ids, mask, last = policy.encode([q], max_len=16, device="cpu")
    tail = gpt2_tok.encode(render_prompt(q), add_special_tokens=False)[-16:]
    assert ids.shape == (1, 16)
    assert ids[0].tolist() == tail
    assert mask[0].tolist() == [1] * 16
    assert last.tolist() == [15]


def test_hf_decoder_right_pads_with_an_attention_mask(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    qs = _questions()[:2]
    ids, mask, last = policy.encode(qs, max_len=512, device="cpu")
    lengths = [len(gpt2_tok.encode(render_prompt(q), add_special_tokens=False)) for q in qs]
    assert last.tolist() == [n - 1 for n in lengths]
    assert mask.sum(1).tolist() == lengths
    short = lengths.index(min(lengths))
    assert mask[short, lengths[short]:].sum() == 0
    assert torch.all(ids[short, lengths[short]:] == gpt2_tok.eos_token_id)  # gpt2 has no pad token


class BosTok:
    """A tokenizer that prepends a BOS id by default, the way Llama-style tokenizers do."""
    bos_token_id = 7
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, s, add_special_tokens=True):
        body = [ord(c) % 1000 + 10 for c in s]
        return [self.bos_token_id] + body if add_special_tokens else body


def test_hf_decoder_keeps_the_bos_token_when_the_tokenizer_adds_one():
    policy = HFDecoderPolicy(tiny_llama(), BosTok(), letters=list(range(100, 126)))
    q = _questions()[0]
    body = BosTok().encode(render_prompt(q), add_special_tokens=False)
    ids, mask, last = policy.encode([q], max_len=512, device="cpu")
    assert ids[0].tolist() == [7] + body
    ids, mask, last = policy.encode([q], max_len=16, device="cpu")
    assert ids[0].tolist() == [7] + body[-15:]
    assert last.tolist() == [15]


def test_hf_decoder_rejects_a_tokenizer_without_single_token_letters():
    class SplitTok(BosTok):
        def encode(self, s, add_special_tokens=True):
            return [ord(c) % 1000 + 10 for c in s]  # " A" becomes two tokens

    with pytest.raises(ValueError, match="single token"):
        HFDecoderPolicy(tiny_llama(), SplitTok())


def test_hf_decoder_train_eval_and_parameters(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    assert policy.name == "hf-decoder"
    assert policy.aux_coef() == 0.0
    assert policy.letters == letter_token_ids(gpt2_tok)
    policy.train()
    assert policy.model.training
    policy.eval()
    assert not policy.model.training
    params = list(policy.trainable_parameters())
    assert len(params) == len(list(policy.model.parameters()))
    assert list(policy.trainable_parameters()) == params  # a fresh iterable on every call


def test_hf_decoder_gradients_reach_the_body_and_the_letter_rows(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok)
    policy.train()
    logits, _ = policy.decision_logits(_questions(), max_len=512, device="cpu")
    torch.log_softmax(logits, -1)[:, 0].sum().backward()
    grads = [p.grad for p in policy.trainable_parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    head = policy.model.get_output_embeddings().weight.grad
    assert head[policy.letters].abs().sum() > 0


def test_hf_decoder_save_and_load_round_trip_through_policy_json(tmp_path, gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama(), gpt2_tok, origin="some/base")
    out = tmp_path / "ckpt"
    policy.save(str(out), {"steps": 3})
    record = json.loads((out / "policy.json").read_text())
    assert record["backend"] == "hf-decoder" and record["lora"] is False
    assert record["base"] is None and record["origin"] == "some/base"  # a full fine-tune sits on nothing
    assert json.loads((out / "meta.json").read_text()) == {"steps": 3}

    assert resolve_backend(str(out), "auto") == "hf-decoder"
    loaded = load_policy(str(out), device="cpu")
    assert isinstance(loaded, HFDecoderPolicy)
    assert all(p.dtype == torch.float32 for p in loaded.model.parameters())
    qs = _questions()
    policy.eval()
    loaded.eval()
    with torch.no_grad():
        a, _ = policy.decision_logits(qs, 512, "cpu")
        b, _ = loaded.decision_logits(qs, 512, "cpu")
    assert torch.allclose(a, b, atol=1e-6)


def test_hf_decoder_lora_trains_only_adapters_and_round_trips(tmp_path, gpt2_tok):
    base_dir = tmp_path / "base"
    HFDecoderPolicy(tiny_llama(), gpt2_tok).save(str(base_dir), {})

    policy = load_policy(str(base_dir), device="cpu", lora=True)
    names = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    trainable = list(policy.trainable_parameters())
    assert len(trainable) == len(names)
    _perturb_lora_b(policy)

    out = tmp_path / "adapter"
    policy.save(str(out), {"steps": 1})
    record = json.loads((out / "policy.json").read_text())
    assert record == {"backend": "hf-decoder", "lora": True, "base": str(base_dir), "origin": None,
                      "revision": None, "prepend_bos": False, "grad_checkpointing": False}
    assert (out / "adapter_config.json").is_file()
    assert not (out / "model.safetensors").exists()  # the adapter only, never the base weights

    loaded = load_policy(str(out), device="cpu")
    qs = _questions()
    policy.eval()
    loaded.eval()
    with torch.no_grad():
        a, _ = policy.decision_logits(qs, 512, "cpu")
        b, _ = loaded.decision_logits(qs, 512, "cpu")
        plain, _ = load_policy(str(base_dir), device="cpu").decision_logits(qs, 512, "cpu")
    assert torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a[0, :3], plain[0, :3], atol=1e-5)


def _perturb_lora_b(policy) -> None:
    """lora_B starts at zero; move it so the adapter changes the output."""
    with torch.no_grad():
        for n, p in policy.model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05 * torch.randn_like(p))


def test_lora_over_a_full_fine_tune_directory_reloads_against_that_directory(tmp_path, gpt2_tok):
    """A LoRA started from a saved full fine-tune must come back on those fine-tuned weights, not
    on the hub lineage the fine-tune itself was started from."""
    sft = tmp_path / "sft"
    HFDecoderPolicy(tiny_llama(), gpt2_tok, origin="some/hub-base").save(str(sft), {})
    assert json.loads((sft / "policy.json").read_text())["origin"] == "some/hub-base"

    policy = load_policy(str(sft), device="cpu", lora=True)
    assert policy.base == str(sft) and policy.origin == "some/hub-base"
    _perturb_lora_b(policy)
    out = tmp_path / "adapter"
    policy.save(str(out), {"steps": 1})
    record = json.loads((out / "policy.json").read_text())
    assert record["base"] == str(sft)            # the weights the adapter was trained against
    assert record["origin"] == "some/hub-base"   # the lineage, for remote-code decisions only
    assert record["revision"] is None            # a directory has no hub commit

    loaded = load_policy(str(out), device="cpu")
    assert loaded.base == str(sft) and loaded.origin == "some/hub-base"
    qs = _questions()
    policy.eval()
    loaded.eval()
    with torch.no_grad():
        a, _ = policy.decision_logits(qs, 512, "cpu")
        b, _ = loaded.decision_logits(qs, 512, "cpu")
    assert torch.allclose(a, b, atol=1e-6)


def test_a_hub_base_records_its_commit_hash_and_reloads_at_that_revision(tmp_path, gpt2_tok, monkeypatch):
    import rlcd.policies as policies
    from transformers import AutoModelForCausalLM
    sha = "0123456789abcdef0123456789abcdef01234567"
    calls = []

    def fake_from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        model = tiny_llama()
        model.config._commit_hash = sha if not os.path.isdir(name) else None
        info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        return (model, info) if kwargs.get("output_loading_info") else model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(policies, "_load_tokenizer", lambda *a, **k: gpt2_tok)
    policy = load_policy("fake/hub-model", device="cpu", lora=True)
    assert (policy.base, policy.origin, policy.revision) == ("fake/hub-model", "fake/hub-model", sha)
    assert "revision" not in calls[-1][1]  # a fresh load takes whatever main is, and records it
    out = tmp_path / "adapter"
    policy.save(str(out), {})
    record = json.loads((out / "policy.json").read_text())
    assert (record["base"], record["origin"], record["revision"]) == ("fake/hub-model", "fake/hub-model", sha)

    loaded = load_policy(str(out), device="cpu")
    assert calls[-1][0] == "fake/hub-model" and calls[-1][1]["revision"] == sha
    assert loaded.revision == sha


def _edit_safetensors(path, drop: str | None = None, add: dict | None = None) -> None:
    from safetensors.torch import load_file, save_file
    sd = load_file(str(path))
    if drop is not None:
        del sd[drop]
    sd.update(add or {})
    save_file(sd, str(path), metadata={"format": "pt"})


def test_a_checkpoint_with_a_missing_tensor_is_refused(tmp_path, gpt2_tok):
    out = tmp_path / "ckpt"
    HFDecoderPolicy(tiny_llama(), gpt2_tok).save(str(out), {})
    _edit_safetensors(out / "model.safetensors", drop="model.layers.1.mlp.up_proj.weight")
    with pytest.raises(RuntimeError, match="model.layers.1.mlp.up_proj.weight"):
        load_policy(str(out), device="cpu")


def test_a_checkpoint_with_an_unexpected_tensor_is_refused(tmp_path, gpt2_tok):
    out = tmp_path / "ckpt"
    HFDecoderPolicy(tiny_llama(), gpt2_tok).save(str(out), {})
    _edit_safetensors(out / "model.safetensors", add={"model.layers.7.stray.weight": torch.zeros(2)})
    with pytest.raises(RuntimeError, match="model.layers.7.stray.weight"):
        load_policy(str(out), device="cpu")


def test_an_adapter_with_a_missing_tensor_is_refused(tmp_path, gpt2_tok):
    base_dir = tmp_path / "base"
    HFDecoderPolicy(tiny_llama(), gpt2_tok).save(str(base_dir), {})
    policy = load_policy(str(base_dir), device="cpu", lora=True)
    out = tmp_path / "adapter"
    policy.save(str(out), {})
    from safetensors.torch import load_file
    key = next(k for k in load_file(str(out / "adapter_model.safetensors")) if "layers.0" in k and "lora_A" in k)
    _edit_safetensors(out / "adapter_model.safetensors", drop=key)
    with pytest.raises(RuntimeError, match="lora_A"):
        load_policy(str(out), device="cpu")


def test_hf_decoder_gradient_checkpointing_gives_the_same_gradients(tmp_path, gpt2_tok):
    base_dir = tmp_path / "base"
    HFDecoderPolicy(tiny_llama(), gpt2_tok).save(str(base_dir), {})
    grads = []
    for flag in (False, True):
        policy = load_policy(str(base_dir), device="cpu", grad_checkpointing=flag)
        policy.train()
        assert policy.model.is_gradient_checkpointing == flag
        logits, _ = policy.decision_logits(_questions(), 512, "cpu")
        torch.log_softmax(logits, -1)[:, 0].sum().backward()
        grads.append([p.grad.clone() for p in policy.trainable_parameters() if p.grad is not None])
    assert len(grads[0]) == len(grads[1]) > 0
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_hf_decoder_logits_are_fp32_under_bf16_autocast(gpt2_tok):
    policy = HFDecoderPolicy(tiny_llama().to("cuda"), gpt2_tok)
    qs = _questions()
    seen = []
    policy.model.model.layers[0].mlp.down_proj.register_forward_hook(lambda m, a, out: seen.append(out.dtype))
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = policy.decision_logits(qs, 512, "cuda")
            ids, mask, last = policy.encode(qs, 512, "cuda")
            hidden = policy.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        assert seen == [torch.bfloat16, torch.bfloat16]  # the body really ran under autocast
        rows = hidden[torch.arange(len(qs), device="cuda"), last]
        weight = policy.model.lm_head.weight[torch.as_tensor(policy.letters, device="cuda")]
        ref = rows.float() @ weight.float().t()
    assert logits.dtype == torch.float32
    assert aux.dtype == torch.float32
    for i, q in enumerate(qs):
        assert torch.all(logits[i, q.k:] == NEG)
        assert torch.allclose(logits[i, : q.k], ref[i, : q.k], atol=1e-6)


# ---------- the Eve backend ----------

class FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny_eve() -> EveMoEForCausalLM:
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=256,
                     num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg)


def test_eve_policy_is_the_existing_policy_functions_unchanged():
    model = tiny_eve().eval()
    letters = list(range(100, 126))
    policy = EvePolicy(model, FakeTok(), letters=letters)
    qs = _questions()
    with torch.no_grad():
        got, got_aux = policy.decision_logits(qs, max_len=128, device="cpu")
        ids, last, k = questions_to_batch(FakeTok(), qs, 128, "cpu")
        want, want_aux = decision_logits(model, ids, last, letters, k)
    assert torch.equal(got, want)
    assert torch.equal(got_aux, want_aux)
    assert policy.name == "eve"
    assert policy.aux_coef() == model.config.router_aux_loss_coef == 0.01
    assert [id(p) for p in policy.trainable_parameters()] == [id(p) for p in model.parameters()]


def test_eve_policy_save_and_auto_load(tmp_path, gpt2_tok):
    model = tiny_eve()
    model.lm_head.weight = model.transformer.wte.weight
    policy = EvePolicy(model, gpt2_tok)
    out = tmp_path / "eve"
    policy.save(str(out), {"steps": 9})
    assert json.loads((out / "policy.json").read_text())["backend"] == "eve"
    assert json.loads((out / "meta.json").read_text()) == {"steps": 9}
    loaded = load_policy(str(out), device="cpu")
    assert isinstance(loaded, EvePolicy)
    assert loaded.letters == letter_token_ids(gpt2_tok)
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], tensor), name


def test_resolve_backend(tmp_path):
    assert resolve_backend(EVE_ID, "auto") == "eve"
    assert resolve_backend("Qwen/Qwen3-0.6B-Base", "auto") == "hf-decoder"
    assert resolve_backend("Qwen/Qwen3-0.6B-Base", "hf-mlm") == "hf-mlm"
    # An Eve checkpoint written before policy.json existed is recognised by its config.json.
    old = tmp_path / "old-eve"
    old.mkdir()
    (old / "config.json").write_text(json.dumps({"model_type": "eve-moe"}))
    assert resolve_backend(str(old), "auto") == "eve"
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert resolve_backend(str(other), "auto") == "hf-decoder"
    (other / "policy.json").write_text(json.dumps({"backend": "hf-mlm"}))
    assert resolve_backend(str(other), "auto") == "hf-mlm"
    with pytest.raises(ValueError):
        resolve_backend(str(other), "nonsense")


def test_lora_is_refused_for_the_eve_backend(tmp_path, gpt2_tok):
    model = tiny_eve()
    model.lm_head.weight = model.transformer.wte.weight
    EvePolicy(model, gpt2_tok).save(str(tmp_path / "eve"), {})
    with pytest.raises(ValueError, match="LoRA"):
        load_policy(str(tmp_path / "eve"), device="cpu", lora=True)


# ---------- the masked-LM backend ----------

def tiny_bert(vocab_size: int):
    from transformers import BertConfig, BertForMaskedLM
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256, hidden_dropout_prob=0.0,
                     attention_probs_dropout_prob=0.0)
    return BertForMaskedLM(cfg).eval()


@pytest.fixture()
def mask_tok():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({"mask_token": "<mask>"})
    return tok


def test_hf_mlm_reads_the_letters_at_the_mask_position(mask_tok):
    policy = HFMaskedLMPolicy(tiny_bert(len(mask_tok)), mask_tok)
    qs = _questions()
    ids, mask, pos = policy.encode(qs, max_len=512, device="cpu")
    assert all(ids[i, pos[i]] == mask_tok.mask_token_id for i in range(len(qs)))
    assert (ids == mask_tok.mask_token_id).sum() == len(qs)
    with torch.no_grad():
        got, aux = policy.decision_logits(qs, max_len=512, device="cpu")
        full = policy.model(input_ids=ids, attention_mask=mask).logits
        for i, q in enumerate(qs):
            want = full[i, pos[i], policy.letters]
            assert torch.allclose(got[i, : q.k], want[: q.k], atol=1e-4)
            assert torch.all(got[i, q.k:] == NEG)
            single, _ = policy.decision_logits([q], max_len=512, device="cpu")
            assert torch.allclose(got[i], single[0], atol=1e-4)
    assert policy.name == "hf-mlm"
    assert aux.item() == 0.0


def test_hf_mlm_never_feeds_a_padded_row_to_the_body(mask_tok):
    """A bidirectional convolution reads one token to the right of the mask, and transformers 4.x
    does not zero LFM2 padding states, so the policy groups rows by length instead of padding."""
    policy = HFMaskedLMPolicy(tiny_bert(len(mask_tok)), mask_tok)
    qs = _questions() + _questions()[:1]  # two rows share a length and go through together
    shapes = []

    def check(module, args, kwargs):
        assert kwargs["attention_mask"].all()
        assert not (kwargs["input_ids"] == policy.pad_id).any()
        shapes.append(tuple(kwargs["input_ids"].shape))

    policy.model.bert.register_forward_pre_hook(check, with_kwargs=True)
    with torch.no_grad():
        logits, _ = policy.decision_logits(qs, max_len=512, device="cpu")
    assert len(shapes) == 3 and sorted(s[0] for s in shapes) == [1, 1, 2]
    assert torch.equal(logits[0], logits[3])


def test_hf_mlm_truncation_keeps_the_mask_token(mask_tok):
    policy = HFMaskedLMPolicy(tiny_bert(len(mask_tok)), mask_tok)
    ids, mask, pos = policy.encode([_questions()[1]], max_len=16, device="cpu")
    assert ids.shape == (1, 16)
    assert ids[0, pos[0]] == mask_tok.mask_token_id


def test_remote_code_is_only_trusted_for_reviewed_and_pinned_revisions(tmp_path):
    from rlcd.policies import PINNED_REMOTE_CODE, remote_code_kwargs
    assert set(PINNED_REMOTE_CODE) <= {"LiquidAI/LFM2.5-Encoder-350M"}
    for pin in PINNED_REMOTE_CODE.values():
        assert len(pin.revision) == 40 and set(pin.revision) <= set("0123456789abcdef")  # a full commit sha
        for name, digest in pin.files.items():
            assert name.endswith(".py")
            assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")  # a full sha256
    assert remote_code_kwargs("someone/unknown-custom-encoder") == {}
    assert remote_code_kwargs("Qwen/Qwen3-0.6B-Base") == {}
    for repo, pin in PINNED_REMOTE_CODE.items():
        assert remote_code_kwargs(repo) == {"trust_remote_code": True, "revision": pin.revision}
    # A directory that names no custom code needs no trust, whatever its origin.
    assert remote_code_kwargs(str(tmp_path), origin="LiquidAI/LFM2.5-Encoder-350M") == {}
    assert remote_code_kwargs(str(tmp_path), origin="someone/unknown-custom-encoder") == {}


def test_the_pinned_encoder_table_matches_the_reviewed_file():
    import hashlib
    from huggingface_hub import try_to_load_from_cache

    from rlcd.policies import PINNED_REMOTE_CODE
    pin = PINNED_REMOTE_CODE["LiquidAI/LFM2.5-Encoder-350M"]
    assert pin.revision == "b886781f7c6f10ca9b7096e21b83e30a073c2f39"
    assert pin.files == {"modeling_lfm2_bidirectional.py":
                         "f171f518be2a07da48b17fdea5655cad0a2452ab548e90e8ae903143686647e2"}
    cached = try_to_load_from_cache("LiquidAI/LFM2.5-Encoder-350M", "modeling_lfm2_bidirectional.py",
                                    revision=pin.revision)
    if not isinstance(cached, str):
        pytest.skip("the pinned encoder is not in the local Hugging Face cache")
    with open(cached, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == pin.files["modeling_lfm2_bidirectional.py"]


def _custom_code_dir(tmp_path, source: str, auto_map: dict, name: str = "modeling_fake.py"):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / name).write_bytes(source.encode())  # exact bytes: write_text would add \r on Windows
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert", "auto_map": auto_map}))
    return str(tmp_path)


def test_local_custom_code_is_trusted_only_when_it_hashes_to_the_reviewed_file(tmp_path, monkeypatch):
    """save_pretrained copies the modeling file into the run directory and names it in auto_map
    without a repo prefix, and transformers then imports that local file, ignoring
    code_revision. So the local file itself has to match the reviewed one."""
    import hashlib

    import rlcd.policies as policies
    from rlcd.policies import PinnedCode, remote_code_kwargs
    genuine = "import torch\nclass FakeModel(torch.nn.Module):\n    pass\n"
    pin = PinnedCode("b" * 40, {"modeling_fake.py": hashlib.sha256(genuine.encode()).hexdigest()})
    monkeypatch.setattr(policies, "PINNED_REMOTE_CODE", {"fake/pinned": pin})
    auto_map = {"AutoModelForMaskedLM": "modeling_fake.FakeModel"}

    good = _custom_code_dir(tmp_path / "good", genuine, auto_map)
    assert remote_code_kwargs(good, origin="fake/pinned") == {"trust_remote_code": True,
                                                              "code_revision": "b" * 40}
    assert remote_code_kwargs(good, origin="someone/else") == {}
    assert remote_code_kwargs(good) == {}

    tampered = _custom_code_dir(tmp_path / "tampered", genuine + "import os\n", auto_map)
    with pytest.raises(RuntimeError, match="modeling_fake.py"):
        remote_code_kwargs(tampered, origin="fake/pinned")

    # A second module smuggled in through another auto_map entry is refused too, as is a
    # missing file, and a prefixed reference that points at some other repo.
    other = _custom_code_dir(tmp_path / "other", genuine, auto_map | {"AutoConfig": "configuration_x.Cfg"})
    (tmp_path / "other" / "configuration_x.py").write_text("x = 1\n")
    with pytest.raises(RuntimeError, match="configuration_x.py"):
        remote_code_kwargs(other, origin="fake/pinned")
    missing = _custom_code_dir(tmp_path / "missing", genuine, auto_map)
    (tmp_path / "missing" / "modeling_fake.py").unlink()
    with pytest.raises(RuntimeError, match="modeling_fake.py"):
        remote_code_kwargs(missing, origin="fake/pinned")
    elsewhere = _custom_code_dir(tmp_path / "elsewhere", genuine,
                                 {"AutoModel": "someone/else--modeling_fake.FakeModel"} | auto_map)
    with pytest.raises(RuntimeError, match="someone/else"):
        remote_code_kwargs(elsewhere, origin="fake/pinned")
    # A prefixed reference back at the pinned repo is fetched at code_revision, so it is fine.
    prefixed = _custom_code_dir(tmp_path / "prefixed", genuine,
                                {"AutoModel": "fake/pinned--modeling_fake.FakeModel"} | auto_map)
    assert remote_code_kwargs(prefixed, origin="fake/pinned")["trust_remote_code"] is True
    # Tokenizer code named in tokenizer_config.json is held to the same rule.
    tok_dir = _custom_code_dir(tmp_path / "tok", genuine, auto_map)
    (tmp_path / "tok" / "tokenization_fake.py").write_text("y = 2\n")
    (tmp_path / "tok" / "tokenizer_config.json").write_text(json.dumps(
        {"auto_map": {"AutoTokenizer": ["tokenization_fake.FakeTok", None]}}))
    with pytest.raises(RuntimeError, match="tokenization_fake.py"):
        remote_code_kwargs(tok_dir, origin="fake/pinned")


def test_loading_a_tampered_local_checkpoint_of_a_pinned_base_is_refused(tmp_path, monkeypatch):
    import rlcd.policies as policies
    from rlcd.policies import PinnedCode
    monkeypatch.setattr(policies, "PINNED_REMOTE_CODE", {"fake/pinned": PinnedCode("b" * 40, {"modeling_fake.py": "0" * 64})})
    ckpt = _custom_code_dir(tmp_path, "import os\n", {"AutoModelForMaskedLM": "modeling_fake.FakeModel"})
    (tmp_path / "policy.json").write_text(json.dumps({"backend": "hf-mlm", "lora": False, "origin": "fake/pinned"}))
    with pytest.raises(RuntimeError, match="modeling_fake.py"):
        load_policy(ckpt, device="cpu")


# ---------- the LFM2 decoder must never share a process with the patched encoder ----------

def tiny_lfm2():
    from transformers import Lfm2Config, Lfm2ForCausalLM
    torch.manual_seed(0)
    cfg = Lfm2Config(vocab_size=50257, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=2, layer_types=["conv", "full_attention"],
                     max_position_embeddings=256)
    return Lfm2ForCausalLM(cfg).eval()


@pytest.fixture()
def lfm2_dir(tmp_path, gpt2_tok):
    out = tmp_path / "lfm2"
    HFDecoderPolicy(tiny_lfm2(), gpt2_tok).save(str(out), {})
    return str(out)


def test_an_untouched_lfm2_decoder_loads_and_matches_its_own_logits(lfm2_dir):
    policy = load_policy(lfm2_dir, device="cpu").eval()
    q = _questions()[0]
    with torch.no_grad():
        got, _ = policy.decision_logits([q], 512, "cpu")
        want = _full_logit_slice(policy, q)
    assert torch.allclose(got[0, : q.k], want[: q.k], atol=1e-4)


def test_an_lfm2_decoder_is_refused_once_the_pinned_encoder_code_has_loaded(lfm2_dir, monkeypatch):
    import rlcd.policies as policies
    monkeypatch.setattr(policies, "_ENCODER_CODE_LOADED", True)
    with pytest.raises(RuntimeError, match="LFM2.5-Encoder"):
        load_policy(lfm2_dir, device="cpu")


def test_an_lfm2_decoder_is_refused_when_the_lfm2_module_has_been_patched(lfm2_dir, monkeypatch):
    import transformers.models.lfm2.modeling_lfm2 as lfm2

    def other_mask(*args, **kwargs):
        return None

    def other_forward(self, *args, **kwargs):
        return None

    monkeypatch.setattr(lfm2, "create_causal_mask", other_mask)
    with pytest.raises(RuntimeError, match="create_causal_mask"):
        load_policy(lfm2_dir, device="cpu")
    monkeypatch.undo()
    monkeypatch.setattr(lfm2.Lfm2ShortConv, "slow_forward", other_forward)
    with pytest.raises(RuntimeError, match="slow_forward"):
        load_policy(lfm2_dir, device="cpu")


def test_loading_the_pinned_encoder_sets_the_process_flag(mask_tok, monkeypatch):
    import rlcd.policies as policies
    from transformers import AutoModelForMaskedLM
    monkeypatch.setattr(policies, "PINNED_REMOTE_CODE", {"fake/pinned": policies.PinnedCode("b" * 40, {})})
    monkeypatch.setattr(policies, "_ENCODER_CODE_LOADED", False)
    seen = []

    def fake_from_pretrained(name, **kwargs):
        seen.append(kwargs)
        model = tiny_bert(len(mask_tok))
        info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        return (model, info) if kwargs.get("output_loading_info") else model

    monkeypatch.setattr(AutoModelForMaskedLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(policies, "_load_tokenizer", lambda *a, **k: mask_tok)
    load_policy("fake/pinned", device="cpu", backend="hf-mlm")
    assert seen[-1]["trust_remote_code"] is True and seen[-1]["revision"] == "b" * 40
    assert policies._ENCODER_CODE_LOADED is True


# ---------- repos written by transformers 5 ----------

def test_tokenizer_falls_back_when_the_config_names_a_transformers_5_class(tmp_path, gpt2_tok):
    from rlcd.policies import _load_tokenizer
    gpt2_tok.save_pretrained(tmp_path)
    for name in ("vocab.json", "merges.txt"):  # a transformers 5 repo ships tokenizer.json only
        (tmp_path / name).unlink()
    path = tmp_path / "tokenizer_config.json"
    config = json.loads(path.read_text())
    config["tokenizer_class"] = "TokenizersBackend"
    path.write_text(json.dumps(config))
    tok = _load_tokenizer(str(tmp_path))
    text = "User: Context:\nsome text\nAssistant: The answer is"
    assert tok.encode(text, add_special_tokens=False) == gpt2_tok.encode(text, add_special_tokens=False)
    assert letter_token_ids(tok) == letter_token_ids(gpt2_tok)


def test_a_rope_theta_that_4x_would_silently_default_is_refused():
    from types import SimpleNamespace

    from rlcd.policies import _check_rope
    _check_rope(SimpleNamespace())
    _check_rope(SimpleNamespace(rope_theta=1e6, rope_parameters={"rope_theta": 1e6, "rope_type": "default"}))
    with pytest.raises(RuntimeError, match="rope_theta"):
        _check_rope(SimpleNamespace(rope_theta=1e4, rope_parameters={"rope_theta": 1e6}))
