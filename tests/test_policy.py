import pytest
import torch

from rlcd.compat import eve_config
from rlcd.eve.modeling_eve import EveMoEForCausalLM
from rlcd.policy import PAD_ID, decision_logits, encode_batch, eve_hidden
from rlcd.schema import NEG


class FakeTok:
    """Maps each character to a token id; enough to test padding logic."""
    def encode(self, s, add_special_tokens=False):
        return [ord(c) % 50000 + 1 for c in s]


def tiny_model() -> EveMoEForCausalLM:
    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16,
                    block_size=128, num_experts=2, top_k=1,
                    expert_intermediate_size=64, shared_expert_intermediate_size=64)
    return EveMoEForCausalLM(cfg).eval()


def test_encode_batch_right_pads_and_tracks_last_index():
    ids, last = encode_batch(FakeTok(), ["abc", "abcdef"], max_len=512)
    assert ids.shape == (2, 6)
    assert ids[0, 3:].tolist() == [PAD_ID] * 3
    assert last.tolist() == [2, 5]


def test_encode_batch_truncates_from_the_left():
    ids, last = encode_batch(FakeTok(), ["abcdefgh"], max_len=4)
    assert ids.shape == (1, 4)
    assert ids[0].tolist() == FakeTok().encode("efgh")
    assert last.tolist() == [3]


def test_padded_batch_matches_single_rows():
    model = tiny_model()
    tok = FakeTok()
    letter_ids = list(range(100, 126))
    prompts = ["short one", "a considerably longer prompt here"]
    ids, last = encode_batch(tok, prompts)
    k = torch.tensor([3, 5])
    with torch.no_grad():
        batched, _ = decision_logits(model, ids, last, letter_ids, k)
        for i, p in enumerate(prompts):
            ids1, last1 = encode_batch(tok, [p])
            single, _ = decision_logits(model, ids1, last1, letter_ids, k[i:i + 1])
            assert torch.allclose(batched[i, : k[i]], single[0, : k[i]], atol=1e-4)


def test_decision_logits_masks_beyond_k():
    model = tiny_model()
    ids, last = encode_batch(FakeTok(), ["xyz"])
    k = torch.tensor([2])
    with torch.no_grad():
        logits, aux = decision_logits(model, ids, last, list(range(100, 126)), k)
    assert logits.shape == (1, 26)
    assert torch.all(logits[0, 2:] == NEG)
    assert torch.isfinite(logits).all()
    assert torch.softmax(logits, -1)[0, 2:].sum() == 0
    assert aux.ndim == 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_decision_logits_are_fp32_under_bf16_autocast():
    model = tiny_model().to("cuda")
    letter_ids = list(range(100, 126))
    ids, last = encode_batch(FakeTok(), ["short one", "a considerably longer prompt here"], device="cuda")
    k = torch.tensor([3, 5], device="cuda")
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = decision_logits(model, ids, last, letter_ids, k)
            hidden, _ = eve_hidden(model, ids)
        rows = hidden[torch.arange(hidden.size(0), device="cuda"), last]
        weight = model.lm_head.weight[torch.as_tensor(letter_ids, device="cuda")]
        ref = rows.float() @ weight.float().t()
    assert ref.dtype == torch.float32
    assert logits.dtype == torch.float32
    assert aux.dtype == torch.float32
    for i in range(2):
        assert torch.all(logits[i, k[i]:] == NEG)
        assert torch.allclose(logits[i, : k[i]], ref[i, : k[i]], atol=1e-2)


def _assert_same_model(loaded, model):
    assert loaded.config.top_k == 1
    assert loaded.transformer.h[0].mlp.top_k == 1
    assert loaded.lm_head.weight.data_ptr() == loaded.transformer.wte.weight.data_ptr()
    assert all(p.dtype == torch.float32 for p in loaded.parameters())
    want = model.state_dict()
    got = loaded.state_dict()
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name


def test_load_eve_roundtrips_a_save_pretrained_checkpoint(tmp_path):
    from transformers import AutoTokenizer

    from rlcd.policy import load_eve
    model = tiny_model()
    model.save_pretrained(tmp_path)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(tmp_path)
    loaded, tok = load_eve(str(tmp_path), device="cpu")
    _assert_same_model(loaded, model)
    assert tok.encode(" A", add_special_tokens=False) == [317]


def test_load_eve_reads_a_hub_style_checkpoint(tmp_path):
    """The hub file stores only lm_head.weight and has no 'format' metadata, which
    transformers' from_pretrained refuses. load_eve must still read it."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from rlcd.policy import load_eve
    model = tiny_model()
    sd = {k: v.clone().contiguous() for k, v in model.state_dict().items()
          if k != "transformer.wte.weight"}
    assert "lm_head.weight" in sd
    save_file(sd, str(tmp_path / "model.safetensors"),
              metadata={"transformer.wte.weight": "lm_head.weight"})
    model.config.save_pretrained(tmp_path)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(tmp_path)
    loaded, _ = load_eve(str(tmp_path), device="cpu")
    _assert_same_model(loaded, model)


def test_load_eve_rejects_a_checkpoint_with_missing_weights(tmp_path):
    import pytest
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from rlcd.policy import load_eve
    model = tiny_model()
    sd = {k: v.clone().contiguous() for k, v in model.state_dict().items()
          if k != "transformer.wte.weight" and "router" not in k}
    save_file(sd, str(tmp_path / "model.safetensors"))
    model.config.save_pretrained(tmp_path)
    AutoTokenizer.from_pretrained("gpt2").save_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="router"):
        load_eve(str(tmp_path), device="cpu")
