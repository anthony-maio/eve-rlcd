import json
import random

import pytest
import torch

from rlcd.loop import (GradWindow, JsonlLogger, batch_indices, cosine_lr, plan_steps, save_checkpoint,
                       slice_meta)


def test_cosine_lr_warmup_peak_floor():
    assert cosine_lr(0, 100, 1.0, warmup=10) == 0.0
    assert abs(cosine_lr(10, 100, 1.0, warmup=10) - 1.0) < 1e-9
    assert abs(cosine_lr(100, 100, 1.0, warmup=10) - 0.1) < 1e-9
    assert 0.1 < cosine_lr(55, 100, 1.0, warmup=10) < 1.0


def test_batch_indices_cover_everything_once():
    batches = list(batch_indices(10, 4, random.Random(0)))
    assert [len(b) for b in batches] == [4, 4, 2]
    assert sorted(i for b in batches for i in b) == list(range(10))


def test_jsonl_logger(tmp_path):
    log = JsonlLogger(tmp_path / "log.jsonl")
    log.log(step=1, loss=0.5)
    log.log(step=2, loss=0.25)
    rows = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert rows == [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}]


def test_slice_meta_records_bounds():
    class Row:
        def __init__(self, id):
            self.id = id

    meta = slice_meta([Row("a"), Row("b"), Row("c")], start=7)
    assert meta == {"slice_start": 7, "slice_rows": 3, "slice_first_id": "a", "slice_last_id": "c"}


def test_jsonl_logger_refuses_to_wipe_an_existing_log(tmp_path):
    path = tmp_path / "log.jsonl"
    JsonlLogger(path).log(step=1)
    with pytest.raises(FileExistsError):
        JsonlLogger(path)
    assert json.loads(path.read_text()) == {"step": 1}
    JsonlLogger(path, overwrite=True).log(step=2)
    assert json.loads(path.read_text()) == {"step": 2}


def test_plan_steps_counts_windows_across_epoch_boundaries():
    assert plan_steps(2000, 8, 8, 1) == 32
    assert plan_steps(2000, 8, 8, 2) == 63
    assert plan_steps(2000, 8, 8, 3) == 94
    assert plan_steps(32000, 8, 16, 2) == 500
    assert plan_steps(32000, 8, 8, 1) == 500
    assert plan_steps(3, 8, 16, 1) == 1


def test_grad_window_gives_the_mean_gradient_over_examples():
    torch.manual_seed(0)
    x, y = torch.randn(19, 5), torch.randn(19, 1)

    def fresh():
        torch.manual_seed(1)
        return torch.nn.Linear(5, 1)

    whole = fresh()
    torch.nn.functional.mse_loss(whole(x), y).backward()

    split = fresh()
    window = GradWindow()
    for lo, hi in ((0, 8), (8, 16), (16, 19)):
        window.backward(torch.nn.functional.mse_loss(split(x[lo:hi]), y[lo:hi]), hi - lo)
    assert (window.micro, window.examples) == (3, 19)
    assert window.finish(split.parameters()) == 19
    assert (window.micro, window.examples) == (0, 0)
    for a, b in zip(split.parameters(), whole.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-6)

    # A plain 1/accum scale would under-weight this window: check the naive form really differs.
    naive = fresh()
    for lo, hi in ((0, 8), (8, 16), (16, 19)):
        (torch.nn.functional.mse_loss(naive(x[lo:hi]), y[lo:hi]) / 3).backward()
    assert not torch.allclose(naive.weight.grad, whole.weight.grad, atol=1e-4)


def test_grad_window_finish_on_an_empty_window_raises():
    with pytest.raises(ValueError):
        GradWindow().finish(torch.nn.Linear(2, 1).parameters())


def test_save_checkpoint_roundtrips_through_load_eve(tmp_path):
    from transformers import AutoTokenizer

    from rlcd.compat import eve_config
    from rlcd.eve.modeling_eve import EveMoEForCausalLM
    from rlcd.policy import load_eve

    torch.manual_seed(0)
    cfg = eve_config(vocab_size=50304, n_layer=2, n_embd=32, n_head=2, head_dim=16, block_size=128,
                    num_experts=2, top_k=1, expert_intermediate_size=64, shared_expert_intermediate_size=64)
    model = EveMoEForCausalLM(cfg)
    model.lm_head.weight = model.transformer.wte.weight
    with torch.no_grad():  # stand in for training: move every weight off its init
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    out = tmp_path / "ckpt"
    save_checkpoint(model, AutoTokenizer.from_pretrained("gpt2"), str(out), {"steps": 7, "arm": "rlcd"})

    loaded, tok = load_eve(str(out), device="cpu")
    assert loaded.lm_head.weight.data_ptr() == loaded.transformer.wte.weight.data_ptr()
    assert loaded.config.top_k == 1
    want, got = model.state_dict(), loaded.state_dict()
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name
    assert json.loads((out / "meta.json").read_text()) == {"steps": 7, "arm": "rlcd"}
    assert tok.encode(" A", add_special_tokens=False) == [317]
