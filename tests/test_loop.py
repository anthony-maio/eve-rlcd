import json
import random

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr


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
