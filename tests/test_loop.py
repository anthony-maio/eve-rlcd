import json
import random

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr, slice_meta, window_size


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


def test_window_size_full_and_trailing_windows():
    # 10 micro-batches with accum 4 give windows of 4, 4, 2
    assert [window_size(i, 10, 4) for i in range(10)] == [4, 4, 4, 4, 4, 4, 4, 4, 2, 2]
    assert [window_size(i, 8, 4) for i in range(8)] == [4] * 8
    assert [window_size(i, 3, 16) for i in range(3)] == [3, 3, 3]


def test_slice_meta_records_bounds():
    class Row:
        def __init__(self, id):
            self.id = id

    meta = slice_meta([Row("a"), Row("b"), Row("c")], start=7)
    assert meta == {"slice_start": 7, "slice_rows": 3, "slice_first_id": "a", "slice_last_id": "c"}
