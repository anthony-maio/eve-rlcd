"""Warmup SFT: teach the letter format on a small labeled slice."""
from __future__ import annotations

import argparse
import random
import time

import torch

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr, save_checkpoint, slice_meta, window_size
from rlcd.policy import EVE_ID, decision_logits, load_eve, questions_to_batch
from rlcd.quick_eval import evaluate, stride_sample
from rlcd.rewards import supervised_loss
from rlcd.schema import Question, letter_token_ids, read_jsonl


def select_slice(rows: list, start: int, n: int) -> list:
    """Rows [start, start + n) in file order. The training file is shuffled at build time, so
    slices by file order are random samples, and the SFT and RL slices stay disjoint."""
    return rows[start:start + n]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--init", default=EVE_ID)
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--start", type=int, default=0, help="first row of the slice, in file order")
    ap.add_argument("--n", type=int, default=32000, help="number of rows in the slice")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--no-save", action="store_true", help="skip the checkpoint; the log is still written")
    return ap


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    qs = select_slice(read_jsonl(args.data), args.start, args.n)
    assert qs, "empty training slice"
    assert all(q.answer is not None for q in qs), "SFT needs a labeled answer on every row"
    answers = torch.tensor([q.answer for q in qs])
    eval_qs: list[Question] = []
    if args.eval_data and args.eval_every > 0:
        eval_qs = stride_sample(read_jsonl(args.eval_data), args.eval_n)
        assert all(q.answer is not None for q in eval_qs), "eval needs a labeled answer on every row"

    model, tok = load_eve(args.init, device="cuda")
    model.train()
    letters = letter_token_ids(tok)
    aux_coef = model.config.router_aux_loss_coef
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    # Accumulation windows run across epoch boundaries, so count micro-batches over the whole run.
    n_micro = args.epochs * ((len(qs) + args.micro - 1) // args.micro)
    total = (n_micro + args.accum - 1) // args.accum
    meta = vars(args) | slice_meta(qs, args.start)
    log = JsonlLogger(f"{args.out}/train_log.jsonl")
    log.log(**slice_meta(qs, args.start), total_steps=total)

    step, micro_index, t0 = 0, 0, time.time()
    window = {"n": 0, "loss": 0.0, "aux": 0.0, "hits": 0.0, "conf": 0.0, "micro": 0}

    def optimizer_step(epoch: int) -> None:
        nonlocal step
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step + 1, total, args.lr, warmup=max(1, total // 20))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == total:
            log.log(step=step, epoch=epoch, loss=window["loss"] / window["micro"],
                    aux=window["aux"] / window["micro"], acc=window["hits"] / window["n"],
                    mean_conf=window["conf"] / window["n"], lr=opt.param_groups[0]["lr"],
                    sec=time.time() - t0)
        for key in window:
            window[key] = 0 if key in ("n", "micro") else 0.0
        if eval_qs and (step % args.eval_every == 0 or step == total):
            log.log(eval_step=step, **evaluate(model, tok, letters, eval_qs, args.max_len),
                    sec=time.time() - t0)

    for epoch in range(args.epochs):
        for idx in batch_indices(len(qs), args.micro, rng):
            batch = [qs[i] for i in idx]
            ids, last, k = questions_to_batch(tok, batch, args.max_len, "cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = decision_logits(model, ids, last, letters, k)
            logp = torch.log_softmax(logits, -1)
            loss = supervised_loss(logp, answers[idx]) + aux_coef * aux
            (loss / window_size(micro_index, n_micro, args.accum)).backward()
            micro_index += 1
            with torch.no_grad():
                window["n"] += len(idx)
                window["micro"] += 1
                window["loss"] += loss.detach().item()
                window["aux"] += aux.detach().item()
                window["hits"] += (logp.argmax(-1).cpu() == answers[idx]).float().sum().item()
                window["conf"] += logp.exp().max(-1).values.sum().item()
            if window["micro"] == args.accum:
                optimizer_step(epoch)
    if window["micro"]:
        # Trailing micro-batches that did not fill an accumulation window still get applied.
        optimizer_step(args.epochs - 1)

    if args.no_save:
        print("checkpoint skipped (--no-save)")
    else:
        save_checkpoint(model, tok, args.out, meta | {"steps": step})
        print("saved", args.out)
    print(f"wall_sec={time.time() - t0:.1f} peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")


if __name__ == "__main__":
    main()
