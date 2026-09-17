"""Warmup SFT: teach the letter format on a small labeled slice."""
from __future__ import annotations

import argparse
import random
import time

import torch

from rlcd.loop import (GradWindow, JsonlLogger, PeriodicSaver, batch_indices, cosine_lr, plan_steps,
                       slice_meta)
from rlcd.policies import BACKENDS, load_policy
from rlcd.policy import EVE_ID
from rlcd.quick_eval import evaluate, stride_sample
from rlcd.rewards import supervised_loss
from rlcd.schema import Question, read_jsonl


def select_slice(rows: list, start: int, n: int) -> list:
    """Rows [start, start + n) in file order. The training file is shuffled at build time, so
    slices by file order are random samples, and the SFT and RL slices stay disjoint."""
    return rows[start:start + n]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--init", default=EVE_ID)
    ap.add_argument("--backend", choices=("auto",) + BACKENDS, default="auto")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--grad-checkpointing", action="store_true")
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
    ap.add_argument("--save-every", type=int, default=0,
                    help="also write the checkpoint to --out every N optimizer steps (0 = only at the end)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing train_log.jsonl in --out")
    return ap


def main():
    args = build_parser().parse_args()

    # Open the log first: it refuses to wipe a finished curve, and that should fail fast.
    log = JsonlLogger(f"{args.out}/train_log.jsonl", overwrite=args.overwrite)
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

    policy = load_policy(args.init, device="cuda", backend=args.backend, lora=args.lora,
                         grad_checkpointing=args.grad_checkpointing)
    policy.train()
    aux_coef = policy.aux_coef()
    params = list(policy.trainable_parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total = plan_steps(len(qs), args.micro, args.accum, args.epochs)
    meta = vars(args) | slice_meta(qs, args.start)
    log.log(**slice_meta(qs, args.start), total_steps=total)
    # A periodic checkpoint overwrites --out in place and is marked in_progress in its meta.json.
    saver = PeriodicSaver(0 if args.no_save else args.save_every, total,
                          lambda at: policy.save(args.out, meta | {"steps": at, "in_progress": True}))

    step, last_logged, last_eval, t0 = 0, 0, 0, time.time()
    grads = GradWindow()
    sums = {"nll": 0.0, "aux": 0.0, "hits": 0.0, "conf": 0.0}
    row: dict = {}

    def run_eval() -> None:
        nonlocal last_eval
        log.log(eval_step=step, **evaluate(policy, eval_qs, args.max_len), sec=time.time() - t0)
        last_eval = step

    def optimizer_step(epoch: int) -> None:
        nonlocal step, row, last_logged
        n = grads.finish(params)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step + 1, total, args.lr, warmup=max(1, total // 20))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        # Means over the examples of the window. loss is the objective without the router aux
        # term; for SFT that is the NLL.
        row = dict(step=step, epoch=epoch, loss=sums["nll"] / n, nll=sums["nll"] / n, aux=sums["aux"] / n,
                   acc=sums["hits"] / n, mean_conf=sums["conf"] / n, grad_norm=grad_norm,
                   lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
        for key in sums:
            sums[key] = 0.0
        if step % 5 == 0:
            log.log(**row)
            last_logged = step
        if eval_qs and step % args.eval_every == 0:
            run_eval()
        saver.after_step(step)

    for epoch in range(args.epochs):
        for idx in batch_indices(len(qs), args.micro, rng):
            batch = [qs[i] for i in idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = policy.decision_logits(batch, args.max_len, "cuda")
            logp = torch.log_softmax(logits, -1)
            nll = supervised_loss(logp, answers[idx])
            grads.backward(nll + aux_coef * aux, len(idx))
            with torch.no_grad():
                sums["nll"] += nll.item() * len(idx)
                sums["aux"] += aux.detach().item() * len(idx)
                sums["hits"] += (logp.argmax(-1).cpu() == answers[idx]).float().sum().item()
                sums["conf"] += logp.exp().max(-1).values.sum().item()
            if grads.micro == args.accum:
                optimizer_step(epoch)
    if grads.micro:
        # Trailing micro-batches that did not fill an accumulation window still get applied.
        optimizer_step(args.epochs - 1)
    # The run always ends with a train row and an eval, whatever the step count turned out to be.
    if row and last_logged != step:
        log.log(**row)
    if eval_qs and last_eval != step:
        run_eval()

    if args.no_save:
        print("checkpoint skipped (--no-save)")
    else:
        policy.save(args.out, meta | {"steps": step})
        print("saved", args.out)
    print(f"wall_sec={time.time() - t0:.1f} peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")


if __name__ == "__main__":
    main()
