"""Warmup SFT: teach the letter format on a small labeled slice."""
from __future__ import annotations

import argparse
import random
import time

import torch

from rlcd.loop import JsonlLogger, batch_indices, cosine_lr, save_checkpoint
from rlcd.policy import EVE_ID, decision_logits, load_eve, questions_to_batch
from rlcd.rewards import supervised_loss
from rlcd.schema import Question, letter_token_ids, read_jsonl


def evaluate(model, tok, letters, questions: list[Question], max_len: int) -> dict:
    """Accuracy, NLL of the true answer, confidence, and last-option rate, overall and
    per source. Runs in eval mode without grad and restores the previous mode."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    pred, conf, nll = [], [], []
    with torch.no_grad():
        for i in range(0, len(questions), 32):
            batch = questions[i:i + 32]
            ids, last, k = questions_to_batch(tok, batch, max_len, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = decision_logits(model, ids, last, letters, k)
            logp = torch.log_softmax(logits, -1)
            truth = torch.tensor([q.answer for q in batch], device=device)
            nll += (-logp.gather(1, truth[:, None])[:, 0]).cpu().tolist()
            pred += logp.argmax(-1).cpu().tolist()
            conf += logp.exp().max(-1).values.cpu().tolist()
    model.train(was_training)
    n = len(questions)
    hits = [float(a == q.answer) for a, q in zip(pred, questions)]
    out = {"eval_acc": sum(hits) / n, "eval_nll": sum(nll) / n, "eval_conf": sum(conf) / n,
           "eval_pred_last": sum(float(a == q.k - 1) for a, q in zip(pred, questions)) / n}
    for src in sorted({q.source for q in questions}):
        rows = [h for h, q in zip(hits, questions) if q.source == src]
        out[f"eval_acc_{src}"] = sum(rows) / len(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--init", default=EVE_ID)
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-data", default="")
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    qs = read_jsonl(args.data)
    rng.shuffle(qs)
    qs = qs[: args.n]
    assert all(q.answer is not None for q in qs), "SFT needs a labeled answer on every row"
    answers = torch.tensor([q.answer for q in qs])
    eval_qs: list[Question] = []
    if args.eval_data and args.eval_every > 0:
        # Stride sample: eval files are sorted by source, so a head slice would miss most of them.
        eval_qs = read_jsonl(args.eval_data)
        eval_qs = eval_qs[:: max(1, len(eval_qs) // args.eval_n)][: args.eval_n]
        assert all(q.answer is not None for q in eval_qs), "eval needs a labeled answer on every row"

    model, tok = load_eve(args.init, device="cuda")
    model.train()
    letters = letter_token_ids(tok)
    aux_coef = model.config.router_aux_loss_coef
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    steps_per_epoch = (len(qs) + args.micro * args.accum - 1) // (args.micro * args.accum)
    total = steps_per_epoch * args.epochs
    log = JsonlLogger(f"{args.out}/train_log.jsonl")

    step, t0 = 0, time.time()
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
            (loss / args.accum).backward()
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

    save_checkpoint(model, tok, args.out, vars(args) | {"steps": step})
    print("saved", args.out)
    print(f"wall_sec={time.time() - t0:.1f} peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")


if __name__ == "__main__":
    main()
