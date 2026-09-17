"""RL over the sliced-softmax policy. Arms: rlvr (r = c), rlcd (r = c - p_a), oracle (full-label NLL).

All three arms start from the same warmup checkpoint and run through the same loop on the same
rows. The bandit arms (rlvr, rlcd) see only BanditEnv.step, the correctness of the actions they
sampled. Only the oracle arm calls BanditEnv.reveal.

The KL coefficient defaults to 0 and then no reference model exists at all. A KL pull toward the
warmup model would drag the RLCD arm toward a miscalibrated reference and would mask the
overconfidence collapse of the RLVR arm, which is the effect under study. KL stays available as
an ablation (--kl 0.05).
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time
from contextlib import nullcontext

import torch

from rlcd.env import BanditEnv
from rlcd.loop import (GradWindow, JsonlLogger, PeriodicSaver, batch_indices, cosine_lr, plan_steps,
                       slice_meta)
from rlcd.policies import BACKENDS, Policy, load_policy
from rlcd.quick_eval import evaluate, stride_sample
from rlcd.rewards import REWARDS, kl_categorical, policy_gradient_loss, supervised_loss
from rlcd.schema import Question, read_jsonl

ARMS = ("rlvr", "rlcd", "oracle")


def rl_step(policy: Policy, ref_policy: Policy | None, batch: list[Question], idx: torch.Tensor,
            env: BanditEnv, arm: str, group: int, kl_coef: float, aux_coef: float, max_len: int,
            device: str = "cuda"):
    """One micro-batch: the full loss with grad, and float stats. idx stays on CPU. With
    ref_policy=None there is no reference forward and kl is reported as 0.0. stats["loss"] is
    the objective without the router aux term, which is reported on its own as stats["aux"]:
    the policy-gradient loss is small next to the aux term and would be hidden by it. The
    oracle arm also reports stats["nll"]."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    on_cuda = torch.device(device).type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else nullcontext()
    with autocast:
        logits, aux = policy.decision_logits(batch, max_len, device)
        ref_logits = None
        if ref_policy is not None:
            with torch.no_grad():
                ref_logits, _ = ref_policy.decision_logits(batch, max_len, device)
    logp = torch.log_softmax(logits, -1)
    probs = logp.exp()
    stats: dict[str, float] = {}

    if arm == "oracle":
        answers = env.reveal(idx).to(logp.device)
        objective = supervised_loss(logp, answers)
        stats["reward"] = float("nan")
        stats["p_taken"] = float("nan")
        stats["sampled_acc"] = (probs.argmax(-1) == answers).float().mean().item()
        stats["nll"] = objective.item()
    else:
        actions = torch.multinomial(probs.detach(), group, replacement=True)  # (B,G)
        outcomes = env.step(idx, actions)                                     # (B,G), bandit feedback only
        p_a = torch.gather(probs.detach(), 1, actions)
        rewards = REWARDS[arm](outcomes, p_a)
        objective = policy_gradient_loss(logp, actions, rewards)
        stats["reward"] = rewards.mean().item()
        stats["p_taken"] = p_a.mean().item()
        stats["sampled_acc"] = outcomes.mean().item()

    if ref_logits is None:
        kl = torch.zeros((), device=logp.device)
    else:
        kl = kl_categorical(logp, torch.log_softmax(ref_logits, -1)).mean()
    without_aux = objective + kl_coef * kl
    loss = without_aux + aux_coef * aux
    with torch.no_grad():
        # Masked letters have exactly zero mass and a finite log-prob, so they add nothing.
        entropy = -(probs * logp).sum(-1).mean()
    stats.update(loss=without_aux.item(), kl=kl.item(), mean_conf=probs.max(-1).values.mean().item(),
                 policy_entropy=entropy.item(), aux=aux.detach().item())
    return loss, stats


def select_slice(rows: list, start: int, limit: int) -> list:
    """Rows from start to the end of the file, in file order, then the first limit of those
    if limit > 0. The warmup takes rows before start, so the two stages never share a row."""
    rows = rows[start:]
    return rows[:limit] if limit > 0 else rows


def should_stop(eval_accs: list[float], base: float, drop: float = 0.10, patience: int = 3) -> bool:
    """True when the last `patience` evals all sit more than `drop` below the step-0 accuracy."""
    if len(eval_accs) < patience:
        return False
    return all(base - acc > drop for acc in eval_accs[-patience:])


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/warmup")
    ap.add_argument("--backend", choices=("auto",) + BACKENDS, default="auto")
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--start", type=int, default=32000, help="first row of the slice, in file order")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N rows of the slice (0 = all)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--kl", type=float, default=0.0)
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
    qs = select_slice(read_jsonl(args.data), args.start, args.limit)
    assert qs, "empty training slice"
    assert all(q.answer is not None for q in qs), "the environment needs a labeled answer on every row"
    env = BanditEnv(torch.tensor([q.answer for q in qs]))
    eval_qs: list[Question] = []
    if args.eval_data and args.eval_every > 0:
        eval_qs = stride_sample(read_jsonl(args.eval_data), args.eval_n)
        assert all(q.answer is not None for q in eval_qs), "eval needs a labeled answer on every row"

    policy = load_policy(args.init, device="cuda", backend=args.backend, lora=args.lora,
                         grad_checkpointing=args.grad_checkpointing)
    policy.train()
    params = list(policy.trainable_parameters())  # taken before the frozen reference copy exists
    ref_policy = None
    if args.kl > 0:
        # Keep the reference in fp32. Eve's RoPE buffer is complex64 and Module.to(bfloat16)
        # would silently drop its imaginary part.
        ref_policy = copy.deepcopy(policy).eval()
        for p in ref_policy.model.parameters():
            p.requires_grad_(False)
    aux_coef = policy.aux_coef()
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total = plan_steps(len(qs), args.micro, args.accum, args.epochs)
    meta = vars(args) | slice_meta(qs, args.start)
    log.log(arm=args.arm, **slice_meta(qs, args.start), total_steps=total)
    # A periodic checkpoint overwrites --out in place and is marked in_progress in its meta.json.
    saver = PeriodicSaver(0 if args.no_save else args.save_every, total,
                          lambda at: policy.save(args.out, meta | {"steps": at, "stopped": "", "in_progress": True}))

    step, last_logged, last_eval, t0 = 0, 0, -1, time.time()
    grads = GradWindow()
    sums: dict[str, float] = {}
    row: dict = {}
    eval_accs: list[float] = []
    stopped = ""

    def run_eval() -> None:
        nonlocal stopped, last_eval
        result = evaluate(policy, eval_qs, args.max_len)
        log.log(eval_step=step, **result, sec=time.time() - t0)
        last_eval = step
        eval_accs.append(result["eval_acc"])
        if should_stop(eval_accs, base=eval_accs[0]):
            stopped = "eval_acc more than 0.10 below step 0 for three consecutive evals"

    def optimizer_step(epoch: int) -> None:
        nonlocal step, sums, row, last_logged
        n = grads.finish(params)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step + 1, total, args.lr, warmup=max(1, total // 20))
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        # Stats are means over the examples of the window. The oracle arm has no reward or
        # p_taken (NaN), and those keys are left out of its rows.
        row = dict(step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], grad_norm=grad_norm,
                   sec=time.time() - t0, **{key: val / n for key, val in sums.items() if not math.isnan(val)})
        sums = {}
        if step % 10 == 0 or step == 1:
            # Step 1 is the first window, sampled from the untouched warmup policy.
            log.log(**row)
            last_logged = step
        if eval_qs and step % args.eval_every == 0:
            run_eval()
        if not stopped:  # a run the stop rule just ended saves once, below
            saver.after_step(step)

    if eval_qs:
        run_eval()  # step 0: every arm's curve starts from the identical warmup point
    for epoch in range(args.epochs):
        for idx in batch_indices(len(qs), args.micro, rng):
            batch = [qs[i] for i in idx]
            loss, stats = rl_step(policy, ref_policy, batch, torch.tensor(idx), env,
                                  args.arm, args.group, args.kl, aux_coef, args.max_len)
            if not math.isfinite(loss.item()):
                stopped = "non-finite loss"
                break
            grads.backward(loss, len(idx))
            for key, val in stats.items():
                sums[key] = sums.get(key, 0.0) + val * len(idx)
            if grads.micro == args.accum:
                optimizer_step(epoch)
            if stopped:
                break
        if stopped:
            break
    if not stopped and grads.micro:
        # Trailing micro-batches that did not fill an accumulation window still get applied.
        optimizer_step(args.epochs - 1)
    # The run always ends with a train row and an eval, whatever the step count turned out to be.
    if row and last_logged != step:
        log.log(**row)
    if eval_qs and last_eval != step and stopped != "non-finite loss":
        run_eval()

    if stopped:
        log.log(stopped=stopped, at_step=step)
    if args.no_save:
        print("checkpoint skipped (--no-save)")
    elif stopped == "non-finite loss":
        print("checkpoint skipped (non-finite loss)")
    else:
        policy.save(args.out, meta | {"steps": step, "stopped": stopped})
        print("saved", args.out)
    print(f"wall_sec={time.time() - t0:.1f} peak_vram_mb={torch.cuda.max_memory_allocated() // 2**20}")


if __name__ == "__main__":
    main()
