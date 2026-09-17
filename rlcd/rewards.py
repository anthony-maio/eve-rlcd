"""Reward functions and the losses built on them."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reward_rlvr(outcomes: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """Verifiable outcome only."""
    return outcomes


def reward_rlcd(outcomes: torch.Tensor, p_a: torch.Tensor) -> torch.Tensor:
    """Outcome minus the stated probability of the taken action.
    REINFORCE with this reward is an unbiased estimator of half the Brier
    score gradient using only bandit feedback."""
    return outcomes - p_a


REWARDS = {"rlvr": reward_rlvr, "rlcd": reward_rlcd}


def gather_logp(logp: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.gather(logp, 1, actions)


def policy_gradient_loss(logp: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    """logp (B,K), actions (B,G), rewards (B,G). Leave-one-out baseline over the group."""
    _, group = rewards.shape
    if group > 1:
        baseline = (rewards.sum(1, keepdim=True) - rewards) / (group - 1)
    else:
        baseline = torch.zeros_like(rewards)
    advantage = (rewards - baseline).detach()
    return -(advantage * gather_logp(logp, actions)).mean()


def kl_categorical(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """KL(p || p_ref) per row. Masked positions carry zero mass in both, so the
    product is exactly zero there as long as the logits are finite (NEG, not -inf)."""
    p = logp.exp()
    return (p * (logp - logp_ref)).sum(-1)


def supervised_loss(logp: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(logp, answers.to(logp.device))
