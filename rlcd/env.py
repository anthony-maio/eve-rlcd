"""Bandit environment: reveals only whether the sampled action was correct."""
from __future__ import annotations

import torch


class BanditEnv:
    def __init__(self, answers: torch.Tensor):
        self._answers = answers.clone().long()

    def __len__(self) -> int:
        return int(self._answers.numel())

    def step(self, idx: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """idx (B,), actions (B,G) -> outcomes (B,G) in {0.0, 1.0}."""
        truth = self._answers[idx.cpu()].to(actions.device)
        return (actions == truth[:, None]).float()

    def reveal(self, idx: torch.Tensor) -> torch.Tensor:
        """Full labels. Only the supervised oracle arm may call this."""
        return self._answers[idx.cpu()]
