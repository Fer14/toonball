"""Stand-in for timm's DropPath (stochastic depth); only active in training."""

import torch
from torch import nn


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand((x.shape[0],) + (1,) * (x.dim() - 1), device=x.device) > self.p)
        return x * keep / (1 - self.p)
