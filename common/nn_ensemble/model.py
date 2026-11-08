"""Standard 64-32-16 GELU MLP with hidden-layer MC Dropout for CALM."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class SimpleMLPRegressor(nn.Module):
    """Small standardized tabular MLP used by NN-UCB and CALM.

    Input imputation and standardization are fitted outside the module on the
    current labelled set. Dropout is deliberately kept in every hidden block
    so inference can obtain predictive dispersion through MC sampling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.20,
    ):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dims = tuple(int(value) for value in hidden_dims)
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(value < 1 for value in hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers: List[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.hidden_dims = hidden_dims
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainConfig:
    """Shared configuration for production MLP fits and OOF calibration."""

    target: str
    seed: int = 42
    epochs: int = 100
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-3
    mlp_hidden_dims: Tuple[int, ...] = (64, 32, 16)
    dropout: float = 0.20
    input_noise: float = 0.0
    warmup_epochs: int = 5
    early_stop_patience: int = 20
    grad_clip_norm: float = 1.0
    huber_delta: float = 1.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
