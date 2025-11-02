"""Simplified temporal encoder used for the demo training loop."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    """A light-weight GRU encoder that summarises the history window."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [batch, history, features]
        _, hidden = self.gru(history)
        return hidden[-1]
