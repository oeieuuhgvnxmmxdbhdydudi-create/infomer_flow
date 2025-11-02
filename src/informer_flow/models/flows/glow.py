"""Placeholder for future Glow-based flow."""

from __future__ import annotations

from torch import nn


class ConditionalGlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - placeholder
        raise NotImplementedError("Glow flow is not implemented in this minimal demo.")
