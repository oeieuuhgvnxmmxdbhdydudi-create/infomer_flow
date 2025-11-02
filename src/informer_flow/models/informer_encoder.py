"""Informer-style encoder blocks."""

from __future__ import annotations

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(10, d_model)

    def forward(self, tfeat: torch.Tensor) -> torch.Tensor:
        return self.proj(tfeat)


class InformerEncoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        d_ctx: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        include_exo: bool,
        include_daymask: bool,
        include_season: bool,
        mean_pred_branch: bool,
    ) -> None:
        super().__init__()
        self.include_exo = include_exo
        self.include_daymask = include_daymask
        self.include_season = include_season

        self.input_proj = nn.LazyLinear(d_model)
        self.time_embed = TimeEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.day_proj = nn.Linear(1, d_model) if include_daymask else None
        self.season_embed = nn.Embedding(4, d_model) if include_season else None

        self.ctx_proj = nn.Linear(2 * d_model, d_ctx)
        self.mean_head = nn.Linear(d_ctx, 2) if mean_pred_branch else None

    def forward(
        self,
        hist_target: torch.Tensor,
        hist_time: torch.Tensor,
        hist_exo: torch.Tensor,
        hist_day: torch.Tensor,
        season_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        components = [hist_target]
        if self.include_exo and hist_exo.size(-1) > 0:
            components.append(hist_exo)
        if self.include_daymask:
            components.append(hist_day.unsqueeze(-1))
        src = torch.cat(components, dim=-1)
        src = self.input_proj(src)
        src = src + self.time_embed(hist_time)

        if self.include_daymask and self.day_proj is not None:
            src = src + self.day_proj(hist_day.unsqueeze(-1))
        if self.include_season and self.season_embed is not None:
            season_emb = self.season_embed(season_id).unsqueeze(1)
            src = src + season_emb

        ctx_seq = self.encoder(src)
        ctx_global = ctx_seq.mean(dim=1)
        ctx_local = ctx_seq[:, -1, :]
        ctx = torch.cat([ctx_global, ctx_local], dim=-1)
        ctx = self.ctx_proj(ctx)

        mean_pred = self.mean_head(ctx) if self.mean_head is not None else None
        return ctx, mean_pred
