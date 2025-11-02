from __future__ import annotations

import torch
from torch import nn
import numpy as np

from .temporal_encoder import TemporalEncoder
from .condition_encoder import ConditionEncoder
from .flows.realnvp import ConditionalRealNVP

__all__ = ["JointFlowForecaster"]


class JointFlowForecaster(nn.Module):
    """Temporal encoder + conditional flow forecaster."""

    def __init__(
        self,
        d_x: int,
        d_exo: int,
        d_static: int,
        H: int,
        d_model: int = 256,
        d_ctx: int = 256,
        flow_layers: int = 6,
        flow_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.horizon = H
        self.output_dim = 2 * H  # two targets (ghi, ws)

        self.temporal = TemporalEncoder(input_dim=d_x, hidden_dim=d_model, num_layers=1)
        self.cond_enc = ConditionEncoder(
            d_hist=d_model,
            d_static=d_static,
            d_exo=d_exo,
            d_ctx=d_ctx,
        )
        self.flow = ConditionalRealNVP(
            dim=self.output_dim,
            d_ctx=d_ctx,
            hidden_dim=flow_hidden,
            n_blocks=flow_layers,
        )

    @staticmethod
    def _build_ctx(cond_steps: torch.Tensor) -> torch.Tensor:
        return cond_steps[:, -1, :] if cond_steps.dim() == 3 else cond_steps

    def log_prob(self, X_hist, X_fut, y, S=None):
        hist_ctx = self.temporal(X_hist)
        cond_steps = self.cond_enc(hist_ctx.unsqueeze(1), X_fut, S)
        ctx = self._build_ctx(cond_steps)
        y_flat = y.reshape(y.size(0), -1)
        return self.flow.log_prob(y_flat, ctx)

    def sample(self, X_hist, X_fut, S=None, n_samples: int = 1, temperatures=None):
        # 确保所有输入都是 float32
        X_hist = X_hist.float()
        X_fut = X_fut.float()
        if S is not None:
            S = S.float()
        
        hist_ctx = self.temporal(X_hist)
        cond_steps = self.cond_enc(hist_ctx.unsqueeze(1), X_fut, S)
        ctx = self._build_ctx(cond_steps)
        
        if temperatures is not None:
            if isinstance(temperatures, np.ndarray):
                temperatures = torch.from_numpy(temperatures).float()
            elif isinstance(temperatures, torch.Tensor):
                temperatures = temperatures.float()
        
        y_flat = self.flow.sample(ctx, n_samples=n_samples, temperatures=temperatures)
        n, B, D = y_flat.shape
        return y_flat.view(n, B, self.horizon, 2)
