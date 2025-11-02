import torch
import torch.nn as nn
from .temporal_encoder import InformerLiteEncoder
from .condition_encoder import ConditionEncoder
from .flows.realnvp import ConditionalRealNVP

__all__ = ["JointFlowForecaster"]

class JointFlowForecaster(nn.Module):
    """X_hist -> Temporal -> Condition -> Flow(展平 y)"""
    def __init__(self, d_x:int, d_exo:int, d_static:int, H:int,
                 d_model:int=256, d_ctx:int=256, flow_layers:int=6, flow_hidden:int=512):
        super().__init__()
        self.H = H
        self.D = 2 * H  # 两个目标：GHI, WS
        self.temporal = InformerLiteEncoder(d_in=d_x, d_model=d_model)
        self.cond_enc = ConditionEncoder(d_hist=d_model, d_static=d_static, d_exo=d_exo, d_ctx=d_ctx)
        self.ctx_proj = nn.Linear(d_ctx * H, d_ctx)
        self.flow = ConditionalRealNVP(D=self.D, d_ctx=d_ctx, K=flow_layers, d_hidden=flow_hidden)

    def _build_ctx(self, C_steps):
        B, H, D = C_steps.shape
        return self.ctx_proj(C_steps.reshape(B, H * D))

    def log_prob(self, X_hist, X_fut, y, S=None):
        C_hist = self.temporal(X_hist)
        C_steps = self.cond_enc(C_hist, X_fut, S)
        ctx = self._build_ctx(C_steps)
        y_flat = y.reshape(y.size(0), -1)
        return self.flow.log_prob(y_flat, ctx)

    def sample(self, X_hist, X_fut, S=None, n_samples:int=1):
        C_hist = self.temporal(X_hist)
        C_steps = self.cond_enc(C_hist, X_fut, S)
        ctx = self._build_ctx(C_steps)
        y_flat = self.flow.sample(ctx, n_samples=n_samples)  # [n,B,2H]
        return y_flat.view(n_samples, X_hist.size(0), self.H, 2)
