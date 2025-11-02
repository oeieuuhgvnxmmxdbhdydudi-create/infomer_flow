import torch
import torch.nn as nn

__all__ = ["ConditionEncoder"]

class ConditionEncoder(nn.Module):
    """融合 历史末状态 + 静态 S + 未来外生 -> 每步上下文 C_t，输出 [B,H,d_ctx]"""
    def __init__(self, d_hist: int, d_static: int, d_exo: int, d_ctx: int):
        super().__init__()
        self.exo_proj = nn.Linear(d_exo, d_ctx)
        self.static_proj = nn.Linear(d_static, d_ctx) if d_static > 0 else None
        self.hist_proj = nn.Linear(d_hist, d_ctx)
        in_cat = d_ctx + d_ctx + (d_ctx if d_static > 0 else 0)
        self.out = nn.Linear(in_cat, d_ctx)
        self.norm = nn.LayerNorm(d_ctx)

    def forward(self, C_hist: torch.Tensor, X_fut: torch.Tensor, S=None) -> torch.Tensor:
        B, H, _ = X_fut.shape
        h_last = self.hist_proj(C_hist[:, -1, :]).unsqueeze(1).expand(B, H, -1)
        exo = self.exo_proj(X_fut)
        parts = [exo, h_last]
        if self.static_proj is not None and S is not None:
            s = self.static_proj(S).unsqueeze(1).expand(B, H, -1)
            parts.append(s)
        C = self.out(torch.cat(parts, dim=-1))
        return self.norm(C)
