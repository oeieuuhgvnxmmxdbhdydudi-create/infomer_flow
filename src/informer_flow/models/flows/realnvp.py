import math
import torch
import torch.nn as nn

__all__ = ["ConditionalRealNVP"]

class CouplingNet(nn.Module):
    def __init__(self, d_in: int, d_ctx: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + d_ctx, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 2 * d_out),
        )

    def forward(self, x_part, ctx):
        st = self.net(torch.cat([x_part, ctx], dim=-1))
        s, t = torch.chunk(st, 2, dim=-1)
        s = torch.tanh(s) * 1.5  # 限幅稳定
        return s, t

class RealNVPBlock(nn.Module):
    def __init__(self, D: int, d_ctx: int, d_hidden: int = 512, mask_even: bool = True):
        super().__init__()
        idx = torch.arange(D)
        m = (idx % 2 == 0).float() if mask_even else (idx % 2 == 1).float()
        self.register_buffer("m", m)
        D_a = int(m.sum().item()); D_b = D - D_a
        self.net = CouplingNet(D_a, d_ctx, d_hidden, D_b)

    def forward(self, y, ctx, reverse: bool = False):
        m = self.m[None, :]
        y_a = y[:, m[0] == 1.0]
        s, t = self.net(y_a, ctx)
        if not reverse:
            y_b = y[:, m[0] == 0.0]
            y_b = y_b * torch.exp(s) + t
            out = torch.zeros_like(y)
            out[:, m[0] == 1.0] = y_a
            out[:, m[0] == 0.0] = y_b
            logdet = s.sum(dim=-1)
        else:
            y_b = y[:, m[0] == 0.0]
            y_b = (y_b - t) * torch.exp(-s)
            out = torch.zeros_like(y)
            out[:, m[0] == 1.0] = y_a
            out[:, m[0] == 0.0] = y_b
            logdet = (-s).sum(dim=-1)
        return out, logdet

class ConditionalRealNVP(nn.Module):
    def __init__(self, D: int, d_ctx: int, K: int = 6, d_hidden: int = 512):
        super().__init__()
        self.blocks = nn.ModuleList([RealNVPBlock(D, d_ctx, d_hidden, mask_even=(k % 2 == 0)) for k in range(K)])

    def forward(self, y, ctx):
        z = y
        logdet = torch.zeros(y.size(0), device=y.device)
        for blk in self.blocks:
            z, ld = blk(z, ctx, reverse=False)
            logdet += ld
        return z, logdet

    def inverse(self, z, ctx):
        y = z
        for blk in reversed(self.blocks):
            y, _ = blk(y, ctx, reverse=True)
        return y

    def log_prob(self, y, ctx):
        z, logdet = self.forward(y, ctx)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * z.size(-1) * math.log(2 * math.pi)
        return log_pz + logdet

    def sample(self, ctx, n_samples: int = 1):
        B = ctx.size(0); D = self.blocks[0].m.numel()
        outs = []
        for _ in range(n_samples):
            z = torch.randn(B, D, device=ctx.device)
            outs.append(self.inverse(z, ctx))
        return torch.stack(outs, dim=0)  # [n,B,D]
