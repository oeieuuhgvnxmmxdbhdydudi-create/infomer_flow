import math
import torch
import torch.nn as nn

__all__ = ["ConditionalRealNVP"]


class CouplingNet(nn.Module):
    """
    小型 MLP，把 (x_part, ctx) → (log_s, t)
    原来就有这一层，这里保留，只是在外面会对 log_s 做一个下限处理。
    """
    def __init__(self, d_in: int, d_ctx: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + d_ctx, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 2 * d_out),
        )

    def forward(self, x_part: torch.Tensor, ctx: torch.Tensor):
        # x_part: (B, m), ctx: (B, d_ctx)
        x = torch.cat([x_part, ctx], dim=-1)
        out = self.net(x)
        # 一半是 scale，一半是 translate
        return out.chunk(2, dim=-1)


class ConditionalRealNVP(nn.Module):
    """
    你的现有 Flow 基本结构不动，只加：
    1) scale 下限 clamp
    2) log_prob 返回一个 stats 字典，供训练时做熵正则
    """
    def __init__(self, dim: int, d_ctx: int, hidden_dim: int = 256, n_blocks: int = 6):
        super().__init__()
        self.dim = dim
        self.d_ctx = d_ctx
        self.n_blocks = n_blocks

        # 构造掩码：交替一半一半
        masks = []
        half = dim // 2
        for i in range(n_blocks):
            if i % 2 == 0:
                m = torch.cat([torch.ones(half), torch.zeros(dim - half)])
            else:
                m = torch.cat([torch.zeros(half), torch.ones(dim - half)])
            masks.append(m)
        self.register_buffer("masks", torch.stack(masks, dim=0))  # (n_blocks, dim)

        self.couplings = nn.ModuleList(
            [
                CouplingNet(d_in=dim, d_ctx=d_ctx, d_hidden=hidden_dim, d_out=dim)
                for _ in range(n_blocks)
            ]
        )

    # -------- 核心小工具：对 log_s 做下限 --------
    @staticmethod
    def _clamp_log_scale(log_s: torch.Tensor, min_log_s: float = -6.9) -> torch.Tensor:
        # min_log_s = log(1e-3) ≈ -6.9
        return torch.clamp(log_s, min=min_log_s)

    # -------- 前向：y → z，算 logdet --------
    def forward(self, y: torch.Tensor, ctx: torch.Tensor):
        """
        y: (B, D)
        ctx: (B, d_ctx)
        return: z, logdet, avg_log_scale
        """
        z = y
        logdet = 0.0
        log_scales = []

        for i in range(self.n_blocks):
            m = self.masks[i]  # (D,)
            m = m.to(z.device)
            x1 = z * m  # 保留的一半
            log_s, t = self.couplings[i](x1, ctx)  # (B,D), (B,D)

            # 这里做下限，防止 s→0
            log_s = self._clamp_log_scale(log_s)

            s = torch.exp(log_s)  # (B,D)
            x2 = (z * (1 - m) - t) / s
            z = x1 + x2 * (1 - m)

            # logdet: 只对被变换的那一半求和
            logdet = logdet + (-(log_s) * (1 - m)).sum(dim=-1)
            log_scales.append(log_s.detach())

        # 统计信息：所有层的 log_s 做个平均，给训练阶段当诊断/正则
        if log_scales:
            avg_log_scale = torch.stack([ls.mean() for ls in log_scales]).mean()
        else:
            avg_log_scale = torch.tensor(0.0, device=y.device)

        return z, logdet, avg_log_scale

    # -------- 逆：z → y，用于采样 --------
    def inverse(self, z: torch.Tensor, ctx: torch.Tensor):
        y = z
        for i in reversed(range(self.n_blocks)):
            m = self.masks[i].to(y.device)
            x1 = y * m
            log_s, t = self.couplings[i](x1, ctx)
            log_s = self._clamp_log_scale(log_s)
            s = torch.exp(log_s)
            x2 = y * (1 - m)
            y = x1 + (x2 * s + t) * (1 - m)
        return y

    # -------- log_prob：训练用 --------
    def log_prob(self, y: torch.Tensor, ctx: torch.Tensor):
        z, logdet, avg_log_scale = self.forward(y, ctx)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * z.size(-1) * math.log(2 * math.pi)
        # 返回 logp 和 stats，供 train.py 用
        return log_pz + logdet, {"avg_log_scale": avg_log_scale}

    # -------- sample：评估/生成用 --------
    def sample(self, ctx: torch.Tensor, n_samples: int = 1, temperatures=None):
        """
        ctx: (B, d_ctx)
        return: (n_samples, B, D)
        temperatures:
            - None / float: 全部同一个温度
            - (D,) / (H,) 展开成 (D,) 再用
        """
        B = ctx.size(0)
        D = self.dim
        device = ctx.device
        dtype = ctx.dtype  # 获取输入的数据类型

        if temperatures is None:
            z = torch.randn(n_samples, B, D, device=device, dtype=dtype)
        else:
            # temperatures 可以是标量，也可以是(1,D)
            if isinstance(temperatures, (int, float)):
                z = torch.randn(n_samples, B, D, device=device, dtype=dtype) * float(temperatures)
            else:
                temps = torch.as_tensor(temperatures, device=device, dtype=dtype).view(1, 1, -1)
                z = torch.randn(n_samples, B, D, device=device, dtype=dtype) * temps

        # 批量 inverse
        ctx_exp = ctx.unsqueeze(0).expand(n_samples, B, -1).reshape(-1, ctx.size(-1))
        z_flat = z.reshape(-1, D)
        y_flat = self.inverse(z_flat, ctx_exp)
        y = y_flat.view(n_samples, B, D)
        return y
