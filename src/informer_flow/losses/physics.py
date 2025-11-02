"""Physics-inspired auxiliary losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def physics_penalty(
    y: torch.Tensor,
    day_mask: torch.Tensor,
    w_night: float = 0.01,
    w_wsneg: float = 0.01,
    clrski: torch.Tensor | None = None,
    w_clr: float = 0.0,
    y_phys: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Penalise violations of simple physical constraints.

    Parameters
    ----------
    y:
        Targets in model space (kept for backwards compatibility).
    day_mask:
        Binary mask with value 1 for daytime, 0 for night.
    y_phys:
        Optional targets converted back to physical scale. When provided it is used for
        the constraint checks.
    """

    target = y_phys if y_phys is not None else y
    ghi = target[..., 0]
    ws = target[..., 1]

    night = 1.0 - day_mask
    loss_night = (night * (ghi ** 2)).mean()
    loss_wsneg = F.relu(-ws).mean()

    loss_clr = 0.0
    if clrski is not None:
        loss_clr = F.relu(ghi - 1.2 * clrski).mean()

    return w_night * loss_night + w_wsneg * loss_wsneg + w_clr * loss_clr
