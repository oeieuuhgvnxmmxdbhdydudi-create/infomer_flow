"""Dataset utilities for conditional flow training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from informer_flow.features.solar import time_fourier_encoding


@dataclass
class TargetStats:
    mean: float
    std: float


class FlowDataset(Dataset):
    """Sliding-window dataset yielding normalised targets and features."""

    target_keys = ("ghi", "ws")

    def __init__(
        self,
        frame,
        *,
        history: int,
        horizon: int,
        extra_exo_cols: Sequence[str] = (),
        use_log1p: bool = True,
    ) -> None:
        if not len(frame):
            raise ValueError("Empty dataframe supplied to FlowDataset.")

        if frame.index.tz is None:
            frame = frame.tz_localize("UTC")
        else:
            frame = frame.tz_convert("UTC")

        self.df = frame.sort_index().copy()
        self.history = int(history)
        self.horizon = int(horizon)
        self.extra_exo_cols = list(extra_exo_cols)
        self.use_log1p = bool(use_log1p)

        # Time features
        t_feats = time_fourier_encoding(self.df.index)
        self.t_cols = [f"time_fourier_{i}" for i in range(t_feats.shape[1])]
        for idx, col in enumerate(self.t_cols):
            self.df[col] = t_feats[:, idx]

        # Day mask (SZA<90 if available, otherwise GHI>0)
        if "SZA" in self.df.columns:
            day_series = (self.df["SZA"].astype(float) < 90.0).astype(np.float32)
        else:
            day_series = (self.df["ghi"].astype(float) > 0.0).astype(np.float32)
        self.df["day_mask_flag"] = day_series

        # Target transforms
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        for key in self.target_keys:
            values = self.df[key].astype(float).to_numpy()
            if self.use_log1p:
                values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
            mean = float(values.mean())
            std = float(values.std())
            if std <= 1e-6:
                raise AssertionError(f"std[{key}] too small -> scaling bug")
            self.means[key] = mean
            self.stds[key] = std
            self.df[f"{key}_z"] = (values - mean) / std
        self.y_cols = [f"{k}_z" for k in self.target_keys]
        self.hist_target_cols = ["ghi_z", "ws_z"]
        self.hist_exo_cols = [f"{c}_z" for c in self.extra_exo_cols]
        self.x_hist_cols = self.hist_target_cols + self.hist_exo_cols + self.t_cols + ["day_mask_flag"]

        # Exogenous z-score
        self.extra_stats: Dict[str, TargetStats] = {}
        for col in self.extra_exo_cols:
            arr = self.df[col].astype(float).to_numpy()
            mean = float(arr.mean())
            std = float(arr.std())
            if std <= 1e-6:
                std = 1.0
            self.extra_stats[col] = TargetStats(mean=mean, std=std)
            self.df[f"{col}_z"] = (arr - mean) / std
        self.x_fut_cols = self.hist_exo_cols + self.t_cols + ["day_mask_flag"]
        self.df["season_id"] = ((self.df.index.month - 1) // 3) % 4

        # Static template (lat/lon + mean GHI)
        lat = self.df["lat"].iloc[0] if "lat" in self.df.columns else 0.0
        lon = self.df["lon"].iloc[0] if "lon" in self.df.columns else 0.0
        self.static_template = np.array(
            [
                float(lat),
                float(lon),
                float(self.df["ghi"].astype(float).mean()),
            ],
            dtype=np.float32,
        )

        self.day_mask_full = self.df["day_mask_flag"].to_numpy(dtype=np.float32)
        self.length = len(self.df) - (self.history + self.horizon) + 1
        if self.length <= 0:
            raise ValueError("Window configuration longer than available data.")
        self._debug_printed = False

    def __len__(self) -> int:
        return self.length

    def inverse_targets(self, arr: np.ndarray) -> np.ndarray:
        """Inverse-transform z-scored targets back to physical scale."""
        out = arr.copy()
        for idx, key in enumerate(self.target_keys):
            out[..., idx] = out[..., idx] * self.stds[key] + self.means[key]
            if self.use_log1p:
                out[..., idx] = np.expm1(out[..., idx])
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        h_slice = slice(idx, idx + self.history)
        f_slice = slice(idx + self.history, idx + self.history + self.horizon)

        hist_df = self.df.iloc[h_slice]
        fut_df = self.df.iloc[f_slice]

        hist_feat = hist_df[self.x_hist_cols].to_numpy(dtype=np.float32)
        fut_feat = fut_df[self.x_fut_cols].to_numpy(dtype=np.float32)
        hist_target = hist_df[self.hist_target_cols].to_numpy(dtype=np.float32)
        if self.hist_exo_cols:
            hist_exo = hist_df[self.hist_exo_cols].to_numpy(dtype=np.float32)
        else:
            hist_exo = np.zeros((self.history, 0), dtype=np.float32)
        hist_time = hist_df[self.t_cols].to_numpy(dtype=np.float32)
        hist_day = hist_df["day_mask_flag"].to_numpy(dtype=np.float32)
        fut_targets_z = fut_df[self.y_cols].to_numpy(dtype=np.float32)
        fut_targets_phys = self.inverse_targets(fut_targets_z.astype(np.float64)).astype(np.float32)

        day_mask = self.day_mask_full[f_slice]
        season_id = int(fut_df["season_id"].iloc[-1])

        if not self._debug_printed:
            y_dbg = self.df[self.y_cols].dropna()
            print(
                "[DATASET] y mean/std:",
                list(np.round(y_dbg.mean().to_numpy(), 4)),
                list(np.round(y_dbg.std().to_numpy(), 4)),
                f"use_log1p={self.use_log1p}",
            )
            self._debug_printed = True

        sample = {
            "X_hist": torch.from_numpy(hist_feat),
            "X_fut": torch.from_numpy(fut_feat),
            "y": torch.from_numpy(fut_targets_z),
            "y_phys": torch.from_numpy(fut_targets_phys),
            "S": torch.from_numpy(self.static_template.copy()),
            "day_mask": torch.from_numpy(day_mask.astype(np.float32)),
            "hist_target": torch.from_numpy(hist_target),
            "hist_exo": torch.from_numpy(hist_exo),
            "hist_time": torch.from_numpy(hist_time),
            "hist_day": torch.from_numpy(hist_day.astype(np.float32)),
            "season_id": torch.tensor(season_id, dtype=torch.long),
        }
        return sample


def build_dataloader(
    frame,
    *,
    L: int,
    H: int,
    batch_size: int,
    extra_exo_cols: Iterable[str] = (),
    shuffle: bool = True,
    use_log1p: bool = True,
) -> DataLoader:
    dataset = FlowDataset(
        frame,
        history=L,
        horizon=H,
        extra_exo_cols=tuple(extra_exo_cols),
        use_log1p=use_log1p,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
