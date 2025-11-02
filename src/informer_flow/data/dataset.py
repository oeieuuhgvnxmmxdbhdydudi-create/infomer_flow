import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from ..features.solar import daytime_mask_from_sun, time_fourier_features

__all__ = ["NasaSlidingWindow", "build_dataloader"]

class NasaSlidingWindow(Dataset):
    def __init__(self, df, L:int, H:int, extra_exo_cols:Tuple[str,...]=()):
        self.df = df.copy()
        self.L = L; self.H = H
        self.extra_exo_cols = extra_exo_cols

        t_feats, t_cols = time_fourier_features(self.df.index)
        for i, c in enumerate(t_cols):
            self.df[c] = t_feats[:, i]

        lat = float(self.df["lat"].iloc[0]); lon = float(self.df["lon"].iloc[0])
        self.df["day_mask"] = daytime_mask_from_sun(self.df.index, lat, lon)

        self.means = {}; self.stds = {}
        for col in ["ghi","ws"] + list(extra_exo_cols):
            m, s = self.df[col].mean(), self.df[col].std()
            self.means[col], self.stds[col] = float(m), float(s if s>1e-6 else 1.0)
            self.df[col+"_z"] = (self.df[col] - m)/self.stds[col]

        self.t_cols = t_cols
        self.x_hist_cols = ["ghi_z","ws_z"] + self.t_cols
        self.x_fut_cols  = list(extra_exo_cols) + self.t_cols
        self.y_cols = ["ghi_z","ws_z"]

        self.N = len(self.df)
        self.max_start = self.N - (L + H)
        if self.max_start < 1:
            raise ValueError("数据长度不足以滑窗.")

        self.S = np.array([[lat, lon, 1.0]], dtype=np.float32)

    def __len__(self):
        return self.max_start

    def __getitem__(self, i: int):
        sl_hist = slice(i, i + self.L)
        sl_fut  = slice(i + self.L, i + self.L + self.H)
        X_hist = self.df.iloc[sl_hist][self.x_hist_cols].values.astype(np.float32)
        X_fut  = self.df.iloc[sl_fut ][self.x_fut_cols].values.astype(np.float32)
        y      = self.df.iloc[sl_fut ][self.y_cols].values.astype(np.float32)
        day_mk = self.df.iloc[sl_fut ]["day_mask"].values.astype(np.float32)
        return dict(
            X_hist=torch.from_numpy(X_hist),
            X_fut=torch.from_numpy(X_fut),
            y=torch.from_numpy(y),
            S=torch.from_numpy(self.S).squeeze(0),
            day_mask=torch.from_numpy(day_mk),
        )

def build_dataloader(df, L:int, H:int, batch_size:int=16, shuffle:bool=True, extra_exo_cols=()):
    ds = NasaSlidingWindow(df, L=L, H=H, extra_exo_cols=extra_exo_cols)
    def _collate(batch):
        X_hist = torch.stack([b["X_hist"] for b in batch], 0).float()
        X_fut  = torch.stack([b["X_fut"]  for b in batch], 0).float()
        y      = torch.stack([b["y"]      for b in batch], 0).float()
        day    = torch.stack([b["day_mask"] for b in batch], 0).float()
        S      = torch.stack([b["S"] for b in batch], 0).float()
        return dict(X_hist=X_hist, X_fut=X_fut, y=y, day_mask=day, S=S)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate)
