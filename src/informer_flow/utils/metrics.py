"""Probability metrics used during evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from scipy.stats import kendalltau, norm


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar) + 1e-6
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var)


def compute_coverages(
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    quantiles: Iterable[float],
) -> Dict[float, float]:
    coverages: Dict[float, float] = {}
    for q in quantiles:
        alpha = (1.0 - q) / 2.0
        lower = mean + std * norm.ppf(alpha)
        upper = mean + std * norm.ppf(1.0 - alpha)
        within = (target >= lower) & (target <= upper)
        coverages[q] = float(np.mean(within))
    return coverages


def kendalls_tau(series_a: np.ndarray, series_b: np.ndarray) -> float:
    tau, _ = kendalltau(series_a, series_b, nan_policy="omit")
    return float(tau)


def interval_coverage(
    samples: np.ndarray,
    targets: np.ndarray,
    alphas: Iterable[float],
) -> Dict[float, float]:
    """
    Compute empirical coverage for central prediction intervals.

    Parameters
    ----------
    samples:
        Array shaped ``[n_samples, batch, horizon, dims]``.
    targets:
        Array shaped ``[batch, horizon, dims]``.
    alphas:
        Iterable of interval widths (e.g. ``0.5`` for the 50% central interval).
    """

    coverages: Dict[float, float] = {}
    for alpha in alphas:
        lower = np.quantile(samples, (1.0 - alpha) / 2.0, axis=0)
        upper = np.quantile(samples, 1.0 - (1.0 - alpha) / 2.0, axis=0)
        inside = (targets >= lower) & (targets <= upper)
        coverages[alpha] = float(np.mean(inside))
    return coverages


def per_horizon_coverage(
    samples: np.ndarray,
    targets: np.ndarray,
    q: float,
) -> np.ndarray:
    lower = np.quantile(samples, (1.0 - q) / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - (1.0 - q) / 2.0, axis=0)
    inside = (targets >= lower) & (targets <= upper)
    return inside.mean(axis=(0, 2))


def pit_hist(
    samples: np.ndarray,
    targets: np.ndarray,
    *,
    target_index: int = 0,
    bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    draws = samples[..., target_index]  # [n, B, H]
    truth = targets[..., target_index][None, ...]  # [1, B, H]
    pit = (draws <= truth).mean(axis=0).reshape(-1)
    hist, edges = np.histogram(pit, bins=bins, range=(0.0, 1.0), density=True)
    return hist, edges, pit


def crps_from_samples(samples: np.ndarray, targets: np.ndarray) -> float:
    s1 = samples
    s2 = np.roll(samples, shift=1, axis=0)
    term1 = np.mean(np.abs(s1 - targets[None, ...]))
    term2 = 0.5 * np.mean(np.abs(s1 - s2))
    return float(term1 - term2)


def hourly_coverage(
    samples: np.ndarray,
    targets: np.ndarray,
    timestamps,
    q: float,
) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.quantile(samples, (1.0 - q) / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - (1.0 - q) / 2.0, axis=0)
    inside = (targets >= lower) & (targets <= upper)
    coverage = inside.mean(axis=2).reshape(-1)  # average across variables

    hours = np.array([ts.hour for ts in timestamps])
    cov_by_hour = np.zeros(24, dtype=np.float32)
    counts = np.zeros(24, dtype=np.int32)
    for hour in range(24):
        mask = hours == hour
        if mask.any():
            cov_by_hour[hour] = float(coverage[mask].mean())
            counts[hour] = int(mask.sum())
    return cov_by_hour, counts
