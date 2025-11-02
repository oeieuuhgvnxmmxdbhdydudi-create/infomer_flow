"""Plotting helpers for training summaries and evaluation diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_loss_curve(losses: Dict[str, Iterable[float]], path: Path) -> None:
    ensure_parent(path)
    plt.figure(figsize=(6, 4))
    for label, series in losses.items():
        plt.plot(series, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_coverage_bar(coverage: Dict[float, float], path: Path) -> None:
    ensure_parent(path)
    plt.figure(figsize=(5, 4))
    labels = [f"{int(q * 100)}%" for q in coverage.keys()]
    values = list(coverage.values())
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Coverage")
    plt.title("Interval coverage")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_per_horizon_coverage(cov05: np.ndarray, cov09: np.ndarray, path: Path) -> None:
    ensure_parent(path)
    horizons = np.arange(1, len(cov05) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(horizons, cov05, label="q=0.5")
    plt.plot(horizons, cov09, label="q=0.9")
    plt.ylim(0, 1)
    plt.xlabel("Horizon step")
    plt.ylabel("Coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pit(hist: np.ndarray, edges: np.ndarray, path: Path, title: str) -> None:
    ensure_parent(path)
    centers = 0.5 * (edges[1:] + edges[:-1])
    plt.figure(figsize=(6, 4))
    plt.bar(centers, hist, width=edges[1] - edges[0])
    plt.xlabel("PIT")
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_hourly_bar(values: np.ndarray, path: Path, title: str) -> None:
    ensure_parent(path)
    hours = np.arange(24)
    plt.figure(figsize=(7, 4))
    plt.bar(hours, values)
    plt.ylim(0, 1)
    plt.xlabel("Hour of day")
    plt.ylabel("Coverage")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_joint_samples(samples: np.ndarray, path: Path) -> None:
    ensure_parent(path)
    plt.figure(figsize=(6, 4))
    ghi = samples[..., 0].reshape(-1)
    ws = samples[..., 1].reshape(-1)
    plt.scatter(ghi, ws, s=1, alpha=0.25)
    plt.xlabel("GHI")
    plt.ylabel("Wind speed")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
