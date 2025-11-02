import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from informer_flow.data.dataset import build_dataloader
from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.models.joint_forecaster import JointFlowForecaster
from informer_flow.utils.metrics import (
    crps_from_samples,
    hourly_coverage,
    interval_coverage,
    per_horizon_coverage,
    pit_hist,
)
from informer_flow.utils.plotting import (
    plot_coverage_bar,
    plot_hourly_bar,
    plot_per_horizon_coverage,
    plot_pit,
)
from informer_flow.utils.split import time_splits


def main(cfg_path: str, ckpt_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dcfg, mcfg = cfg["data"], cfg["model"]

    df_all = load_nasa_power_hourly(
        dcfg["csv_path"],
        lat=dcfg.get("lat"),
        lon=dcfg.get("lon"),
        ghi_col=dcfg["ghi_col"],
        ws_col=dcfg["ws_col"],
        extra_columns=dcfg.get("extra_exo_cols"),
        skiprows=dcfg.get("skiprows"),
        convert_wh_to_wm2=dcfg.get("convert_wh_to_wm2", False),
    )
    if "split" not in dcfg:
        raise ValueError("configs/nasa_default.yaml 缺少 data.split 配置")
    _, _, df_te = time_splits(df_all, dcfg["split"])

    L, H = int(dcfg["L"]), int(dcfg["H"])
    extra = tuple(dcfg.get("extra_exo_cols", []))
    use_log1p = bool(dcfg.get("use_log1p", True))
    loader = build_dataloader(
        df_te,
        L=L,
        H=H,
        batch_size=int(dcfg["batch_size"]),
        extra_exo_cols=extra,
        shuffle=False,
        use_log1p=use_log1p,
    )

    d_time = 10
    d_x = 2 + d_time
    d_exo = d_time + len(extra)
    d_static = 3
    model = JointFlowForecaster(
        d_x=d_x,
        d_exo=d_exo,
        d_static=d_static,
        H=H,
        d_model=mcfg["d_model"],
        d_ctx=mcfg["d_ctx"],
        flow_layers=mcfg["flow_layers"],
        flow_hidden=mcfg["flow_hidden"],
        n_heads=mcfg.get("n_heads"),
        n_layers=mcfg.get("n_layers"),
        d_ff=mcfg.get("d_ff"),
        dropout=mcfg.get("dropout"),
        include_exo=mcfg.get("include_exo"),
        include_daymask=mcfg.get("include_daymask"),
        include_season=mcfg.get("include_season"),
        mean_pred_branch=mcfg.get("mean_pred_branch"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = loader.dataset
    batch = next(iter(loader))
    with torch.no_grad():
        logp = model.log_prob(batch).mean()
        nll = float((-logp).item())
        temperature = 1.25
        samples = model.sample(batch, n_samples=50, temperature=temperature).numpy()

    samples_phys = dataset.inverse_targets(samples)
    targets_phys = dataset.inverse_targets(batch["y"].numpy())

    cov = interval_coverage(samples_phys, targets_phys, alphas=(0.5, 0.9))
    print("Final NLL (test batch):", nll)
    print("Coverage (overall):", cov)
    print(f"[EVAL] temperature={temperature}")

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_coverage_bar(cov, out_dir / "coverage_overall.png")

    cov05_h = per_horizon_coverage(samples_phys, targets_phys, q=0.5)
    cov09_h = per_horizon_coverage(samples_phys, targets_phys, q=0.9)
    plot_per_horizon_coverage(cov05_h, cov09_h, out_dir / "coverage_per_horizon.png")

    crps = crps_from_samples(samples_phys, targets_phys)
    print("Sample-based CRPS:", crps)

    hist, edges, _ = pit_hist(samples_phys, targets_phys, target_index=0, bins=20)
    plot_pit(hist, edges, out_dir / "pit_ghi.png", "PIT (GHI)")

    times = pd.date_range(start=df_te.index[L], periods=H * batch["y"].shape[0], freq="H")
    cov_by_hour, cnt = hourly_coverage(samples_phys, targets_phys, times, q=0.9)
    plot_hourly_bar(cov_by_hour, out_dir / "coverage_by_hour_90.png", "Coverage by Hour (q=0.9)")
    pd.DataFrame({"hour": list(range(24)), "coverage_q90": cov_by_hour, "count": cnt}).to_csv(
        out_dir / "coverage_by_hour_q90.csv", index=False
    )

    if model.latest_mean_prediction is not None:
        mean_pred = model.latest_mean_prediction.numpy()
        truth_mean_z = batch["y"].numpy().mean(axis=1)
        corr_vals = []
        for idx in range(mean_pred.shape[-1]):
            if np.std(mean_pred[:, idx]) < 1e-6 or np.std(truth_mean_z[:, idx]) < 1e-6:
                continue
            corr = np.corrcoef(mean_pred[:, idx], truth_mean_z[:, idx])[0, 1]
            corr_vals.append(corr)
        if corr_vals:
            print(f"[ENCODER] mean_pred_corr={np.mean(corr_vals):.3f}")

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    main(args.config, args.ckpt)

