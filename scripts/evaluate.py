# scripts/evaluate.py

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ----------------------------------------------------
# 强行把 src/ 放在最前，防止 import 到旧的 informer_flow
# ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.data.dataset import build_dataloader
from informer_flow.models.joint_forecaster import JointFlowForecaster
from informer_flow.utils.metrics import (
    crps_from_samples,
    hourly_coverage,
    interval_coverage,
    per_horizon_coverage,
)
from informer_flow.utils.plotting import (
    plot_coverage_bar,
    plot_per_horizon_coverage,
    plot_hourly_bar,
)
from informer_flow.utils.split import time_splits
import inspect


def main(cfg_path: str, ckpt_path: str):
    # 1. 读配置 ----------------------------------------------------
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    ecfg = cfg.get("eval", {})

    # 2. 读原始 NASA 数据 -------------------------------------------
    df = load_nasa_power_hourly(
        dcfg["csv_path"],
        dcfg.get("lat"),
        dcfg.get("lon"),
        ghi_col=dcfg["ghi_col"],
        ws_col=dcfg["ws_col"],
        skiprows=dcfg.get("skiprows", 0),
    )
    df = df.rename(columns={"wind_speed": "ws"})
    
    print(f"原始数据时间范围: {df.index.min()} -> {df.index.max()}")
    print(f"原始数据总长度: {len(df)}")

    # 3. 时间切分（注意这里一定要传一个 dict） ---------------------
    split_cfg = dcfg.get("splits", dcfg.get("split", {}))
    df_tr, df_va, df_te = time_splits(df, split_cfg)
    
    # 打印测试集信息
    print(f"测试集时间范围: {df_te.index.min()} -> {df_te.index.max()}")
    print(f"测试集数据点数: {len(df_te)}")

    # 4. 统一列名，防止 Dataset 里找不到 ghi / ws --------------------
    #    训练能跑说明 train 那段肯定已经是 ghi/ws 了，
    #    但 evaluate 只拿 test 段，所以这里要再兜一层
    rename_map = {}
    if "ghi" not in df_te.columns:
        raw_ghi = dcfg.get("ghi_col", "ALLSKY_SFC_SW_DWN")
        if raw_ghi in df_te.columns:
            rename_map[raw_ghi] = "ghi"
    if "ws" not in df_te.columns:
        raw_ws = dcfg.get("ws_col", "WS10M")
        if raw_ws in df_te.columns:
            rename_map[raw_ws] = "ws"
    if rename_map:
        df_te = df_te.rename(columns=rename_map)

    # 5. 把没有的外生量补成 0，避免 build_dataloader 里标准化时报 KeyError
    extra_cols = list(dcfg.get("extra_exo_cols", []))
    for col in extra_cols:
        if col not in df_te.columns:
            df_te[col] = 0.0

    # 6. 建 dataloader（兼容有/没有 use_log1p 的版本） --------------
    L = int(dcfg["L"])
    H = int(dcfg["H"])

    sig = inspect.signature(build_dataloader)
    common_kwargs = dict(
        df=df_te,
        L=L,
        H=H,
        batch_size=int(dcfg["batch_size"]),
        extra_exo_cols=tuple(extra_cols),
        shuffle=False,
    )
    if "use_log1p" in sig.parameters:
        loader = build_dataloader(**common_kwargs, use_log1p=dcfg.get("use_log1p", True))
    else:
        loader = build_dataloader(**common_kwargs)

    # 7. 搭模型（跟 train.py 保持一模一样的超参接口） --------------
    d_time = 10
    d_x = 2 + d_time
    d_exo = d_time + len(extra_cols)
    d_static = 3

    model = JointFlowForecaster(
        d_x=d_x,
        d_exo=d_exo,
        d_static=d_static,
        H=H,
        d_model=mcfg.get("d_model", 256),
        d_ctx=mcfg.get("d_ctx", 256),
        flow_layers=mcfg.get("flow_layers", 6),
        flow_hidden=mcfg.get("flow_hidden", 512),
    )

    # 确保模型使用float32
    model = model.float()
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 确保加载的权重也是float32
    for k, v in ckpt["model"].items():
        if torch.is_floating_point(v):
            ckpt["model"][k] = v.float()
    
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 8. 拿一个 batch 做评估 ----------------------------------------
    batch = next(iter(loader))

    with torch.no_grad():
        # 确保模型使用 float32
        model = model.float()
        
        # 确保所有输入都是 float32
        batch = {k: v.float() if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 你的 joint_forecaster 目前是 4 个输入的版本
        out = model.log_prob(
            batch["X_hist"],
            batch["X_fut"],
            batch["y"],
            batch["S"]
        )
        if isinstance(out, tuple):
            logp, _ = out
        else:
            logp = out

        nll = float((-logp).mean().item())
        print("Final NLL (test batch):", nll)

        # 温度按步数拉开
        base_T = float(ecfg.get("temperature_base", 1.10))
        slope_T = float(ecfg.get("temperature_slope", 0.20))
        temps = base_T + slope_T * (np.arange(H) / max(H - 1, 1))  # (H,)
        temps_2d = np.repeat(temps, 2)  # (2H,)
        temps_2d = torch.from_numpy(temps_2d).float()

        n_samples = int(ecfg.get("n_samples", 100))
        samples_z = model.sample(
            batch["X_hist"],
            batch["X_fut"],
            batch["S"],
            n_samples=n_samples,
            temperatures=temps_2d,
        ).numpy()  # (S, B, H, 2)

    # 9. 反标准化回物理尺度 -----------------------------------------
    #    这里我们已经在 test 里保证有 "ghi"/"ws" 了
    ghi_mean = float(df_te["ghi"].mean())
    ghi_std = float(df_te["ghi"].std())
    ws_mean = float(df_te["ws"].mean())
    ws_std = float(df_te["ws"].std())

    targets = batch["y"].numpy()  # 使用标准化的y
    targets_phys = targets.copy()
    targets_phys[..., 0] = targets_phys[..., 0] * ghi_std + ghi_mean
    targets_phys[..., 1] = targets_phys[..., 1] * ws_std + ws_mean

    samples_phys = samples_z.copy()
    samples_phys[..., 0] = samples_phys[..., 0] * ghi_std + ghi_mean
    samples_phys[..., 1] = samples_phys[..., 1] * ws_std + ws_mean

    # 10. 计算覆盖率 & CRPS & 出图 ----------------------------------
    cov = interval_coverage(samples_phys, targets_phys, alphas=(0.5, 0.9))
    print("Coverage:", cov)
    plot_coverage_bar(cov, out_dir / "coverage_overall.png")

    cov05_h = per_horizon_coverage(samples_phys, targets_phys, q=0.5)
    cov09_h = per_horizon_coverage(samples_phys, targets_phys, q=0.9)
    plot_per_horizon_coverage(cov05_h, cov09_h, out_dir / "coverage_per_horizon.png")

    crps = crps_from_samples(samples_phys, targets_phys)
    print("Sample-based CRPS:", crps)

    # 有些时候 test 很短，我们就按实际 batch 拼时间
    times = pd.date_range(start=df_te.index[L], periods=H * batch["y"].shape[0], freq="h")
    cov_by_hour, cnt = hourly_coverage(samples_phys, targets_phys, times, q=0.9)
    plot_hourly_bar(cov_by_hour, out_dir / "coverage_by_hour_90.png", "Coverage by Hour (q=0.9)")
    pd.DataFrame({"hour": list(range(24)), "coverage_q90": cov_by_hour}).to_csv(
        out_dir / "coverage_by_hour_q90.csv", index=False
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt)
