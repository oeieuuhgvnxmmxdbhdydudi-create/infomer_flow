import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.utils as U
import yaml

from informer_flow.data.dataset import build_dataloader
from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.losses.physics import physics_penalty
from informer_flow.models.joint_forecaster import JointFlowForecaster
from informer_flow.utils.seed import set_seed
from informer_flow.utils.split import time_splits


def to_device(batch, device):
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def main(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    set_seed(cfg["train"].get("seed", 42))
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    device = torch.device(tcfg.get("device", "cpu"))

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
    df_tr, df_va, _ = time_splits(df_all, dcfg["split"])

    L, H = int(dcfg["L"]), int(dcfg["H"])
    extra = tuple(dcfg.get("extra_exo_cols", []))
    batch_size = int(dcfg["batch_size"])
    use_log1p = bool(dcfg.get("use_log1p", True))
    loader_tr = build_dataloader(
        df_tr,
        L=L,
        H=H,
        batch_size=batch_size,
        extra_exo_cols=extra,
        shuffle=True,
        use_log1p=use_log1p,
    )
    loader_va = build_dataloader(
        df_va,
        L=L,
        H=H,
        batch_size=batch_size,
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]))
    lambda_phys = float(tcfg.get("lambda_phys", 0.01))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_tr, hist_va = [], []
    epochs = int(tcfg["epochs"])
    for ep in range(epochs):
        model.train()
        tr_losses = []
        printed_dbg = False
        for batch in loader_tr:
            batch = to_device(batch, device)
            opt.zero_grad()
            logp = model.log_prob(batch)
            nll = -logp.mean()
            l_phys = physics_penalty(
                batch["y"],
                batch["day_mask"],
                w_night=lambda_phys,
                w_wsneg=lambda_phys,
                y_phys=batch["y_phys"],
            )
            loss = nll + l_phys
            loss.backward()
            U.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            if not printed_dbg:
                y_np = batch["y"].detach().cpu().numpy()
                print(
                    f"[TRAIN] y mean={y_np.mean():.4f} std={y_np.std():.4f} "
                    f"day_mask_mean={batch['day_mask'].mean().item():.4f}"
                )
                printed_dbg = True
            tr_losses.append(float(loss.item()))
        tr_mean = float(np.mean(tr_losses))
        hist_tr.append(tr_mean)

        model.eval()
        va_losses = []
        with torch.no_grad():
            for batch in loader_va:
                batch = to_device(batch, device)
                logp = model.log_prob(batch)
                nll = -logp.mean()
                l_phys = physics_penalty(
                    batch["y"],
                    batch["day_mask"],
                    w_night=lambda_phys,
                    w_wsneg=lambda_phys,
                    y_phys=batch["y_phys"],
                )
                va_losses.append(float((nll + l_phys).item()))
        va_mean = float(np.mean(va_losses))
        hist_va.append(va_mean)

        print(f"epoch {ep} | train_total={tr_mean:.4f} | val_total={va_mean:.4f}")

    torch.save({"model": model.state_dict(), "cfg": cfg}, out_dir / "checkpoint.pt")

    plt.figure()
    plt.plot(hist_tr, label="train_total")
    plt.plot(hist_va, label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    print("Saved:", out_dir / "checkpoint.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
