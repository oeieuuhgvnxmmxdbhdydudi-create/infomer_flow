import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.utils as U
import yaml

from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.data.dataset import build_dataloader
from informer_flow.models.joint_forecaster import JointFlowForecaster
from informer_flow.losses.physics import physics_penalty
from informer_flow.utils.seed import set_seed
from informer_flow.utils.split import time_splits


def main(cfg_path: str):
    # 0. 读配置
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"].get("seed", 42))

    dcfg = cfg["data"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]

    # 1. 读 NASA 原始 CSV（自动/手动 skip 都支持）
    df = load_nasa_power_hourly(
        dcfg["csv_path"],
        dcfg.get("lat"),
        dcfg.get("lon"),
        ghi_col=dcfg["ghi_col"],
        ws_col=dcfg["ws_col"],
        skiprows=dcfg.get("skiprows", None),
    )
    df = df.rename(columns={"wind_speed": "ws"})
    # 2. 切分 —— 你的 time_splits 返回的是 (train, val, test)
    split_cfg = dcfg.get("splits", {})
    df_tr, df_va, df_te = time_splits(df, split_cfg)

    L = int(dcfg["L"])
    H = int(dcfg["H"])
    batch_size = int(dcfg["batch_size"])
    extra_exo = tuple(dcfg.get("extra_exo_cols", []))

    # 3. dataloader（按你项目里 dataset.py 的签名来）
    loader_tr = build_dataloader(
        df_tr,
        L=L,
        H=H,
        batch_size=batch_size,
        extra_exo_cols=extra_exo,
    )
    loader_va = build_dataloader(
        df_va,
        L=L,
        H=H,
        batch_size=batch_size,
        extra_exo_cols=extra_exo,
    )

    # 4. 建模型
    d_time = 10
    d_x = 2 + d_time          # (ghi_z, ws_z) + time enc
    d_exo = d_time + len(extra_exo)
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    out_dir = Path(tcfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    lambda_phys = tcfg.get("lambda_phys", 0.05)
    lambda_ent = tcfg.get("lambda_ent", 0.05)
    log_scale_target = tcfg.get("log_scale_target", -1.5)
    max_grad_norm = tcfg.get("max_grad_norm", 1.0)

    history = []
    printed_dbg = False

    # 5. 训练循环
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        for batch in loader_tr:
            opt.zero_grad()

            # 你的 joint_forecaster 我们已经改成返回 (logp, stats)
            S = batch.get("S")
            if S is None:
                B = batch["X_hist"].shape[0]
                if getattr(model.cond_enc, "static_proj", None) is not None:
                    d_static = model.cond_enc.static_proj.in_features
                    S = torch.zeros(B, d_static, device=batch["X_hist"].device, dtype=batch["X_hist"].dtype)
                else:
                    S = None

            logp, stats = model.log_prob(
                batch["X_hist"], batch["X_fut"], batch["y"], S
            )
            nll = -logp.mean()

            # 有的 dataset 可能没有 y_phys，这里兜底
            y_phys = batch.get("y_phys", batch["y"])

            l_phys = physics_penalty(
                batch["y"],
                batch["day_mask"],
                w_night=lambda_phys,
                w_wsneg=lambda_phys,
                y_phys=y_phys,
            )

            avg_log_scale = stats.get("avg_log_scale", None)
            if avg_log_scale is not None:
                l_ent = torch.relu(log_scale_target - avg_log_scale) * lambda_ent
            else:
                l_ent = 0.0

            loss = nll + l_phys + l_ent
            loss.backward()
            U.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            history.append(float(loss.detach().cpu()))

            if not printed_dbg:
                y_np = batch["y"].detach().cpu().numpy()
                print(
                    f"[TRAIN] y mean={y_np.mean():.4f} std={y_np.std():.4f} "
                    f"day_mask_mean={batch['day_mask'].float().mean().item():.4f}"
                )
                printed_dbg = True

        print(
            f"epoch {epoch} | nll={nll.item():.4f} phys={float(l_phys):.4f} "
            f"ent={float(l_ent):.4f} total={loss.item():.4f}"
        )

    # 6. 保存
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg}, ckpt_path)

    plt.figure()
    plt.plot(history)
    plt.title("train loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    print("Saved:", ckpt_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
