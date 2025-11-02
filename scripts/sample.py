import argparse, yaml, numpy as np, torch
from pathlib import Path
from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.data.dataset import build_dataloader
from informer_flow.models.joint_forecaster import JointFlowForecaster
from informer_flow.utils.plotting import plot_joint_samples


def main(cfg_path, ckpt_path, n):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg = cfg["data"], cfg["model"]

    df = load_nasa_power_hourly(
        dcfg["csv_path"],
        lat=dcfg.get("lat"),
        lon=dcfg.get("lon"),
        ghi_col=dcfg["ghi_col"],
        ws_col=dcfg["ws_col"],
        extra_columns=dcfg.get("extra_exo_cols"),
        skiprows=dcfg.get("skiprows"),
        convert_wh_to_wm2=dcfg.get("convert_wh_to_wm2", False),
    )
    L, H = int(dcfg["L"]), int(dcfg["H"])
    loader = build_dataloader(
        df,
        L=L,
        H=H,
        batch_size=int(dcfg["batch_size"]),
        extra_exo_cols=tuple(dcfg.get("extra_exo_cols", [])),
        shuffle=False,
        use_log1p=bool(dcfg.get("use_log1p", True)),
    )

    d_time = 10
    d_x = 2 + d_time
    d_exo = d_time + len(tuple(dcfg.get("extra_exo_cols", [])))
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
        samples = model.sample(batch, n_samples=n, temperature=1.25).numpy()
    samples_phys = dataset.inverse_targets(samples)

    out_dir = Path(cfg["train"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "samples.npy", samples_phys)
    plot_joint_samples(samples_phys, out_dir / "ghi_ws_joint.png")
    print("Saved:", out_dir / "samples.npy", out_dir / "ghi_ws_joint.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.n)


