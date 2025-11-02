import argparse, yaml
from pathlib import Path
from informer_flow.data.nasa_power import load_nasa_power_hourly

def main(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    df = load_nasa_power_hourly(
        d["csv_path"],
        lat=d.get("lat"),
        lon=d.get("lon"),
        ghi_col=d["ghi_col"],
        ws_col=d["ws_col"],
        extra_columns=d.get("extra_exo_cols"),
        skiprows=d.get("skiprows"),
        convert_wh_to_wm2=d.get("convert_wh_to_wm2", False),
    )
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "standard.csv"
    df.to_csv(out_path)
    print(
        "Saved:",
        out_path,
        "rows:",
        len(df),
        "range:",
        df.index.min(),
        "->",
        df.index.max(),
        "lat/lon:",
        d.get("lat"),
        d.get("lon"),
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
