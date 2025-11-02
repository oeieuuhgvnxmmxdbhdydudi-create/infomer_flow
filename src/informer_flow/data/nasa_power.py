import re
from pathlib import Path
import pandas as pd

__all__ = ["load_nasa_power_hourly", "parse_latlon_from_name"]


def parse_latlon_from_name(name: str):
    m = re.search(r"_(\d{2,3})d(\d{2})([NS])_(\d{2,3})d(\d{2})([EW])_", name)
    if not m:
        return None
    lat_deg, lat_min, lat_hemi, lon_deg, lon_min, lon_hemi = m.groups()
    lat = int(lat_deg) + int(lat_min) / 60.0
    lon = int(lon_deg) + int(lon_min) / 60.0
    if lat_hemi == "S":
        lat = -lat
    if lon_hemi == "W":
        lon = -lon
    return round(lat, 6), round(lon, 6)


def _detect_data_start(csv_path: Path) -> int:
    """自动找出哪一行是表头."""
    with csv_path.open("r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        up = line.strip().upper()
        if up.startswith("YEAR,") or up.startswith("TIMESTAMP") or up.startswith("TIME,"):
            return i
    return 0


def load_nasa_power_hourly(
    csv_path: str,
    lat=None,
    lon=None,
    ghi_col: str = "ALLSKY_SFC_SW_DWN",
    ws_col: str = "WS10M",
    skiprows: int = None,
    convert_mj_to_wm2: bool = False,
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"未找到文件: {csv_path}")

    if skiprows is None:
        skiprows = _detect_data_start(csv_path)

    df = pd.read_csv(
        csv_path,
        skiprows=skiprows,
        encoding="utf-8-sig",
    )

    # 标准化列名
    df.columns = [c.strip().upper() for c in df.columns]
    cols = set(df.columns)

    # 时间识别
    if {"YEAR", "MO", "DY", "HR"}.issubset(cols):
        dt = pd.to_datetime(
            dict(
                year=df["YEAR"].astype(int),
                month=df["MO"].astype(int),
                day=df["DY"].astype(int),
                hour=df["HR"].astype(int),
            ),
            errors="coerce",
            utc=True,
        )
    elif "TIMESTAMP" in cols:
        dt = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)
    elif "TIME" in cols:
        dt = pd.to_datetime(df["TIME"], errors="coerce", utc=True)
    else:
        raise ValueError(
            "未识别时间字段；需要 YEAR,MO,DY,HR / TIME / TIMESTAMP\n"
            f"当前文件列名: {list(df.columns)}"
        )

    df = df.loc[dt.notna()].copy()
    df["datetime"] = dt[dt.notna()]
    df = df.set_index("datetime").sort_index()

    # 取 GHI / 风
    ghi_up = ghi_col.upper()
    ws_up = ws_col.upper()

    if ghi_up not in df.columns:
        raise ValueError(f"找不到辐射列 {ghi_col}；当前列有: {list(df.columns)}")
    if ws_up not in df.columns:
        raise ValueError(f"找不到风速列 {ws_col}；当前列有: {list(df.columns)}")

    if convert_mj_to_wm2:
        df[ghi_up] = df[ghi_up].astype(float) * 1e6 / 3600.0

    out = df.rename(
        columns={
            ghi_up: "ghi",
            ws_up: "wind_speed",
        }
    )

    out["ghi"] = pd.to_numeric(out["ghi"], errors="coerce")
    out["wind_speed"] = pd.to_numeric(out["wind_speed"], errors="coerce")
    out = out.dropna(subset=["ghi", "wind_speed"])
    out.loc[out["ghi"] < 0, "ghi"] = 0

    # 自动补经纬度
    if lat is None or lon is None:
        latlon = parse_latlon_from_name(csv_path.name)
        if latlon:
            lat, lon = latlon
    out["lat"] = lat if lat is not None else 0.0
    out["lon"] = lon if lon is not None else 0.0

    return out
