import re
import pandas as pd

__all__ = ["load_nasa_power_hourly", "parse_latlon_from_name"]

def parse_latlon_from_name(name: str):
    # 例：..._033d22N_122d43E_...
    m = re.search(r"_(\d{2,3})d(\d{2})([NS])_(\d{2,3})d(\d{2})([EW])_", name)
    if not m:
        return None
    lat_deg, lat_min, lat_hemi, lon_deg, lon_min, lon_hemi = m.groups()
    lat = int(lat_deg) + int(lat_min)/60.0
    lon = int(lon_deg) + int(lon_min)/60.0
    if lat_hemi == "S": lat = -lat
    if lon_hemi == "W": lon = -lon
    return round(lat, 6), round(lon, 6)

def load_nasa_power_hourly(csv_path: str, lat=None, lon=None,
                           ghi_col: str = "ALLSKY_SFC_SW_DWN", ws_col: str = "WS10M",
                           skiprows: int = 0, convert_mj_to_wm2: bool = False):
    df = pd.read_csv(csv_path, skiprows=skiprows)

    if {"YEAR","MO","DY","HR"}.issubset(df.columns):
        dt = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR), utc=True)
    elif {"YYYYMMDD","HH"}.issubset(df.columns):
        day = pd.to_datetime(df["YYYYMMDD"].astype(str), format="%Y%m%d", utc=True)
        dt  = day + pd.to_timedelta(df["HH"], unit="h")
    elif "time" in df.columns:
        dt = pd.to_datetime(df["time"], utc=True)
    else:
        raise ValueError("未识别时间字段；需要 YEAR,MO,DY,HR 或 YYYYMMDD,HH 或 time")

    if ghi_col not in df.columns or ws_col not in df.columns:
        raise ValueError(f"缺少必要列: {ghi_col} 或 {ws_col}")

    if convert_mj_to_wm2:
        df[ghi_col] = df[ghi_col].astype(float) * 1e6 / 3600.0

    out = pd.DataFrame({
        "ghi": pd.to_numeric(df[ghi_col], errors="coerce"),
        "ws":  pd.to_numeric(df[ws_col],  errors="coerce"),
    }, index=pd.DatetimeIndex(dt, name="datetime")).sort_index()

    out.loc[out["ghi"] < 0, "ghi"] = 0
    out = out.dropna()

    if lat is None or lon is None:
        latlon = parse_latlon_from_name(str(csv_path).split("/")[-1])
        if latlon:
            lat, lon = latlon
    out["lat"] = lat if lat is not None else 0.0
    out["lon"] = lon if lon is not None else 0.0
    return out
