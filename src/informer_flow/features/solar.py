import numpy as np
import pandas as pd

__all__ = ["solar_elevation_deg", "daytime_mask_from_sun", "time_fourier_features"]

def solar_elevation_deg(dt_utc: pd.Series, lat_deg: float, lon_deg: float) -> np.ndarray:
    ts = pd.to_datetime(dt_utc, utc=True)
    jd = ts.view("int64") / 86400000000000.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T**2) % 360
    M  = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    e  = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    C  = (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(np.deg2rad(M)) \
       + (0.019993 - 0.000101 * T) * np.sin(np.deg2rad(2*M)) \
       + 0.000289 * np.sin(np.deg2rad(3*M))
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = true_long - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    eps0 = 23 + 26/60 + 21.448/3600 - (46.8150*T + 0.00059*T**2 - 0.001813*T**3)/3600
    eps  = eps0 + 0.00256 * np.cos(np.deg2rad(omega))
    sin_dec = np.sin(np.deg2rad(eps)) * np.sin(np.deg2rad(lam))
    dec = np.rad2deg(np.arcsin(sin_dec))
    y = np.tan(np.deg2rad(eps/2))**2
    EoT = 4 * np.rad2deg(y*np.sin(2*np.deg2rad(L0)) - 2*e*np.sin(np.deg2rad(M)) \
          + 4*e*y*np.sin(np.deg2rad(M))*np.cos(2*np.deg2rad(L0)) \
          - 0.5*y*y*np.sin(4*np.deg2rad(L0)) - 1.25*e*e*np.sin(2*np.deg2rad(M)))
    minutes = (ts.hour*60 + ts.minute + ts.second/60.0)
    tst = minutes + EoT + 4*lon_deg
    H = (tst/4.0 - 180)
    lat = lat_deg
    elev = np.rad2deg(np.arcsin(
        np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(dec)) +
        np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(dec))*np.cos(np.deg2rad(H))
    ))
    return elev

def daytime_mask_from_sun(dt_utc: pd.Series, lat_deg: float, lon_deg: float, thresh_deg: float = 0.5) -> np.ndarray:
    elev = solar_elevation_deg(dt_utc, lat_deg, lon_deg)
    return (elev > thresh_deg).astype(float)

def time_fourier_features(dt: pd.DatetimeIndex, max_k_hour: int = 3, max_k_doy: int = 2):
    h = dt.hour.values + dt.minute.values/60.0
    doy = dt.dayofyear.values.astype(float)
    feats = []
    for k in range(1, max_k_hour+1):
        feats += [np.sin(2*np.pi*k*h/24), np.cos(2*np.pi*k*h/24)]
    for k in range(1, max_k_doy+1):
        feats += [np.sin(2*np.pi*k*doy/365.25), np.cos(2*np.pi*k*doy/365.25)]
    cols = [f"hour_s{k}" for k in range(1, max_k_hour+1)] + [f"hour_c{k}" for k in range(1, max_k_hour+1)] + \
           [f"doy_s{k}" for k in range(1, max_k_doy+1)] + [f"doy_c{k}" for k in range(1, max_k_doy+1)]
    return np.vstack(feats).T, cols
