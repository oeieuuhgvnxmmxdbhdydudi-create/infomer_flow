from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.utils.split import time_splits
from informer_flow.features.solar import daytime_mask_from_sun
from informer_flow.data.dataset import NasaSlidingWindow
import numpy as np

path = "POWER_Point_Hourly_20150101_20250101_033d22N_122d43E_UTC.csv"
df = load_nasa_power_hourly(path, None, None, ghi_col='ALLSKY_SFC_SW_DWN', ws_col='WS10M', skiprows=15)
df = df.rename(columns={'wind_speed': 'ws'})
split_cfg = {
    'train': {'start': '2015-01-01', 'end': '2022-12-31'},
    'val': {'start': '2023-01-01', 'end': '2023-12-31'},
    'test': {'start': '2024-01-01', 'end': '2024-12-31'}
}
train_df, _, _ = time_splits(df, split_cfg)
mask = daytime_mask_from_sun(train_df.index, float(train_df['lat'].iloc[0]), float(train_df['lon'].iloc[0]))
print('mask nan count', int(np.isnan(mask).sum()))
nsw = NasaSlidingWindow(train_df, L=48, H=24)
print('nsw nan count', int(np.isnan(nsw.df['day_mask']).sum()))
print('mask head', mask.head())
print('nsw head', nsw.df['day_mask'].head())
