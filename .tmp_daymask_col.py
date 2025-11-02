from informer_flow.data.dataset import NasaSlidingWindow
from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.utils.split import time_splits
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
ds = NasaSlidingWindow(train_df, L=48, H=24)
print('NaN in day_mask column:', np.isnan(ds.df['day_mask']).sum())
print(ds.df['day_mask'].head())
