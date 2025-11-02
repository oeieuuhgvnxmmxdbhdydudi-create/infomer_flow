from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.utils.split import time_splits
from informer_flow.features.solar import daytime_mask_from_sun
import numpy as np

path = "POWER_Point_Hourly_20150101_20250101_033d22N_122d43E_UTC.csv"
df = load_nasa_power_hourly(path, None, None, ghi_col='ALLSKY_SFC_SW_DWN', ws_col='WS10M', skiprows=15)
train_df, _, _ = time_splits(df, {
    'train': {'start': '2015-01-01', 'end': '2022-12-31'},
    'val': {'start': '2023-01-01', 'end': '2023-12-31'},
    'test': {'start': '2024-01-01', 'end': '2024-12-31'}
})
mask = daytime_mask_from_sun(train_df.index, float(train_df['lat'].iloc[0]), float(train_df['lon'].iloc[0]))
print('nan count', int(np.isnan(mask).sum()))
print(mask.head())
