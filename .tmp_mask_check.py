from informer_flow.data.dataset import build_dataloader
from informer_flow.data.nasa_power import load_nasa_power_hourly
from informer_flow.utils.split import time_splits
import numpy as np

path = 'POWER_Point_Hourly_20150101_20250101_033d22N_122d43E_UTC.csv'
df = load_nasa_power_hourly(path, None, None, ghi_col='ALLSKY_SFC_SW_DWN', ws_col='WS10M', skiprows=15)
df = df.rename(columns={'wind_speed': 'ws'})
split_cfg = {
    'train': {'start': '2015-01-01', 'end': '2022-12-31'},
    'val': {'start': '2023-01-01', 'end': '2023-12-31'},
    'test': {'start': '2024-01-01', 'end': '2024-12-31'}
}
train_df, val_df, test_df = time_splits(df, split_cfg)
loader = build_dataloader(train_df, L=48, H=24, batch_size=32)
batch = next(iter(loader))
print('day_mask shape', batch['day_mask'].shape)
print('has_nan', np.isnan(batch['day_mask']).any())
print(batch['day_mask'][0, :10])
