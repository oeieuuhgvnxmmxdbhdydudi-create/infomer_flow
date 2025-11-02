import numpy as np
from informer_flow.features.solar import daytime_mask_from_sun
from informer_flow.data.nasa_power import load_nasa_power_hourly

path = "POWER_Point_Hourly_20150101_20250101_033d22N_122d43E_UTC.csv"
df = load_nasa_power_hourly(path, None, None, ghi_col='ALLSKY_SFC_SW_DWN', ws_col='WS10M', skiprows=15)
mask = daytime_mask_from_sun(df.index, float(df['lat'].iloc[0]), float(df['lon'].iloc[0]))
print(mask.shape, np.isnan(mask).sum())
print(mask[:10])
