from informer_flow.data.nasa_power import load_nasa_power_hourly

path = "POWER_Point_Hourly_20150101_20250101_033d22N_122d43E_UTC.csv"
df = load_nasa_power_hourly(path, None, None, ghi_col='ALLSKY_SFC_SW_DWN', ws_col='WS10M', skiprows=15)
print('index NaT:', int(df.index.isna().sum()))
print('total rows:', len(df))
print(df.index[:5])
