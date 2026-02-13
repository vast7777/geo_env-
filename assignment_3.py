import tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

df_isd = tools.read_isd_csv(r'C:\Users\15837\geo_env-\41024099999 .csv')
ax = df_isd.plot(figsize=(12, 6), title="ISD data for Jeddah")
fig = ax.get_figure()
fig.savefig("ISD_data_for_Jeddah.png", dpi=300)
plt.show()
df_isd['RH']=tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP'].values)
df_isd['HI']=tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)
print(df_isd.max())
print('-----------------------------')

print(df_isd.idxmax())
print('-----------------------------')

hi_max_time_utc = df_isd['HI'].idxmax()
hi_max_time_local = hi_max_time_utc + pd.Timedelta(hours=3)
print(f"HI_max_time_utc: {hi_max_time_utc}")
print(f"HI_max_time_local: {hi_max_time_local}")
print('-----------------------------')

print(df_isd.loc[[hi_max_time_utc]])
print('relative humidity at HI max time: ', df_isd.loc[hi_max_time_utc, 'RH'])
print('temperature at HI max time: ', df_isd.loc[hi_max_time_utc, 'TMP'])
tem_F= df_isd.loc[hi_max_time_utc, 'TMP'] * 9/5 + 32
print('temperature in F at HI max time: ', tem_F)
hindex = tools.gen_heat_index(df_isd.loc[hi_max_time_utc, 'TMP'], df_isd.loc[hi_max_time_utc, 'RH'])
print('heat index at HI max time: ', hindex)
print('-----------------------------')

print("NWS heat index category at HI max time: ", '124','danger')

print('-----------------------------')
print("Is this a heat wave? ", 'No, because the heat index is not above 35°C for at least three consecutive days.' \
'this is only one day with a high heat index, so it does not meet the criteria for a heat wave.')

print('-----------------------------')
print("Using daily data to calculate HI:")
df_daily = df_isd[['TMP', 'DEW']].resample('D').mean()
df_daily['RH'] = tools.dewpoint_to_rh(df_daily['DEW'].values, df_daily['TMP'].values)
df_daily['HI'] = tools.gen_heat_index(df_daily['TMP'].values, df_daily['RH'].values)
print(f"    daily_max_HI: {df_daily['HI'].max():.2f} °C")
print(f"    hourly_max_HI: {df_isd['HI'].max():.2f} °C")
print("It makes no sense.The daily maximum HI is lower than the hourly maximum HI because daily averaging smooths out short-term peaks in temperature and humidity, resulting in a lower calculated heat index compared to using hourly data which captures those peaks.")
print('-----------------------------')

plt.figure(figsize=(12, 6))
plt.plot(df_isd.index, df_isd['HI'], label='Hourly HI' )
plt.plot(df_daily.index, df_daily['HI'], label='Daily HI')
plt.title('Hourly and Daily Heat Index of Jeddah')
plt.xlabel('Date')
plt.ylabel('Heat Index (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()  
plt.savefig('heat_index_comparison.png', dpi=300)
plt.show()
print('-----------------------------')
#apply +3 C to the air temperature data and recalculate the HI warming to the air temperature data and recalculate the HI
df_isd['TMP_future'] = df_isd['TMP'] + 3.0
df_isd['RH_future'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP_future'].values)
df_isd['HI_future'] = tools.gen_heat_index(df_isd['TMP_future'].values, df_isd['RH_future'].values)

hi_max_current = df_isd['HI'].max()
hi_max_future = df_isd['HI_future'].max()
print(f"current max HI: {hi_max_current:.2f} °C")
print(f" future max HI: {hi_max_future:.2f} °C")
print(f" HI increase: {hi_max_future - hi_max_current:.2f} °C")
