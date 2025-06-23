''' 



En este archivo se van a comparar los valores de CO2 y NDVI


'''


from picarro import *
from picarro_clean import *




folder_path = 'DATOS Sensores/gei/L1b/minuto/2024'

ndvi_folder_path = 'gei/CO2_NDVI/files/MOD13Q1_NDVI_2024.txt'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])






def resample24_month(df):


    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month
    df_resampled['Dia'] = df_resampled.index.day

    
    df_monthly_avg_24h = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    df_month=df_monthly_avg_24h.groupby('Mes').mean().reset_index()
    df_month['Mes'] = df_month['Mes']
    df_month['Mes'] = pd.to_datetime(df_month['Mes'], format='%m').dt.strftime('%m')
    df_month['CO2_Avg'] = df_month['CO2_Avg'].round(3)
    df_month= df_month[['Mes', 'CO2_Avg']]
    return df_month


geib=resample24_month(gei)

print(geib)


''' 
gei_monthly = gei.resample('ME', on='Time').mean().reset_index()
gei_monthly = gei_monthly[['Time', 'CO2_Avg']]
gei_monthly['Time'] = gei_monthly['Time'].dt.strftime('%m')


print(gei_monthly.head())


'''


ndvi_avg = pd.read_csv(ndvi_folder_path, sep=',', header=0,)

ndvi_avg['Mes'] = pd.to_datetime(ndvi_avg['Mes'], format='%m').dt.strftime('%m')

print(ndvi_avg)








# Unimos los dataframes por mes
ndvi_co2 = pd.merge(geib, ndvi_avg, left_on='Mes', right_on='Mes', how='inner')
ndvi_co2 = ndvi_co2[['Mes', 'CO2_Avg', 'NDVI_Mean']]
print(ndvi_co2)



# ploteamos los datos
plt.figure(figsize=(12, 6))


axx=ndvi_co2['Mes']


ax1 = plt.gca()
ax1.set_xlabel('Month')
ax1.set_ylabel('CO2 (ppm)', color='blue')
ax1.plot(axx, ndvi_co2['CO2_Avg'], marker='o', color='blue', label='CO2 Avg')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(420, 450)  


ax2 = ax1.twinx()
ax2.set_ylabel('NDVI', color='green')
ax2.plot(axx, ndvi_co2['NDVI_Mean'], marker='x', color='green', label='NDVI Mean')
ax2.tick_params(axis='y', labelcolor='green')
ax2.set_ylim(0,1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('CO2 y NDVI (2024)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


