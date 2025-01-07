import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes
folder_met = '/home/jonathan_mn/Descargas/data/met/minuto'
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/LO/minuto'


all_files_met = glob.glob(os.path.join(folder_met, "*.csv"))

all_files_t64 = glob.glob(os.path.join(folder_t64, "*.txt"))


df_met = []
df_t64=[]

for file in all_files_met:
    df = pd.read_csv(file,header=7, encoding='ISO-8859-1',low_memory=False)
    df_met.append(df)

for file in all_files_t64:
    df = pd.read_csv(file) 
    df_t64.append(df)


met_ = pd.concat(df_met, ignore_index=True)
t64_=pd.concat(df_t64,ignore_index=True)


met_['yyyy-mm-dd HH:MM:SS']=pd.to_datetime(met_['yyyy-mm-dd HH:MM:SS'])
met_=met_.sort_values(by='yyyy-mm-dd HH:MM:SS')
met_=met_.reset_index(drop=True)



t64_['Date & Time (Local)']=pd.to_datetime(t64_['Date & Time (Local)'])
t64_=t64_.sort_values(by='Date & Time (Local)')
t64_=t64_.reset_index()

renombrart64 = {

    'Date & Time (Local)':'date',
    '  PM10 Conc':'PM10 Conc',
    '  PM2.5 Conc':'PM2.5 Conc'
}

renombrar = {
    'yyyy-mm-dd HH:MM:SS':'date',
    '°C': 'Temp_Avg',
    '%': 'RH_Avg',
    'm/s': 'WSpeed_Avg',
    'm/s.1': 'WSpeed_Max',
    'deg': 'WDir_Avg',
    'deg.1': 'WDir_SD',
    'mm': 'Rain_Tot',
    'hPa': 'Press_Avg',
    'W/m^2': 'Rad_Avg'
}

met_ = met_.rename(columns=renombrar)
t64_=t64_.rename(columns=renombrart64)

#unir
cmul = pd.merge(met_[['date','Temp_Avg','RH_Avg','WSpeed_Avg','WDir_Avg','Rain_Tot','Press_Avg','Rad_Avg']],
                 t64_[['date','PM10 Conc', 'PM2.5 Conc']], on='date', how='inner')




cmul['date']=pd.to_datetime(cmul['date'])



''' Estadistica'''

mean_pm10 = np.mean(cmul['PM10 Conc'])
max_pm10 = np.max(cmul['PM10 Conc'])
mean_pm25 = np.mean(cmul['PM2.5 Conc'])
max_pm25 = np.max(cmul['PM2.5 Conc'])

plt.figure(figsize=(10, 12))
plt.subplot(2,3,(1,3))


plt.plot(cmul['date'], cmul['PM10 Conc'], label='PM10 Conc', color='#4C4B16')
plt.plot(cmul['date'], cmul['PM2.5 Conc'], label='PM2.5 Conc', alpha=0.9, color='#E6C767')

plt.title('\nObservatorio Atmosférico Calakmul Concentraciones de PM2.5 y PM10')
plt.ylabel('Concentración ($\mathregular{μg/m^{-3}}$)')

plt.legend()
plt.grid(True)

plt.subplot(2, 3,4)

plt.hist(cmul['PM10 Conc'], bins=40, label='PM10', alpha=1, color='#4C4B16')
plt.hist(cmul['PM2.5 Conc'], bins=40, label='PM2.5', alpha=0.85, color='#E6C767')

plt.xlabel('Concentración PM ($\mathregular{μg/m^{-3}}$)')
plt.ylabel('Número de datos (Log)')
plt.yscale('log')
plt.title('\nHistograma PM2.5 y PM10')
plt.legend()
plt.grid(True)

fig_text = f"Promedio PM10: {mean_pm10:.2f}\nPromedio PM2.5: {mean_pm25:.2f} \nMax PM10: {max_pm10:.2f}\nMax PM2.5: {max_pm25:.2f}"
plt.figtext(0.8, 0.38, fig_text, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))

#rosa

ax = plt.subplot(2, 3, 5, projection="windrose") 

wind_data = cmul[['WSpeed_Avg', 'WDir_Avg']].dropna()
ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('Wind Rose')

ax = plt.subplot(2, 3, 6, projection="windrose") 


#Rosa de los vientos concetracion y direccion de viento  
#datos filtrados de PM10 mayores a 300
wind_data = cmul[['PM10 Conc', 'WDir_Avg']].dropna()

filtered_wind_data = wind_data[wind_data['PM10 Conc'] >= 300]


ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM10 Conc'], normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('Wind Rose Concetracion')


plt.tight_layout()
plt.show()




