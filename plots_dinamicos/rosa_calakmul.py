import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes


''' Este script grafica la direccion del viento y concentrraciones mayores a 300 '''

folder_met = r'DATOS Sensores\met\L2\minuto'
folder_t64 = r'DATOS Sensores/pm/L0/minuto'


all_files_met = glob.glob(os.path.join(folder_met, "*.csv"))

all_files_t64 = glob.glob(os.path.join(folder_t64, "*.txt"))


df_met = []
df_t64=[]

for file in all_files_met:
    df = pd.read_csv(file,header=6, encoding='ISO-8859-1',low_memory=False)
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

plt.figure(figsize=(12, 6))


#rosa de los vientos de direccion y velocidad del viento

ax = plt.subplot(1, 3, 1, projection="windrose") 

wind_data = cmul[['WSpeed_Avg', 'WDir_Avg']].dropna()
ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white')
#ax.set_legend()




ax.legend(title="Velocidad (m/s)",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title('Velocidad y dirección del viento 2024')





#Rosa de los vientos concetracion y direccion de viento  PM10
#datos filtrados de PM10 mayores a 300
ax = plt.subplot(1, 3, 2, projection="windrose") 


wind_data = cmul[['PM10 Conc', 'WDir_Avg']].dropna()

PM10_umbral = 400
filtered_wind_data = wind_data[wind_data['PM10 Conc'] >= PM10_umbral]


ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM10 Conc'], normed=True, opening=0.8, edgecolor='white')

ax.legend(title="Concentración",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title(f'PM 10 mayor a {PM10_umbral} μg/m^3')



#Rosa de los vientos concetracion y direccion de viento  PM2.5
#datos filtrados de PM2.5 mayores a 300

ax = plt.subplot(1, 3, 3, projection="windrose") 
wind_data = cmul[[ 'PM2.5 Conc', 'WDir_Avg']].dropna()

PM25_umbral = 200
filtered_wind_data = wind_data[wind_data['PM2.5 Conc'] >= PM25_umbral]
ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM2.5 Conc'], normed=True, opening=0.8, edgecolor='white')
ax.legend(title="Concentración",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title(f'PM 2.5 mayor a {PM25_umbral} μg/m^3')


plt.suptitle('Rosa de los vientos Calakmul ')
plt.tight_layout()

plt.show()