
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
from windrose import WindroseAxes

def met_cmul_L1_L2(folder_path):


    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                header_row = None
                for i, line in enumerate(lines):
                    if line.strip() == '':
                        header_row = 6
                        break
                    if 'yyyy-mm-dd HH:MM:SS' in line:
                        header_row = i
                        break
                
            df = pd.read_csv(file_path, header=header_row, encoding='ISO-8859-1')
            all_dfs.append(df)

    
    cmul = pd.concat(all_dfs, ignore_index=True)
    cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.sort_values(by=['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.reset_index(drop=True)
    
    cmul.rename(columns={'deg': 'WDir_Avg','m/s': 'WSpeed_Avg'}, inplace=True)

    return cmul




''' PMT64, mismo timestamp hace que coincida la serie de tiempo en ambos archivos '''

def mismo_timestamp(patht64, cmul):
  
  files_t64 = glob.glob(os.path.join(folder_t64, "*.txt"))
  df_t64 = []
  for file in files_t64:
    df = pd.read_csv(file,delimiter=',')  
    df_t64.append(df)
    t64 = pd.concat(df_t64, ignore_index=True)
  
  t64['Date & Time (Local)'] = pd.to_datetime(t64['Date & Time (Local)'])
  full_datetime_range = pd.date_range(
      start=cmul['yyyy-mm-dd HH:MM:SS'].min(),
      end=cmul['yyyy-mm-dd HH:MM:SS'].max(),
      freq='min'
  )
  t64 = t64.set_index('Date & Time (Local)')
  t64 = t64.reindex(full_datetime_range)
  t64 = t64.reset_index().rename(columns={'index': 'yyyy-mm-dd HH:MM:SS'})
  t64.rename(columns={'  PM10 Conc': 'PM10 Conc','  PM2.5 Conc':'PM2.5 Conc'}, inplace=True)
  return t64


''' Aqui va la direccion de los folder de MET y PM                   '''

folder_cmul = '/home/jonathan_mn/Descargas/data/met/minuto' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/LO/minuto'

'''aqui se mandan a llamar las funciones '''
cmul = met_cmul_L1_L2(folder_cmul)
t64 = mismo_timestamp(folder_t64, cmul)






cmul_winddata = cmul[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
PMdata = t64[['yyyy-mm-dd HH:MM:SS', 'PM10 Conc', 'PM2.5 Conc']]
wr_cmul = pd.merge(cmul_winddata,PMdata, on='yyyy-mm-dd HH:MM:SS', how='inner')
wr_cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(wr_cmul['yyyy-mm-dd HH:MM:SS'])
print(wr_cmul.head())

#ploteo

plt.figure(figsize=(10, 10))  
plt.subplot(2, 3, (1,3))  
plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM10 Conc'], label='PM10 Conc',color='#4C4B16')
plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM2.5 Conc'], label='PM2.5 Conc',alpha=0.9,color='#E6C767')


plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))


plt.title('Observatorio Atmosférico Calakmul Concentraciones de PM2.5 y PM10 ')
plt.ylabel('Concentración ($\mathregular{μg/m^{-3}}$)')
plt.legend()
plt.grid(True)

# Rosa direccion y velocidad de viento 


ax = plt.subplot(2, 3, 4, projection="windrose") 

wind_data = wr_cmul[['WSpeed_Avg', 'WDir_Avg']].dropna()
ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white')
ax.legend(title="Velocidad (m/s)",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title('Velocidad y dirección del viento')


#Rosa de los vientos concetracion y direccion de viento  PM10
#datos filtrados de PM10 mayores a 300
ax = plt.subplot(2, 3, 5, projection="windrose") 
wind_data = wr_cmul[['PM10 Conc', 'WDir_Avg']].dropna()

filtered_wind_data = wind_data[wind_data['PM10 Conc'] >= 300]
ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM10 Conc'], normed=True, opening=0.8, edgecolor='white')
ax.legend(title="Concentración",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title('PM 10 mayor a 300 ($\mathregular{μg/m^{-3}}$)')


#Rosa de los vientos concetracion y direccion de viento  PM2.5
#datos filtrados de PM2.5 mayores a 300

ax = plt.subplot(2, 3, 6, projection="windrose") 
wind_data = wr_cmul[[ 'PM2.5 Conc', 'WDir_Avg']].dropna()

filtered_wind_data = wind_data[wind_data['PM2.5 Conc'] >= 300]
ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM2.5 Conc'], normed=True, opening=0.8, edgecolor='white')
ax.legend(title="Concentración",title_fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.3),prop={'size': 7})
ax.set_title('PM 2.5 mayor a 300 ($\mathregular{μg/m^{-3}}$)')


plt.tight_layout()  
plt.show()



