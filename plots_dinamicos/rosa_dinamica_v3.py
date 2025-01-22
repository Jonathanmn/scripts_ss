
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
from windrose import WindroseAxes
from datetime import datetime

''' Aqui va la direccion de los folder de MET y PM                   '''

'''

 hora
folder_cmul = '/home/jonathan_mn/Descargas/data/met/L2/hora' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/L0/hora'
min
folder_cmul = '/home/jonathan_mn/Descargas/data/met/L2/minuto' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/L0/minuto'
'''



folder_cmul = '/home/jonathan_mn/Descargas/data/met/L2/hora' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/L0/hora'

start_date1 = datetime(2024, 5, 15, 6, 0, 0)
end_date1 = datetime(2024, 5, 16, 0, 0, 0)


def met_cmul(folder_path):

    all_dfs = []  

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            
            df = pd.read_csv(file_path, encoding='ISO-8859-1', header=6) 
            all_dfs.append(df)

    
    cmul = pd.concat(all_dfs, ignore_index=True) 
    cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.sort_values(by=['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.reset_index(drop=True)
    
    cmul.rename(columns={'deg': 'WDir_Avg','m/s': 'WSpeed_Avg'}, inplace=True)

    return cmul 

    



''' PMT64, mismo timestamp hace que coincida la serie de tiempo en ambos archivos '''

def t64_cmul(patht64):
  
  files_t64 = glob.glob(os.path.join(folder_t64, "*.txt"))
  df_t64 = []
  for file in files_t64:
    df = pd.read_csv(file,delimiter=',')  
    df_t64.append(df)
    t64 = pd.concat(df_t64, ignore_index=True)
  
  t64['Date & Time (Local)'] = pd.to_datetime(t64['Date & Time (Local)'])
  t64 = t64.sort_values(by=['Date & Time (Local)'])
  t64 = t64.reset_index().rename(columns={'index': 'yyyy-mm-dd HH:MM:SS'})

  t64.rename(columns={'  PM10 Conc': 'PM10 Conc','  PM2.5 Conc':'PM2.5 Conc'}, inplace=True)
  
  return t64




'''aqui se mandan a llamar las funciones '''
cmul = met_cmul(folder_cmul)
t64 = t64_cmul(folder_t64)
#se toma por fecha o toda la serie de tiempo 
cmul_winddata = cmul[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
PMdata = t64[['PM10 Conc', 'PM2.5 Conc']]
wr_all_time = pd.concat([cmul_winddata, PMdata], axis=1, join='inner')
wr_all_time['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(wr_all_time['yyyy-mm-dd HH:MM:SS'])
wr_per_date = wr_all_time[(wr_all_time['yyyy-mm-dd HH:MM:SS'] >= start_date1) & (wr_all_time['yyyy-mm-dd HH:MM:SS']<= end_date1)]


#ploteo
def rosa_pm(wr_cmul):

    plt.figure(figsize=(18, 9))  

    
    


    plt.subplot(2, 3, (1, 3))  
    plt.subplot(2, 3, (1, 3))  
    plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM10 Conc'], 
             label='PM10 Conc', color='#4C4B16')
    plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM2.5 Conc'], 
             label='PM2.5 Conc', alpha=0.9, color='#E6C767')
    
    
    plt.title('Observatorio Atmosférico Calakmul Concentraciones de PM2.5 y PM10 \n')
    plt.ylabel('\nConcentración ($\mathregular{μg/m^{-3}}$)')
    
    #datos estadisticos.
    
    max_ws = wr_cmul['WSpeed_Avg'].max()
    max_pm10=wr_cmul['PM10 Conc'].max()
    max_pm25=wr_cmul['PM2.5 Conc'].max()
    mean_pm10=wr_cmul['PM10 Conc'].mean()
    mean_pm25=wr_cmul['PM2.5 Conc'].mean()
    
    plt.figtext(0.1, 0.8, f'Velocidad máxima: {max_ws:.1f}m/s \nMáx PM10: {max_pm10:.1f} \nMáx PM 2.5: {max_pm25:.1f}\nConcentración promedio\nPM10: {mean_pm10:.1f}\nPM 2.5: {mean_pm25:.1f}',  # Adjust position as needed
            fontsize=10, color='black',bbox=dict(facecolor='white', alpha=1, boxstyle='round'))  

    plt.legend()
    plt.grid(True)



    '''Rosas de viento           '''

    # DIRECCION DE VIENTO
    ax = plt.subplot(2, 3, 4, projection="windrose")  
    wind_data = wr_cmul[['WSpeed_Avg', 'WDir_Avg']].dropna()
    ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Velocidad (m/s)", title_fontsize=8,
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title('\nVelocidad y dirección del viento')

    # PM10 CONC
    ax = plt.subplot(2, 3, 5, projection="windrose")  
    wind_data = wr_cmul[['PM10 Conc', 'WDir_Avg']].dropna()

    max_pm10 = wr_cmul['PM10 Conc'].max()
    PM10_umbral=round(max_pm10-(max_pm10*0.50),2)
    

    filtered_wind_data = wind_data[wind_data['PM10 Conc'] >= PM10_umbral]
    ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM10 Conc'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Concentración", title_fontsize=8, 
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title(f'PM 10 mayor a {PM10_umbral} μg/m^3')

    # PM2.5
    ax = plt.subplot(2, 3, 6, projection="windrose")  
    wind_data = wr_cmul[['PM2.5 Conc', 'WDir_Avg']].dropna()

    max_pm25 = wr_cmul['PM2.5 Conc'].max()
    PM25_umbral=round(max_pm25-(max_pm25*0.5),2)

    filtered_wind_data = wind_data[wind_data['PM2.5 Conc'] >= PM25_umbral]
    ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM2.5 Conc'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Concentración", title_fontsize=8, 
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title(f'PM 2.5 mayor a {PM25_umbral} μg/m^3')

    plt.tight_layout()  
    plt.show()




#funcion que se manda a llamar

''' La funcion rosa_pm, grafica la serie de tiempo y rosas de concentracion de pm 10 y 2.5  
 el agumento per_date o all_time significa graficar por periodo de fecha o toda la serie de tiempo'''

#rosa_pm(wr_per_date)
rosa_pm(wr_all_time)

