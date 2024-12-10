import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re  
from datetime import datetime

''' ESTE SCRIPT PERMITE REALIZAR UN PLOT EN CUALQUIER FECHA U HORA EN INTERVALOS.,ingresa un intervalo de fechas en formato yyyy-mm-dd hh:mm:ss   
si quieres poner 2023-11-18 15:00:00 se escribe en datetime así '''

start_date1 = datetime(2024, 1, 6, 12, 0, 0)
end_date1 = datetime(2024, 6, 6, 20, 0, 0)

folder_path = '/home/jonathan_mn/Descargas/data/t64/LO/minuto'



start_date = start_date1.strftime('%Y_%m')  
end_date = end_date1.strftime('%Y_%m')   
all_files = os.listdir(folder_path)

files_to_read = []
for file in all_files:
    if re.match(r'\d{4}_\d{2}_CMUL_PM.txt', file):  
        file_date = file[:7]  
        if start_date <= file_date <= end_date:
            files_to_read.append(file)


if not files_to_read:
    print("No se encontró archivos, revisa tu rango de fecha.")
else:
    
    dfs = []
    for file in files_to_read:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df['Date & Time (Local)'] = pd.to_datetime(df['Date & Time (Local)'])  
        dfs.append(df)

    print('graficando')

    t64 = pd.concat(dfs, ignore_index=True)

    t64['Date & Time (Local)'] = pd.to_datetime(t64['Date & Time (Local)'])
    t64 = t64.sort_values(by='Date & Time (Local)')
    t64 = t64.reset_index(drop=True)



    filtered_t64 = t64[(t64['Date & Time (Local)'] >= start_date1) & (t64['Date & Time (Local)'] <= end_date1)]

    ''' Estadistica'''

    mean_pm10 = np.mean(filtered_t64['  PM10 Conc'])
    max_pm10 = np.max(filtered_t64['  PM10 Conc'])
    mean_pm25 = np.mean(filtered_t64['  PM2.5 Conc'])
    max_pm25 = np.max(filtered_t64['  PM2.5 Conc'])






    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(filtered_t64['Date & Time (Local)'], filtered_t64['  PM10 Conc'], label='PM10 Conc',color='#4C4B16')
    plt.plot(filtered_t64['Date & Time (Local)'], filtered_t64['  PM2.5 Conc'], label='PM2.5 Conc',alpha=0.9,color='#E6C767')



    plt.title('\nObservatorio Atmosférico Calakmul Concentraciones de PM2.5 y PM10\n de {} al {}'.format(start_date1.strftime('%Y-%m-%d %H:%M:%S'), end_date1.strftime('%Y-%m-%d %H:%M:%S')))
    plt.ylabel('Concentración ($\mathregular{μg/m^{-3}}$)')

    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.hist(filtered_t64['  PM10 Conc'], bins=40, label='PM10', alpha=1, color='#4C4B16')
    plt.hist(filtered_t64['  PM2.5 Conc'], bins=40, label='PM2.5', alpha=0.85, color='#E6C767')

    plt.xlabel('Concentración PM ($\mathregular{μg/m^{-3}}$)')
    plt.ylabel('Número de datos (Log)')
    plt.yscale('log')
    plt.title('\nHistograma PM2.5 y PM10')
    plt.legend()
    plt.grid(True)

    fig_text = f"Promedio PM10: {mean_pm10:.2f}\nPromedio PM2.5: {mean_pm25:.2f} \nMax PM10: {max_pm10:.2f}\nMax PM2.5: {max_pm25:.2f}" 
    plt.figtext(0.8, 0.38, fig_text, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))  
    plt.tight_layout()
    plt.show()  

