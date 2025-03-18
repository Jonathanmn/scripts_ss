
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
from windrose import WindroseAxes
from datetime import datetime




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
  
  files_t64 = glob.glob(os.path.join(patht64, "*.txt"))
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







'''PLOTEOS DE ROSAS DE VIENTO     MET Y PM '''





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
    plt.ylabel('\nConcentración μg/m^3')
    
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



def met_windrose(wr_cmul):
    """
    Esta función toma un DataFrame y plotea rosas de viento en un subplot de 4x3 para cada mes de 2024.
    """
    # Filtrar datos para el año 2024
    wr_cmul_2024 = wr_cmul[wr_cmul['yyyy-mm-dd HH:MM:SS'].dt.year == 2024]

    # Crear una figura con subplots 4x3
    fig, axs = plt.subplots(4, 3, figsize=(9, 15), subplot_kw={'projection': 'windrose'})
    fig.suptitle('Rosas de Viento Mensuales para 2024', fontsize=16)

    # Nombres de los meses
    month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

    # Iterar sobre cada mes y crear una rosa de viento
    for month in range(1, 13):
        ax = axs[(month-1)//3, (month-1)%3]
        monthly_data = wr_cmul_2024[wr_cmul_2024['yyyy-mm-dd HH:MM:SS'].dt.month == month]
        wind_data = monthly_data[['WSpeed_Avg', 'WDir_Avg']].dropna()

        if not wind_data.empty:
            ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white', bins=4)
            ax.set_title(month_names[month], fontsize=10, pad=10)  # Ajustar la posición del título
            #ax.set_rlabel_position(-50)  # Ajustar la posición de las etiquetas de los ticks r
            ax.yaxis.set_tick_params(direction='in')
            ax.xaxis.set_tick_params(direction='in')
        else:
            ax.set_title(month_names[month], fontsize=10, pad=10)  # Ajustar la posición del título
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Crear una leyenda global
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title="Velocidad (m/s)", title_fontsize=10, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=4, prop={'size': 10})

    # Ajustar el espaciado entre los subplots
    plt.subplots_adjust(hspace=0.9, wspace=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()





import seaborn as sns


def met_windrose_sns(wr_cmul):
    """
    Esta función toma un DataFrame y plotea rosas de viento en un subplot de 4x3 para cada mes de 2024 utilizando Seaborn para la configuración de estilo.
    """
    # Configurar el estilo de Seaborn
    sns.set(style="whitegrid")

    # Filtrar datos para el año 2024
    wr_cmul_2024 = wr_cmul[wr_cmul['yyyy-mm-dd HH:MM:SS'].dt.year == 2024]

    # Crear una figura con subplots 4x3
    fig, axs = plt.subplots(4, 3, figsize=(18, 24), subplot_kw={'projection': 'windrose'})
    fig.suptitle('Rosas de Viento Mensuales para 2024', fontsize=16)

    # Nombres de los meses
    month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

    # Iterar sobre cada mes y crear una rosa de viento
    for month in range(1, 13):
        ax = axs[(month-1)//3, (month-1)%3]
        monthly_data = wr_cmul_2024[wr_cmul_2024['yyyy-mm-dd HH:MM:SS'].dt.month == month]
        wind_data = monthly_data[['WSpeed_Avg', 'WDir_Avg']].dropna()

        if not wind_data.empty:
            ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white', bins=4)
            ax.set_title(month_names[month], fontsize=10, pad=10)  # Ajustar la posición del título
            ax.set_rlabel_position(-22.5)  # Ajustar la posición de las etiquetas de los ticks r
            ax.tick_params(axis='y', direction='in', pad=-15)  # Colocar las etiquetas de los ticks r dentro
        else:
            ax.set_title(month_names[month], fontsize=10, pad=10)  # Ajustar la posición del título
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    # Crear una leyenda global
    #handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels, title="Velocidad (m/s)", title_fontsize=10, loc="center", bbox_to_anchor=(0.5, -0.05), ncol=4, prop={'size': 10})

    # Ajustar el espaciado entre los subplots
    plt.subplots_adjust(hspace=0.9, wspace=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



def met_windrose1(wr_cmul):
    """
    Esta función toma un DataFrame y plotea rosas de viento en un subplot de 4x3 para cada mes de 2024.
    """
    # Filtrar datos para el año 2024
    wr_cmul_2024 = wr_cmul[wr_cmul['yyyy-mm-dd HH:MM:SS'].dt.year == 2024]

    # Crear una figura con subplots 4x3
    fig, axs = plt.subplots(4, 3, figsize=(18, 12), subplot_kw={'projection': 'windrose'})
    fig.suptitle('Rosas de Viento Mensuales para 2024', fontsize=16)

    # Nombres de los meses
    month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}

    # Iterar sobre cada mes y crear una rosa de viento
    for month in range(1, 13):
        ax = axs[(month-1)//3, (month-1)%3]
        monthly_data = wr_cmul_2024[wr_cmul_2024['yyyy-mm-dd HH:MM:SS'].dt.month == month]
        wind_data = monthly_data[['WSpeed_Avg', 'WDir_Avg']].dropna()

        if not wind_data.empty:
            ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], normed=True, opening=0.8, edgecolor='white', bins=4)
            ax.set_title(month_names[month], fontsize=12)
            ax.legend(title="Velocidad (m/s)", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
            ax.yaxis.set_ticks_position('inside')
            ax.yaxis.set_tick_params(direction='in', length=6)

        else:
            ax.set_title(month_names[month], fontsize=12)
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()











