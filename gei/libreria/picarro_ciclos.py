import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta


def copy_and_rename_columns(df):

    # Copiar las columnas especificadas
    df_copy = df[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()
    
    # Renombrar las columnas
    df_copy.rename(columns={'CH4_Avg': 'CH4', 'CO2_Avg': 'CO2', 'CO_Avg': 'CO'}, inplace=True)
    
    return df_copy



def ciclo_diurno_avg_19_05(ciclo_filtrado):

    ciclo_filtrado['Time']=ciclo_filtrado['Time'] - timedelta(hours=5)
    ciclo_filtrado = ciclo_filtrado.set_index('Time')
    #resampleo por dia
    ciclo_dia = ciclo_filtrado.resample('1D').agg(['mean', 'std'])
    
    # Rename columns
    ciclo_dia.columns = ['_'.join(col).replace('_mean', '_Avg').replace('_std', '_SD') for col in ciclo_dia.columns]
    ciclo_dia = ciclo_dia.reset_index()
 

    
    return ciclo_dia


def plot_gei_nocturno_19_05(gei_nocturno, std_ch4=None, std_co2=None):
    gei_nocturno_2024 = gei_nocturno[gei_nocturno['Time'].dt.year == 2024]
    # estadistica 
    ch4_mean = gei_nocturno_2024['CH4_Avg'].mean()
    ch4_max = gei_nocturno_2024['CH4_Avg'].max()
    ch4_min = gei_nocturno_2024['CH4_Avg'].min()

    co2_mean = gei_nocturno_2024['CO2_Avg'].mean()
    co2_max = gei_nocturno_2024['CO2_Avg'].max()
    co2_min = gei_nocturno_2024['CO2_Avg'].min()


    month_names = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
    7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
    # solo datos de 2024


    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # CH4
    ax1.plot(gei_nocturno_2024['Time'], gei_nocturno_2024['CH4_Avg'], color='b', label='CH4_Avg')
    ax1.set_ylabel('CH4_Avg')
    ax1.tick_params(axis='y')

    # ch4 sd
    if std_ch4 is not None:
        ax1.set_title('CH4 avg y sd')
        ax2 = ax1.twinx()
        ax2.plot(gei_nocturno_2024['Time'], gei_nocturno_2024[std_ch4], color='r',alpha=0.7, label=std_ch4)
        ax2.set_ylabel(std_ch4, color='r')
        ax2.tick_params(axis='y', labelcolor='r')

    
    ax1.set_xlabel('CH4 avg')

    # Co2
    ax3.plot(gei_nocturno_2024['Time'], gei_nocturno_2024['CO2_Avg'], color='b', label='CO2_Avg')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('CO2_Avg')
    ax3.tick_params(axis='y')

    # co2 sd
    if std_co2 is not None:
        ax3.set_title('CO2 Avg y sd')
        ax4 = ax3.twinx()
        ax4.plot(gei_nocturno_2024['Time'], gei_nocturno_2024[std_co2], color='r', alpha=0.7,label=std_co2)
        ax4.set_ylabel(std_co2, color='r')
        ax4.tick_params(axis='y', labelcolor='r')

    
    ax3.set_xlabel('CO2 Avg')

    
    months = gei_nocturno_2024['Time'].dt.month.unique()
    month_starts = [gei_nocturno_2024[gei_nocturno_2024['Time'].dt.month == month]['Time'].iloc[0] for month in months]
    month_labels = [month_names[month] for month in months]

    ax3.set_xticks(month_starts)
    ax3.set_xticklabels(month_labels, rotation=45)


    fig.suptitle('Ciclo Nocturno  CO2 y CH4 (19 - 05 h) 2024', fontsize=20)
    # texto de estadisticas
    ax1.text(0.02, 0.95, f'Avg: {ch4_mean:.2f}\nMax: {ch4_max:.2f}\nMin: {ch4_min:.2f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))


    ax3.text(0.02, 0.95, f'Avg: {co2_mean:.2f}\nMax: {co2_max:.2f}\nMin: {co2_min:.2f}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

 
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()

