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



def ciclo_1d_avg(ciclo_filtrado):

    #ciclo_filtrado['Time']=ciclo_filtrado['Time'] - timedelta(hours=5)
    ciclo_filtrado = ciclo_filtrado.set_index('Time')
    #resampleo por dia
    ciclo_dia = ciclo_filtrado.resample('1D').agg(['mean', 'std'])
    
    # Rename columns
    ciclo_dia.columns = ['_'.join(col).replace('_mean', '_Avg').replace('_std', '_SD') for col in ciclo_dia.columns]
    ciclo_dia = ciclo_dia.reset_index()
 
    return ciclo_dia



def intervalo_horas(df, h0, hf):
    """
    filtra en que intervalo de horas (hh:mm) quieres mantener en el df, df=dataframe h0=hora inicial, hf=hora final

    ejemplo ciclo_9_16h=intervalo_horas(ciclo_9_16h,'09:00','16:00')

    """
    df = df.set_index('Time')
    df = df.between_time(h0, hf).reset_index()
    return df






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




def plot_comparacion(*dfs, column='CO2_Avg'):
    plt.figure(figsize=(10, 6))
    
    for i, df in enumerate(dfs):
        if column in df.columns:
            plt.plot(df['Time'], df[column], label=f' {df}')
        else:
            print(f"Column '{column}' not found in DataFrame {i+1}")
    
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(f'Comparison of {column} across DataFrames')
    plt.legend()
    plt.show()



def plot_comparacion(*dfs, column='CO2_Avg'):
    month_names = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }
    
    plt.figure(figsize=(10, 6))
    
    for df in dfs:
        if isinstance(df, tuple) and len(df) == 2:
            df_name, df_data = df
            if column in df_data.columns:
                plt.plot(df_data['Time'], df_data[column], label=df_name)
            else:
                print(f"Column '{column}' no se encontro en '{df_name}'")
        else:
            print("se deben meter valores en tupla ('df_name', df_data)")
    
 
    plt.ylabel('CO$_{2}$ ppm')
    plt.title('Comparación de intervalos de tiempo de CO$_{2}$')
    plt.legend()
    
    # Set month names as x-tick labels
    months = dfs[0][1]['Time'].dt.month.unique()
    month_starts = [dfs[0][1][dfs[0][1]['Time'].dt.month == month]['Time'].iloc[0] for month in months]
    month_labels = [month_names[month] for month in months]

    plt.xticks(month_starts, month_labels, rotation=45)
    plt.grid()
    plt.show()








def plot_intervalos_subplot_4x1(df1, df2, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')]):
    """
    Esta función toma dos DataFrames y una lista de intervalos de tiempo, filtra los datos según los intervalos,
    aplica ciclo_1d_avg y plotea los resultados en un subplot de 4x1.
    """
    fig, axs = plt.subplots(4, 1, figsize=(6, 10), sharex=True)

    '''
    df2_interval_full = intervalo_horas(df2, intervalos[0][0], intervalos[0][1])
    df2_avg_full = ciclo_1d_avg(df2_interval_full)
    df2_monthly_avg = df2_avg_full.set_index('Time').resample('ME').mean().reset_index()
    df2_monthly_avg['Time'] = df2_monthly_avg['Time'] + pd.offsets.MonthBegin(1) - pd.offsets.Day(15)
'''

    for i, (h0, hf) in enumerate(intervalos):
        df1_interval = intervalo_horas(df1, h0, hf)
        df2_interval = intervalo_horas(df2, h0, hf)

        df1_avg = ciclo_1d_avg(df1_interval)
        df2_avg = ciclo_1d_avg(df2_interval)

        axs[i].plot(df1_avg['Time'], df1_avg[column], label='L1', color='orange', alpha=1)
        axs[i].plot(df2_avg['Time'], df2_avg[column], label='L1b', color='#1062b4',alpha=1)

        df2_monthly_avg = df2_avg.set_index('Time').resample('ME').mean().reset_index()
        df2_monthly_avg['Time'] = df2_monthly_avg['Time'] + pd.offsets.MonthBegin(1) - pd.offsets.Day(15)
        
        axs[i].scatter(df2_monthly_avg['Time'], df2_monthly_avg[column], color='red', label='Promedio Mensual', s=30, zorder=5)
        axs[i].plot(df2_monthly_avg['Time'], df2_monthly_avg[column], color='red', linestyle='--', linewidth=1, zorder=4)
        axs[i].set_title(f'Horario:{h0}-{hf}', fontsize=10)
        axs[i].set_ylabel('CO$_{2}$ ppm')
        #axs[i].legend(loc='upper right',fontsize='x-small')
        axs[i].grid(True)
        #axs[i].set_ylim(405, 580)

    axs[-1].set_xlabel('2024')
    fig.suptitle('Promedios diarios de CO$_{2}$ para diferentes horarios', fontsize=14)
    # Set month names as x-tick labels
    months = df1['Time'].dt.month.unique()
    month_starts = [df1[df1['Time'].dt.month == month]['Time'].iloc[0] for month in months]
    month_names = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
    month_labels = [month_names[month] for month in months]

    for ax in axs:
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, rotation=45)
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Crear una leyenda única para todos los subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small', bbox_to_anchor=(0.5, 0.95))



    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()