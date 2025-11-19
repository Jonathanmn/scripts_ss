from picarro import *
from picarro_clean import *
from picarro_horarios import *

''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path ='scripts_ss/_files/gei/L1/minuto/2023'
folder_pathb='scripts_ss/_files/gei/L1/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])





<<<<<<< HEAD
geib=reverse_rename_columns(geib)
gei= reverse_rename_columns(gei)
=======
print(geib.columns)
>>>>>>> ec4519cb90e79902d027db2384c69611a0690e84

'''

'''

<<<<<<< HEAD
#plot_comparacion(('9 16h',gei_dia),('9 16 b',gei_9_16_b), column='CO2_Avg')
plot_intervalos_subplot_4x1(gei, geib, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')])

#plot_comparacion(('19-23h', gei_nocturno),('00-05h',gei_0_5am), ('09-16h', gei_dia),('24h',gei24h), column='CO2_Avg')
=======

import matplotlib.pyplot as plt

def intervalo_horas_gei(df, column, intervalos=[]):
    """
    Plots the average of `column` for each hour in the specified intervals.
    Each interval is plotted in a separate subplot (n x 1).
    The average is computed over all months (not separated by month).
    """
    n = len(intervalos)
    fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=False)  # <-- sharex=False
    if n == 1:
        axes = [axes]

    for idx, (start, end) in enumerate(intervalos):
        # Filter by hour interval
        mask = (
            (df['Time'].dt.strftime('%H:%M') >= start) &
            (df['Time'].dt.strftime('%H:%M') <= end)
            if start <= end else
            ((df['Time'].dt.strftime('%H:%M') >= start) | (df['Time'].dt.strftime('%H:%M') <= end))
        )
        df_interval = df[mask].copy()
        df_interval['hour'] = df_interval['Time'].dt.hour

        # Group by hour only, average over all months
        grouped = df_interval.groupby('hour')[column].mean().reset_index()
        axes[idx].plot(grouped['hour'], grouped[column], marker='o', label='Avg All Months')
        axes[idx].set_title(f'{start} - {end}')
        axes[idx].set_xlabel('Hour')
        axes[idx].set_ylabel(column)
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    plt.show()
#intervalo_horas_gei(geib, column='CO2_Avg', intervalos=[('00:00', '05:00'), ('09:00', '16:00'),('19:00', '23:59')])


def ciclo_1d_avg(ciclo_filtrado):

    #ciclo_filtrado['Time']=ciclo_filtrado['Time'] - timedelta(hours=5)
    ciclo_filtrado = ciclo_filtrado.set_index('Time')
    #resampleo por dia
    ciclo_dia = ciclo_filtrado.resample('1D').agg(['mean', 'std'])
    
    # Rename columns
    ciclo_dia.columns = ['_'.join(col).replace('_mean', '_Avg').replace('_std', '_SD') for col in ciclo_dia.columns]
    ciclo_dia = ciclo_dia.reset_index()
 
    return ciclo_dia





def plot_intervalos_subplot_4x1(df1, df2, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')]):
    """
    Esta función toma dos DataFrames y una lista de intervalos de tiempo, filtra los datos según los intervalos,
    aplica ciclo_1d_avg y plotea los resultados en un subplot de 4x1.
    """
    fig, axs = plt.subplots(4, 1, figsize=(6, 10), sharex=True)

    
    df2_interval_full = intervalo_horas(df2, intervalos[0][0], intervalos[0][1])
    df2_avg_full = ciclo_1d_avg(df2_interval_full)
    df2_monthly_avg = df2_avg_full.set_index('Time').resample('ME').mean().reset_index()
    df2_monthly_avg['Time'] = df2_monthly_avg['Time'] + pd.offsets.MonthBegin(1) - pd.offsets.Day(15)

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


plot_intervalos_subplot_4x1(gei, geib, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')])
>>>>>>> ec4519cb90e79902d027db2384c69611a0690e84
