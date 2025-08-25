from picarro import *
from picarro_clean import *
from picarro_horarios import *

''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path ='_files/gei/L1/minuto/2024/'
folder_pathb='_files/gei/L1b/minuto/2024/'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])





print(geib.columns)

'''

'''


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
intervalo_horas_gei(geib, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')])