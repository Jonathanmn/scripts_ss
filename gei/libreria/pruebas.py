from picarro import *
from picarro_clean import *

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import timedelta

folder_path = '/home/jmn/L1/minuto/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

gei_nocturno = gei[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()
gei_diario = gei[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()
gei_diario.rename(columns={'CH4_Avg': 'CH4', 'CO2_Avg': 'CO2', 'CO_Avg': 'CO'}, inplace=True)
gei_nocturno.rename(columns={'CH4_Avg': 'CH4', 'CO2_Avg': 'CO2', 'CO_Avg': 'CO'}, inplace=True)

gei_nocturno['Time'] = pd.to_datetime(gei_nocturno['Time'])

ciclo_filtrado = gei_nocturno[((gei_nocturno['Time'].dt.hour >= 19) | (gei_nocturno['Time'].dt.hour <= 5))].copy().reset_index(drop=True)





print(ciclo_filtrado.head())

ciclo_filtrado['Time'] = ciclo_filtrado['Time'] - timedelta(hours=5)

print(ciclo_filtrado.head())

ciclo_filtrado = ciclo_filtrado.set_index('Time')

ciclo_dia = ciclo_filtrado.resample('1D').agg(['mean', 'std'])
# Rename columns
ciclo_dia.columns = ['_'.join(col).replace('_mean', '_Avg').replace('_std', '_SD') for col in ciclo_dia.columns]
ciclo_dia = ciclo_dia.reset_index()

print(ciclo_dia.head())

print(ciclo_dia.columns)



# Lista de columnas de gases
gas_cols = ['CO2_Avg', 'CH4_Avg']

fig, axes = plt.subplots(len(gas_cols), 1, sharex=True, figsize=(10, 6))  # Crear subplots

# Diccionario para mapear los números de los meses a sus nombres
month_names = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
    7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}

for i, gas in enumerate(gas_cols):
    ax = axes[i]  # Obtener el subplot actual
    ax.plot(ciclo_dia['Time'], ciclo_dia[gas], label='Mean', color='blue')  # Graficar la media
    
    # Añadir una línea de tendencia
    x = mdates.date2num(ciclo_dia['Time'])
    y = ciclo_dia[gas]
    coefficients = np.polyfit(x, y, 1)  # 1 para tendencia lineal
    trend_line = np.poly1d(coefficients)  # Crear función de línea de tendencia
    ax.plot(ciclo_dia['Time'], trend_line(x), "--", color='green', label='Trend')

    ax.set_ylabel(gas, color='blue')  # Establecer la etiqueta del eje y para la media
    
    # Combinar leyendas
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels)

# Formatear el eje x para mostrar ticks mensuales
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))

# Ajustar las etiquetas de los ticks para mostrar los nombres de los meses
ax.set_xticklabels([month_names[int(tick.get_text())] for tick in ax.get_xticklabels()])

plt.xlabel('Mes')  # Establecer la etiqueta del eje x para toda la figura
plt.suptitle('Mean and Trend Line of Gases')  # Establecer el título general
plt.tight_layout()  # Ajustar el diseño para un mejor espaciado
plt.show()
