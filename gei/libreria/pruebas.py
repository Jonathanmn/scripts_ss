from picarro import *
from picarro_ciclos import *




folder_path = '/home/jmn/L1/minuto/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


gei_nocturno= copy_and_rename_columns(gei)

print(gei_nocturno.head())

gei_nocturno['Time'] = pd.to_datetime(gei_nocturno['Time'])
gei_nocturno = gei_nocturno[((gei_nocturno['Time'].dt.hour >= 19) | (gei_nocturno['Time'].dt.hour <= 5))].copy().reset_index(drop=True)

print(gei_nocturno.head())

#se calcula el ciclo diurno
gei_nocturno= ciclo_diurno_avg_19_05(gei_nocturno)

print(gei_nocturno.head())

plot_gei_nocturno_19_05(gei_nocturno)