from picarro import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path ='DATOS Sensores/gei/L1/minuto/2024'
folder_pathb='DATOS Sensores/gei/L1b/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])


print(geib.info())  #48.3 mb

