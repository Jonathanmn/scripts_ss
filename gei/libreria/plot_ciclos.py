from picarro import *
from picarro_ciclos import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path ='./DATOS/gei/L1/minuto/2024'
folder_pathb='./DATOS/gei/L1b/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])





geib=copy_and_rename_columns(geib)
gei= copy_and_rename_columns(gei)

'''

'''

#plot_comparacion(('9 16h',gei_dia),('9 16 b',gei_9_16_b), column='CO2_Avg')

plot_intervalos_subplot_4x1(gei,geib, column='CO2_Avg', intervalos=[('00:00','23:59'),('19:00', '23:59'),('00:00','05:00'),('09:00', '16:00')])
#plot_comparacion(('19-23h', gei_nocturno),('00-05h',gei_0_5am), ('09-16h', gei_dia),('24h',gei24h), column='CO2_Avg')