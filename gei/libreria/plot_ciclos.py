from picarro import *
from picarro_ciclos import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path = '/home/jmn/L1/minuto/2024'
folder_pathb='/home/jmn/L1b/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])





geib=copy_and_rename_columns(geib)
gei= copy_and_rename_columns(gei)

'''
gei = gei.set_index('Time')
gei = gei.between_time('19:00', '23:59').reset_index()
'''

gei_nocturno=intervalo_horas(gei,'19:00','23:59')
gei_dia=intervalo_horas(gei,'09:00','16:00')

gei_0_5am=intervalo_horas(gei,'00:00','05:00')

gei_9_16_b=intervalo_horas(geib,'09:00','16:00')


print(gei_0_5am)



gei24h=ciclo_1d_avg(gei)
gei24hb=ciclo_1d_avg(geib)

gei_9_16_b=ciclo_1d_avg(gei_9_16_b)

gei_nocturno=ciclo_1d_avg(gei_nocturno)
gei_dia=ciclo_1d_avg(gei_dia)



gei_0_5am=ciclo_1d_avg(gei_0_5am)




plot_comparacion(('9 16h',gei_dia),('9 16 b',gei_9_16_b), column='CO2_Avg')

#plot_comparacion(('19-23h', gei_nocturno),('00-05h',gei_0_5am), ('09-16h', gei_dia),('24h',gei24h), column='CO2_Avg')