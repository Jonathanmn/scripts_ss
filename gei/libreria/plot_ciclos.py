from picarro import *
from picarro_ciclos import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_path = '/home/jmn/L1/minuto/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


gei_nocturno= copy_and_rename_columns(gei)
gei_dia=copy_and_rename_columns(gei)

gei_nocturno['Time'] = pd.to_datetime(gei_nocturno['Time'])
gei_9_16 = gei_nocturno[((gei_nocturno['Time'].dt.hour >= 16) | (gei_nocturno['Time'].dt.hour <= 9))].copy().reset_index(drop=True)

gei_nocturno_19_5 = gei_nocturno[((gei_nocturno['Time'].dt.hour >= 19) | (gei_nocturno['Time'].dt.hour <= 5))].copy().reset_index(drop=True)
print(' ssssssssssssssssssssssssssssssssssssssssssssssss')
print(gei_nocturno_19_5)
print(' ssssssssssssssssssssssssssssssssssssssssssssssss')
#gei_nocturno_19_5['Time']=gei_nocturno_19_5['Time'] - timedelta(hours=5)



#se calcula el ciclo diurno
gei_9_16= ciclo_diurno_avg_19_05(gei_9_16)
gei_nocturno_19_5=ciclo_diurno_avg_19_05(gei_nocturno_19_5)
gei_dia=ciclo_diurno_avg_19_05(gei_dia)


print('dia')
print(gei_dia.head())

print('9-16 h')



print('nocturno')
print(gei_nocturno_19_5.head())


plt.plot(gei_nocturno_19_5['Time'],gei_nocturno_19_5['CO2_Avg'], color='blue')
plt.plot(gei_9_16['Time'],gei_9_16['CO2_Avg'],color='red')
plt.plot(gei_dia['Time'],gei_dia['CO2_Avg'], color='black', alpha=0.5)
plt.tight_layout()
plt.show()


