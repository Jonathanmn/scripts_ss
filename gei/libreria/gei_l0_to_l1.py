'''Este script va a a tratar de dejar un archivo l0 limpio con flags y umbrales aplicados'''

from picarro import *


#folder_path = '/home/gei/scripts_j/raw'
#output_folder = '/home/gei/scripts_j/l0'


output_folder = '/home/jmn/L1_umbrales'




'''
folder_path = '/home/jmn/l0-1/minuto'
gei = read_raw_lite(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])

'''

folder_path = '/home/jmn/L1/minuto/2024'

#gei = read_raw_lite(folder_path, 'Time')
gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)

print(gei.head())
gei=umbrales_sd(gei)


print('estamos guardando')
save_gei_l1_minuto(gei,output_folder)
save_gei_l1_hora(gei,output_folder)

print('listo')


