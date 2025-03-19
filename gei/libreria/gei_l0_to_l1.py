'''Este script va a a tratar de dejar un archivo l0 limpio con flags y umbrales aplicados'''

from picarro import *


#folder_path = '/home/gei/scripts_j/raw'
#output_folder = '/home/gei/scripts_j/l0'


output_folder = '/home/jmn/L1_umbrales'


folder_path = '/home/jmn/l0-1/minuto'


gei = read_raw_lite(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])


gei = umbrales_sd(gei)

print('estamos guardando')
save_gei_l1_minuto(gei,output_folder)
save_gei_l1_hora(gei,output_folder)


