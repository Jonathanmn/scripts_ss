'''Este script va a a tratar de dejar un archivo l0 limpio con flags y umbrales aplicados'''

from picarro import *


#folder_path = '/home/gei/scripts_j/raw'
#output_folder = '/home/gei/scripts_j/l0'


output_folder = '/home/jmn/gei-l1'


folder_path = '/home/jmn/l0-1'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])

save_gei_l1(gei,output_folder)