'''Este script va a a tratar de dejar un archivo l0 limpio con flags y umbrales aplicados'''

from picarro_l0_server import *


#folder_path = '/home/gei/scripts_j/raw'
#output_folder = '/home/gei/scripts_j/l0'

folder_path = '/home/jmn/server_gei'
output_folder = '/home/jmn/gei-l0'


gei = read_raw_gei_folder(folder_path, 'Time')

gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
#aplicamos species y pos de valvula + conteo de archivos validos

gei=flags_species_1min(gei)

#revisamos que los timestamps sean exactos


#se aplica la correcion -6h - 170s de valvula 
gei=correccion_utc(gei, 'Time')
gei=timestamp_l0(gei,'Time')



save_gei_l0(gei,output_folder)