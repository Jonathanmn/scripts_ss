'''Este script va a a tratar de dejar un archivo l0 limpio con flags y umbrales aplicados'''

from picarro_server import *


folder_path = '/home/gei/scripts_j/raw'
output_folder = '/home/gei/scripts_j/l0'


gei = read_raw_gei_folder(folder_path, 'Time')

gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
#aplicamos species y pos de valvula + conteo de archivos validos

gei=flags_species_1min(gei)

#revisamos que los timestamps sean exactos

gei=timestamp_l0(gei,'Time')
#se aplica la correcion -6h - 170s de valvula 
correccion_utc(gei, 'Time')




save_gei_l0(gei,output_folder)