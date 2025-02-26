''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''


from picarro import *


folder_path= '/home/jmn/server_gei'
output_folder= '/home/jmn/picarro_data'


gei=read_raw_gei_folder(folder_path)
gei=correccion_utc(gei, 'timestamp')


#gei_raw1min=resample_to_1min(gei,timestamp_column='timestamp')


gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
print('aplicando flags a especies mpv')

gei_species=flags_species_1min(gei)
gei_species=timestamp_l0(gei_species,'Time')


'''Ploteo 
columns1=['CO2_Avg','CH4_Avg','CO_Avg']
columns2=['CO2_Avg','CH4_Avg','CO_Avg']

time_column1='Time'
time_column2='Time'

#plot_comparacion_monthly(gei_raw1min,gei_species, columns1, columns2, time_column1='Time', time_column2='Time')
plot_comparacion(gei_species,clean, columns1, columns2, time_column1='Time', time_column2='Time')


'''
save_gei_l0(gei_species,output_folder)
