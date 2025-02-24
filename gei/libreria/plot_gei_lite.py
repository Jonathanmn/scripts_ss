''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''


from picarro import *


folder_path= '/home/jmn/server_gei'
output_folder= '/home/jmn/picarro_data'


gei=read_raw_gei_folder(folder_path)


gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
print('aplicando flags a especies mpv')


gei=flags_species_1min(gei)

print('flags de MPV')
gei=flags_mpv(gei,'CO2_Avg','CH4_Avg','CO_Avg')
gei=correccion_utc(gei, 'Time')
gei=filter_sd(gei,num_sd=2)
plot_avg_sd_month(gei)



#gei_l0(gei,output_folder)
