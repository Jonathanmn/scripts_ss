''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''


from picarro import *

#plot_raw
folder_path= '/home/jmn/server_gei'

gei=raw_gei_folder(folder_path)
gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
print('ya amonos al mpv')

print('ya fue')

gei=flags_species_1min(gei)

print('amonos a las flags')
gei=flags_mpv(gei,'CO2_Avg','CH4_Avg','CO_Avg')

plot_gei_avg_sd_monthly(gei)




