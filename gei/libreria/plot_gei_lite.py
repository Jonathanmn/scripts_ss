''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''


from picarro import *


folder_path= '/home/jmn/plot_raw'

gei=raw_gei_folder(folder_path)
gei=umbrales_gei(gei, CO2_umbral=300, CH4_umbral=1.6)
print('ya amonos al mpv')
gei=flags_mpv(gei,'CO2_dry','CH4_dry','CO')
print('ya fue')
gei=flags_species_1min(gei)


plot_gei_avg_sd_monthly(gei)




