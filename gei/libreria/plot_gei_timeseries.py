''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''


from picarro import *

#folder_path = '/home/jmn/L1/minuto/2024'

folder_path = '/home/jmn/L1_umbrales/L1/minuto/2024'

#gei = read_raw_lite(folder_path, 'Time')
gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)

print(gei.head())
#gei=umbrales_sd(gei)


# Llamada a la función plot_1min_avg con argumentos dinámicos, rango de meses y año
plot_1min_avg_month_scatter(gei, CO2=True, CH4=True, CO=True, start_month=1, end_month=12, year=2024)