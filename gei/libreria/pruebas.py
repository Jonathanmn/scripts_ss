
from picarro import *
from picarro_clean import *


'''Lab

folder_path = '/home/jonathan_mn/l0-1/minuto/2024/03'


// local


folder_path = '/home/jmn/L1/minuto/2024/03'
'''

folder_path = '/home/jmn/L1/minuto/2024'


gei=read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei=reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


gei=custom_resample_and_group(gei, 'Time')



#ciclo_diurno_plottly_6(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')


#ciclo_diurno_mensual_matplot2(gei, 'CH4_Avg', 'CO2_Avg')


#ciclo_diurno_3(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
'''
save_data = input("revisamos la linea de tiempo? (yes/no): ")

if save_data.lower() == 'yes':

    plot_1min_avg_sd(gei)
else:
    print("Los datos no se han guardado.")




#plot_hourly_subplots(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')


'''




'''
gei_clean = clean_plotly_gei(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
plot_scatter(gei_clean, 'CH4_Avg')

limpieza usando el scatter 


save_data = input("¿Desea guardar los datos limpios? (yes/no): ")

if save_data.lower() == 'yes':
    folder = '/home/jmn'
    save_to(gei_clean, 'Time', folder)
    print("Datos guardados en la carpeta:", folder)
else:
    print("Los datos no se han guardado.")

#save_to(gei, 'Time', folder)  #home/jonathan_mn/clean_prueba

#home/jonathan_mn/clean_prueba
'''