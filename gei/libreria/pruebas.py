import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from picarro import *

from picarro_clean import *

# Cargar los datos
#folder_path = '/home/jmn/l0/minuto/2024/07'
#/home/jonathan_mn/server_gei/minuto/2024/06
#folder_path = '/home/jonathan_mn/clean_prueba'

folder_path = '/home/jmn/l0-1/minuto/2024/02'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])



# Example usage
updated_gei = interactive_plot(gei)



#plot_hourly_subplots(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')




#diurno(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')

'''
gei_clean = clean_plotly_gei(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
plot_scatter(gei_clean, 'CH4_Avg')


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