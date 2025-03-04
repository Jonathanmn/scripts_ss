
from picarro import *
from picarro_clean import *


folder_path = '/home/jonathan_mn/l0-1/minuto/2024/03'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])


ciclo_diurno_plottly_4(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')







#plot_hourly_subplots(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')




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