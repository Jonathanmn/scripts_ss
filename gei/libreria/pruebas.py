
from picarro import *
from picarro_clean import *


'''Lab

folder_path = '/home/jonathan_mn/l0-1/minuto/2024/03'



// local
folder_path = '/home/jmn/l0-1/minuto/2024/05'


l1 /home/jmn/gei-l1/minuto/2023/10

'''

folder_path = '/home/jmn/gei-l1/minuto/2023/10'
try:
    gei = pd.read_csv(folder_path, delimiter=',', header=7, error_bad_lines=False, warn_bad_lines=True)
    gei['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(gei['yyyy-mm-dd HH:MM:SS'])
except pd.errors.ParserError as e:
    print(f"Error al leer el archivo CSV: {e}")


#gei = read_l1_gei_folder(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
#gei['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(gei['yyyy-mm-dd HH:MM:SS'])



'''
folder_path = '/home/jmn/gei-l1/minuto/2023/10'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])
'''

#ciclo_diurno_plottly_5(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')







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