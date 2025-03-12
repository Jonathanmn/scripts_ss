from picarro import *
from picarro_ciclos import *




folder_path = '/home/jmn/L1/minuto/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])







#gei=umbrales_sd(gei, CO2_umbral=0.2,CH4_umbral=0.002)
gei=umbrales_sd(gei)




plot_1min_avg_sd(gei)

output_folder='/home/jmn/L1_2'

'''
save_gei_l1_minuto(gei,output_folder)

save_gei_l1_hora(gei,output_folder)'''