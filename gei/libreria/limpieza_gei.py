''' este codigo va a limpiar los archivos l0 que faltaron'''
from picarro import *


folder_path= '/home/jmn/picarro_data/minuto/2024/02'

gei=read_raw_gei_folder(folder_path,'Time')


plot_1min_index(gei)


intervals_CO2 = [(23, 56), (700, 766)]
intervals_CH4 = [(23, 54), (500, 766)]
intervals_CO = [(16695,16725)]


#gei_clean=apply_nan_intervals(gei, intervals_CO2=None, intervals_CH4=None, intervals_CO=intervals_CO)

#plot_1min_index(gei_clean)




