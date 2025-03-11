from picarro import *
from picarro_clean import *
from picarro_l0_server import *

'''se grafican datos de ciclo diurno mensual '''

folder_path = '/home/jmn/L1/minuto/2024'
output_folder= '/home/jmn/L2'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


#plot_avg_sd_monthly(gei)


ciclo_diurno_mensual_subplot_matplot(gei,CO2='CO2_Avg',start_month=1, end_month=12)

#ciclo_diurno_mensual_matplot(gei, CO2='CO2_Avg', CH4='CH4_Avg',start_month=1, end_month=12)

