from picarro import *
from picarro_clean import *
from picarro_l0_server import *

'''se grafican datos de ciclo diurno mensual '''

folder_path = '/home/jmn/L1b/hora/2024'


folder_path2='/home/jmn/L1/hora/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

gei_b=read_L0_or_L1(folder_path2, 'yyyy-mm-dd HH:MM:SS', header=7)
gei_b=reverse_rename_columns(gei_b)
gei_b['Time']=pd.to_datetime(gei_b['Time'])


#plot_avg_sd_monthly(gei)


#plot_24h_anual_subplot(gei,CO2='CO2_Avg',start_month=1, end_month=12)



plot_24h_anual_subplot_comparacion(gei, gei_b, CO2='CO2_Avg', start_month=1, end_month=12)

#ciclo_diurno_mensual_matplot(gei, CO2='CO2_Avg', CH4='CH4_Avg',start_month=1, end_month=12)

