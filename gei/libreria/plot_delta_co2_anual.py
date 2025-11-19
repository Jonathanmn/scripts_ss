from picarro import *
from picarro_clean import *


'''se grafican datos delta de ciclos 24h '''

folder_path = 'DATOS Sensores/gei/L1b/minuto/2024'



gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])





#plot_24h_anual_subplot_comp_delta(gei, CO2='CO2_Avg', start_month=1, end_month=12)


# grafica maxm min y avg de CO2 y el delta  de cada mes


co2_delta=timeseries_delta_per_day(gei, CO2='CO2_Avg', start_month=1, end_month=12)

print(co2_delta)



