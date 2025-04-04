
from picarro import *
from picarro_clean import *
#folder_path = '/home/jmn/L1/minuto/2024'

folder_path = './DATOS/gei/L1/minuto/2024'

#gei = read_raw_lite(folder_path, 'Time')
gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)

print(gei.head())
#gei=umbrales_sd(gei)


clean_plotly_gei(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')