
from windrose_lib import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_met = './DATOS/met/L2/minuto'
folder_gei = './DATOS/gei/L1/minuto/2024' 
folder_t64= './DATOS/pm/L0/minuto'





gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)

t64 = t64_cmul(folder_t64)
met = met_cmul(folder_met)   


t64=t64[['Date & Time (Local)','PM10 Conc', 'PM2.5 Conc']]
met= met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]




cmul = merge_df(
    [met,gei,t64],
    ['yyyy-mm-dd HH:MM:SS', 'Time', 'Date & Time (Local)'])


print(cmul.columns)








# por intervalos
intervals = {'CO2_Avg': (500, 550),'CH4_Avg': (2.2, 2.4)}
    
    
#plot_windrose_subplots_intervalos(gei_met, columns=['CO2_Avg', 'CH4_Avg'], intervals=intervals)





# Para una sola columna
#plot_wr_timeseries_dynamic(gei_met, columns=['CO2_Avg'])

# Para múltiples columnas
#plot_wr_timeseries_dynamic(gei_met, columns=['CO2_Avg', 'CH4_Avg', 'CO_Avg'])


plot_wr_timeseries_date2(cmul, columns=['CO2_Avg', 'CH4_Avg'], inicio='2024-01-06 18:00:00', fin='2024-01-07 00:00:00')# Para una sola columna con intervalo de fechas

