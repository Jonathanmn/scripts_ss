
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




#print(cmul.describe())

max_co2 = cmul['CO2_Avg'].max()
max_ch4 = cmul['CH4_Avg'].max()
max_co = cmul['CO_Avg'].max()
max_pm10 = cmul['PM10 Conc'].max()
max_pm25 = cmul['PM2.5 Conc'].max()



# por intervalos
intervals = {'CO2_Avg': (600, max_co2),'CH4_Avg': (2.2, max_ch4)}
    
    
#plot_windrose_subplots_intervalos(cmul, columns=['CO2_Avg', 'CH4_Avg'], intervals=intervals)


met_windrose(cmul, timestamp='Time',column='CO2_Avg')
#plot_wr_timeseries_date(cmul, columns=['CO2_Avg', 'CH4_Avg','PM10 Conc','PM2.5 Conc'], inicio='2024-08-25 23:00:00', fin='2024-08-26 00:00:00')# Para una sola columna con intervalo de fechas
#plot_wr_timeseries_plotly(cmul, columns=['CO2_Avg', 'CH4_Avg','PM10 Conc','PM2.5 Conc'], inicio='2024-08-25 23:00:00', fin='2024-08-26 00:00:00')# Para una sola columna con intervalo de fechas
