
from windrose_lib import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

folder_met = './DATOS/met/L2/minuto'
folder_gei = './DATOS/gei/L1/minuto/2024' 




gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


met = met_cmul(folder_met)   

met_winddata = met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei_winddata = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]



gei_met=pd.concat([gei_winddata, met_winddata], axis=1, join='inner')

print (gei_met.columns)








#plot_windrose_subplots(gei_met, columns=['CO2_Avg', 'CH4_Avg'])


# Assuming `gei_met` is your DataFrame
intervals = {'CO2_Avg': (500, 550),'CH4_Avg': (2.2, 2.4)}
    
    

#plot_windrose_subplots_intervalos(gei_met, columns=['CO2_Avg', 'CH4_Avg'], intervals=intervals)





# Para una sola columna
#plot_wr_timeseries_dynamic(gei_met, columns=['CO2_Avg'])

# Para múltiples columnas
#plot_wr_timeseries_dynamic(gei_met, columns=['CO2_Avg', 'CH4_Avg', 'CO_Avg'])


plot_wr_timeseries_date(gei_met, columns=['CO2_Avg', 'CH4_Avg', 'CO_Avg'], inicio='2024-01-06 18:00:00', fin='2024-01-07 00:00:00')# Para una sola columna con intervalo de fechas

