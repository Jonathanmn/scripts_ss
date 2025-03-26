
from windrose_lib import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

''' folders laptop 
folder_cmul = '/home/jmn/DATA/met/L2/hora' 
folder_gei = '/home/jmn/L1/minuto/2024'
'''
#folders lab pc
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





plot_wr_timeseries(gei_met, column='CO2_Avg')
