from windrose_lib import *


''' folders laptop 
folder_cmul = '/home/jmn/DATA/met/L2/hora' 
folder_gei = '/home/jmn/L1/minuto/2024'
'''
#folders lab pc
#folder_met = '/home/jonathan_mn/Descargas/data/met/L2/minuto'
folder_gei = './DATOS/gei/L1/minuto/2024' 

#folder_t64 = '/home/jmn/DATA/t64/L0/hora'


gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)

print(gei.head())

'''
gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])
t64 = t64_cmul(folder_t64)

met = met_cmul(folder_met)   

met_winddata = met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei_winddata = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]



gei_met=pd.concat([gei_winddata, met_winddata], axis=1, join='inner')
'''