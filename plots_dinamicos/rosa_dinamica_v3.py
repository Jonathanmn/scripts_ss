
from windrose_lib import *


''' Aqui va la direccion de los folder de MET y PM                   '''

'''

 hora
folder_cmul = '/home/jonathan_mn/Descargas/data/met/L2/hora' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/L0/hora'
min
folder_cmul = '/home/jonathan_mn/Descargas/data/met/L2/minuto' 
folder_t64 = '/home/jonathan_mn/Descargas/data/t64/L0/minuto'
'''



folder_cmul = '/home/jmn/DATA/met/L2/hora' 
folder_t64 = '/home/jmn/DATA/t64/L0/hora'

start_date1 = datetime(2024, 5, 15, 6, 0, 0)
end_date1 = datetime(2024, 5, 16, 0, 0, 0)





'''aqui se mandan a llamar las funciones '''
cmul = met_cmul(folder_cmul)
t64 = t64_cmul(folder_t64)
#se toma por fecha o toda la serie de tiempo 
cmul_winddata = cmul[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
PMdata = t64[['PM10 Conc', 'PM2.5 Conc']]
wr_all_time = pd.concat([cmul_winddata, PMdata], axis=1, join='inner')
wr_all_time['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(wr_all_time['yyyy-mm-dd HH:MM:SS'])
wr_per_date = wr_all_time[(wr_all_time['yyyy-mm-dd HH:MM:SS'] >= start_date1) & (wr_all_time['yyyy-mm-dd HH:MM:SS']<= end_date1)]


#funcion que se manda a llamar

''' La funcion rosa_pm, grafica la serie de tiempo y rosas de concentracion de pm 10 y 2.5  
 el agumento per_date o all_time significa graficar por periodo de fecha o toda la serie de tiempo'''

#rosa_pm(wr_per_date)
#rosa_pm(wr_all_time)


#met_windrose(wr_all_time)

met_windrose(wr_all_time)

