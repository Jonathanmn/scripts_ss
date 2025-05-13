

''' este script pasa de minutos a hora el archivo l2   '''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

file_path = '/home/jonathan_mn/Descargas/data/met/L2/minuto/2024-12-cmul_minuto_L2.csv'
cmul = pd.read_csv(file_path, header=6,encoding='ISO-8859-1')

first_timestamp = cmul['yyyy-mm-dd HH:MM:SS'].iloc[0]  
datetime_object = datetime.strptime(first_timestamp, '%Y-%m-%d %H:%M:%S')
year_month = datetime_object.strftime('%Y-%m')


cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
mes = cmul['yyyy-mm-dd HH:MM:SS'].dt.month_name().iloc[0]
year = cmul['yyyy-mm-dd HH:MM:SS'].dt.year.iloc[0]
year =int(year)
month = cmul['yyyy-mm-dd HH:MM:SS'].dt.month.iloc[0]
month_number = "{:02d}".format(int(month))




def mean_wind_direction(directions):
    radians = np.radians(directions)
    sin_mean = np.sin(radians).mean()
    cos_mean = np.cos(radians).mean()
    mean_direction = np.arctan2(sin_mean, cos_mean)
    mean_direction_degrees = np.degrees(mean_direction)




    return mean_direction_degrees % 360

cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
cmul = cmul.set_index('yyyy-mm-dd HH:MM:SS')
'''agg_funcs = {col: 'mean' for col in cmul.columns if col not in ['mm', 'deg']}  
agg_funcs['mm'] = 'sum'
agg_funcs['deg'] = mean_wind_direction  

cmul_hora = cmul.resample('h').agg(agg_funcs)[cmul.columns]
'''




cols_to_resample = ['°C', '%', 'm/s', 'deg.1', 'hPa', 'W/m^2'] 


agg_funcs = {col: 'mean' for col in cols_to_resample}  
agg_funcs['mm'] = 'sum' 
agg_funcs['deg'] = mean_wind_direction  
agg_funcs['m/s.1'] = 'max'  


cmul_hora = cmul[['°C', '%', 'm/s', 'm/s.1', 'deg', 'deg.1', 'mm', 'hPa', 'W/m^2']].resample('h').agg(agg_funcs)
orden_met = ['°C', '%', 'm/s', 'm/s.1', 'deg', 'deg.1', 'mm', 'hPa', 'W/m^2']
cmul_hora = cmul_hora[orden_met]

#guardado del archivo cvs 

descriptive_text = "Red Universitaria de Observatorios Atmosfericos (RUOA)\n" \
                       "Atmospheric Observatory Calakmul (cmul), Campeche\n" \
                       "Lat 18.5956 N, Lon 89.4137 W, Alt 275 masl\n" \
                       "Time UTC-6h\n" \
                       "\n" \
                      "TIMESTAMP,Temp_Avg,RH_Avg,WSpeed_Avg,WSpeed_Max,WDir_Avg,WDir_SD,Rain_Tot,Press_Avg,Rad_Avg\n"
  








#print(cmul.head())

filename = f"{year_month}-cmul_hora_L2.csv"


folder_path = "/home/jonathan_mn/Descargas/data/met/L2/hora"
file_path = os.path.join(folder_path, filename)


with open(file_path, 'w', encoding='ISO-8859-1') as f:
    f.write(descriptive_text)  
    cmul_hora.to_csv(f, index=False, encoding='ISO-8859-1')  


print('ya estuvo')



