

''' este script pasa de minutos a hora el archivo l2   '''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


file_path = '/content/drive/Othercomputers/Mi portátil/Servicio Social/data/l2/2024-10-cmul_minuto_L2.csv'
cmul = pd.read_csv(file_path, header=6,encoding='ISO-8859-1')


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
agg_funcs = {col: 'mean' for col in cmul.columns if col not in ['mm', 'deg']}  
agg_funcs['mm'] = 'sum'
agg_funcs['deg'] = mean_wind_direction 

cmul_hora = cmul.resample('h').agg(agg_funcs)[cmul.columns]

