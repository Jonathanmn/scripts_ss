'''se plotea linea por linea los documentos de GEI usando los flags de especies para tomar el valor '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import sys
from IPython.display import clear_output
#pip install ipython


def gei_1min(df, columns_to_resample):

  
  df = df.set_index('timestamp')
  df_resampled = df[columns_to_resample].resample('1min').mean()
  df_resampled = df_resampled.reset_index()

  return df_resampled








folder_path = '/home/jonathan_mn/12'
file_paths = glob.glob(os.path.join(folder_path, '*.dat'))
gei_list = []

conteo=0

#se agregan todos los datos de raw
for file_path in file_paths:
    gei = pd.read_csv(file_path, delimiter="\s+")
    gei['timestamp'] = pd.to_datetime(gei['DATE'] + ' ' + gei['TIME']).dt.floor('s')
    gei_list.append(gei)
    conteo+=1

    clear_output(wait=True)
    
    print(f'archivos leidos:{conteo}',end="\r")
    sys.stdout.flush() 
   

print(f'archivos leidos: {conteo} ...')


#unimos los dataframe de todos los archivos.
gei = pd.concat(gei_list, ignore_index=True)
gei = gei.sort_values(by=['timestamp'])
gei = gei.reset_index(drop=True)



# se filtran las variables por flag

flag_co2 = gei[(gei['species'] == 2) | (gei['species'] == 3)][['timestamp', 'species', 'CO2_dry']].dropna()
flag_ch4 = gei[(gei['species'] == 3)][['timestamp', 'species', 'CH4_dry']].dropna()
flag_co= gei[(gei['species'] == 1) | (gei['species'] == 4)][['timestamp', 'species', 'CO']].dropna()



flag_co2_1min = gei_1min(flag_co2, ['CO2_dry'])
flag_ch4_1min = gei_1min(flag_ch4, ['CH4_dry'])
flag_co_1min = gei_1min(flag_co, ['CO'])



def merge_gei(df1, df2, df3):

  
  df1 = df1.set_index('timestamp')
  df2 = df2.set_index('timestamp')
  df3 = df3.set_index('timestamp')

  
  merged_df = pd.merge(df1, df2, on='timestamp', how='outer')
  merged_df = pd.merge(merged_df, df3, on='timestamp', how='outer')

  merged_df = merged_df.reset_index()

  return merged_df


gei_resampled = merge_gei(flag_co2_1min, flag_ch4_1min , flag_co_1min)




#ploteo 

def plot_gei_data(gei_data):

  import matplotlib.pyplot as plt

  
  fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)  # 3 rows, 1 column

  
  axes[0].plot(merged_data['timestamp'], merged_data['CO2_dry'])
  axes[0].set_ylabel('CO2_dry')

  
  axes[1].plot(merged_data['timestamp'], merged_data['CH4_dry'])
  axes[1].set_ylabel('CH4_dry')

  
  axes[2].plot(merged_data['timestamp'], merged_data['CO'])
  axes[2].set_ylabel('CO')

  
  plt.xlabel('Timestamp')
  plt.tight_layout()

 
  plt.show()



plot_gei_data(gei_resampled)




