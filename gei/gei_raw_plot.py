import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import sys
from IPython.display import clear_output
#pip install ipython

folder_path = '/home/jonathan_mn/Descargas/data/gei/01'
file_paths = glob.glob(os.path.join(folder_path, '*.dat'))
gei_list = []

conteo=0

#se agregan todos los datos de raw
for file_path in file_paths:
    gei = pd.read_csv(file_path, delimiter="\s+")
    gei['timestamp'] = pd.to_datetime(gei['DATE'] + ' ' + gei['TIME'])
    gei_list.append(gei)
    conteo+=1

    clear_output(wait=True)
    
    print(f'archivos leidos:{conteo}',end="\r")
    sys.stdout.flush() 
   

print(f'archivos leidos: {conteo},graficando')


#unimos los dataframe de todos los archivos.
gei = pd.concat(gei_list, ignore_index=True)
gei = gei.sort_values(by=['timestamp'])
gei = gei.reset_index(drop=True)


#ploteo de los gases 


fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)


axes[0, 0].plot(gei['timestamp'], gei['CO'])
axes[0, 0].set_ylabel('CO')

axes[1, 0].plot(gei['timestamp'], gei['CO2'])
axes[1, 0].set_ylabel('CO2')

axes[1, 1].plot(gei['timestamp'], gei['CO2_dry'])
axes[1, 1].set_ylabel('CO2_dry')

axes[2,0].plot(gei['timestamp'], gei['CH4'])
axes[2, 0].set_ylabel('CH4')

axes[2, 1].plot(gei['timestamp'], gei['CH4_dry'])
axes[2, 1].set_ylabel('CH4_dry')


fig.delaxes(axes[0, 1])

plt.xticks(rotation=45)
plt.xlabel('Date')
plt.tight_layout()
plt.show()
