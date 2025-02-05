import pandas as pd
from pathlib import Path
import time
import os


'''
este script busca tener los datos de raw solo de columnas que nos importan, ojo sin aplicar 
la correccion de hora utc + tiempo de llegada a la valvula

'''
def save_data_to_txt(data_frame, file_path):

  data_frame.to_csv(file_path, sep=',', index=False) 

folder_path = '/home/jonathan_mn/Descargas/data/gei/raw_data/10'
folder_name = os.path.basename(folder_path)  

columns_to_read = ['DATE', 'TIME', 'species', 'MPVPosition', 'CO2_dry', 'CH4_dry', 'CO']
dataframes = []
start_time = time.time()

for file_path in Path(folder_path).rglob('*.dat'):
    
    df = pd.read_csv(file_path, delimiter=r'\s+')#, usecols=columns_to_read)
    
    dataframes.append(df)


gei = pd.concat(dataframes, ignore_index=True)
gei['timestamp'] = pd.to_datetime(gei['DATE'] + ' ' + gei['TIME'])
gei = gei.sort_values(by='timestamp').reset_index(drop=True)
gei = gei.drop(['DATE', 'TIME'], axis=1)

end_time = time.time()
time_taken = end_time - start_time
print(f"archivos leidos en: {time_taken:.2f}s")

MPVcount= gei['MPVPosition'].value_counts(dropna=False)
print(MPVcount)
print(gei.head())

gei = gei[['timestamp', 'species', 'MPVPosition', 'CO2_dry', 'CH4_dry', 'CO']]

output_file_path = f'/home/jonathan_mn/Descargas/data/gei/L0_v1/raw/MPVcount{folder_name}.txt'
MPVcount.to_csv(output_file_path, sep='\t', header=True, index=True)

save_data_to_txt(gei, f'/home/jonathan_mn/Descargas/data/gei/L0_v1/raw/{folder_name}_raw_mvp.txt')
