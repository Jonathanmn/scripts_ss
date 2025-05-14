import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#mensual
''' ESTE SCRIPT ESTA CODIFICADO PARA PASAR DE DATOS RAW A L0, AGREGANDO FECHAS FALTANTES AL ARCHIVO 1MIN.-DATA.TXT, separando datos por minuto y hora mensuales '''
#lectura de los datos crudos
t64_raw = pd.read_csv('/home/jonathan_mn/Descargas/data/t64/1MIN-DATA.txt', delimiter=',')
print(' ****** Leyendo archivo de PM T640 Mass Monitor *****')
raw_lines=len(t64_raw)
print(f'RAW tiene {raw_lines} lineas')

t64_raw['Date & Time (Local)'] = pd.to_datetime(t64_raw['Date & Time (Local)'])

start_date = pd.Timestamp(t64_raw['Date & Time (Local)'].min()).floor('D').replace(day=1)
#esta linea rellena los datos con nan si el archivo no tienen datos hasta el final de mes
#end_date = t64_raw['Date & Time (Local)'].max().replace(hour=23, minute=59, second=59) + pd.offsets.MonthEnd(0)
end_date= t64_raw['Date & Time (Local)'].max()
complete_range = pd.date_range(start=start_date, end=end_date, freq='1min')
t64_raw = t64_raw.set_index('Date & Time (Local)')
t64_raw = t64_raw.reindex(complete_range)
t64_raw = t64_raw.reset_index().rename(columns={'index': 'Date & Time (Local)'})


l0_lines=len(t64_raw)
new_lines= l0_lines - raw_lines

print(f'LO tiene: {l0_lines} lineas, se agregaron {new_lines} lineas')

#limpieza de datos _ agrega nuevos intervalos por indice si encuentras más datos por corregir

intervals=[(19459,22420),(57359,62433),(65700,66860),(78822,78864),(84807,86995),(126748,130750)]

for start, end in intervals:
    t64_raw.loc[start:end, ['  PM2.5 Conc', '  PM10 Conc']] = np.nan


# Resample por hora

t64_raw[ '  PWM válvula'] = t64_raw[ '  PWM válvula'].replace('-----', np.nan)
t64_raw[ '  Flujo en desviación T640x'] = t64_raw[ '  Flujo en desviación T640x'].replace('-----', np.nan)
t64_raw['  PWM válvula'] = pd.to_numeric(t64_raw['  PWM válvula'], errors='coerce')
t64_raw['  Flujo en desviación T640x'] = pd.to_numeric(t64_raw['  Flujo en desviación T640x'], errors='coerce')
t64_raw[ ' Date & Time (UTC)'] = pd.to_datetime(t64_raw[' Date & Time (UTC)'], format='%m/%d/%Y %H:%M:%S')
t64_raw = t64_raw.set_index('Date & Time (Local)')


t64_raw_hourly= t64_raw.resample('h').agg({
    ' Date & Time (UTC)': 'first',
    **{col: 'mean' for col in t64_raw.columns if col != ' Date & Time (UTC)'}
})

t64_raw = t64_raw.reset_index().rename(columns={'index': 'Date & Time (Local)'})
t64_hh=t64_raw_hourly.reset_index()




''' Guardado del documento de forma dinamica, revisa si ya existen los documentos los omite y si hay nueva
la actualiza                     '''


l0_path = '/home/jonathan_mn/Descargas/data/t64/LO'
min_path = os.path.join(l0_path, 'minuto')
hora_path = os.path.join(l0_path, 'hora')

if not os.path.exists(l0_path):
    os.makedirs(l0_path)

if not os.path.exists(min_path):
    os.makedirs(min_path)

if not os.path.exists(hora_path):
    os.makedirs(hora_path)

last_month = t64_raw['Date & Time (Local)'].dt.to_period('M').max()

print('\nAnalizando datos por minuto\n')
for year_month, group_data in t64_raw.groupby(pd.Grouper(key='Date & Time (Local)', freq='ME')):
    file_name = f"{year_month.year}_{year_month.month:02}_CMUL_PM.txt"
    file_path = os.path.join(min_path, file_name)

    if os.path.exists(file_path) and year_month != last_month:
        print(f'{file_name} minuto: Ya existe, buscando nueva información')
    
    else:
        group_data.to_csv(file_path, sep=',', index=False)     
        print(f' minuto: {file_name} Se ha actualizado correctamente')
    

print('\nAnalizando datos por hora\n')   
for year_month, group_data in t64_raw.groupby(pd.Grouper(key='Date & Time (Local)', freq='ME')):
    file_name_h = f"{year_month.year}_{year_month.month:02}_CMUL_PM_hora.txt"
    file_path_h = os.path.join(hora_path, file_name_h)


    
    if os.path.exists(file_path_h) and year_month != last_month:
        print(f'{file_name_h} hora:, Ya existe, buscando nueva información')

    else:
        
        group_data.to_csv(file_path_h, sep=',', index=False)
        print(f'hora: {file_name_h} Se ha actualizado correctamente ')
   




print(f'\nEl ultimo dato registrado es {end_date}')
