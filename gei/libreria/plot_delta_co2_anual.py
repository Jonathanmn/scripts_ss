from picarro import *
from picarro_ciclos import *


folder_path='DATOS Sensores/gei/L1b/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])




print(gei.columns)


