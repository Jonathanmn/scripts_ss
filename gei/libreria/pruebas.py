import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from picarro import *

from picarro_clean import *
# Cargar los datos
#folder_path = '/home/jmn/l0/minuto/2024/07'
#/home/jonathan_mn/server_gei/minuto/2024/06
folder_path = '/home/jonathan_mn/clean_prueba'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])






gei_clean = clean_plotly_gei(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
plot_scatter(gei_clean, 'CH4_Avg')
folder= '/home/jonathan_mn'
#save_to(gei, 'Time', folder)


