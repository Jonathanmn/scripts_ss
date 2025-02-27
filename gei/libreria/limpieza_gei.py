''' este codigo va a limpiar los archivos l0 que faltaron'''
from picarro import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

folder_path = '/home/jmn/picarro_data/02'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])

plot_1min_index(gei)



gei_clean = apply_nan_intervals(gei, intervals_CO2=None, intervals_CH4=intervals_CH4, intervals_CO=None)









