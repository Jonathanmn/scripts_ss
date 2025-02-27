import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from picarro import *

from picarro_clean import *
# Cargar los datos
folder_path = '/home/jmn/l0/minuto/2024/07'
gei = read_raw_gei_folder(folder_path, 'Time')
gei['Time'] = pd.to_datetime(gei['Time'])


clean_plotly(gei, 'CH4_Avg')



#plot_scatter(gei, 'CH4_Avg')

