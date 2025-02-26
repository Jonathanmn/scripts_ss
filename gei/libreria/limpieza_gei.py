''' este codigo va a limpiar los archivos l0 que faltaron'''
from picarro import *


folder_path= '/home/jmn/picarro_data/minuto/2024/01'

gei=read_raw_gei_folder(folder_path,'Time')


plot_1min_index(gei)


intervals_CO2 = [(23, 56), (700, 766)]
intervals_CH4 = [(23, 54), (500, 766)]
intervals_CO = [(23, 546), (600, 766)]

for start_index, end_index in intervals_CO2:
    gei.loc[start_index:end_index, 'CO2_Avg'] = np.nan

for start_index, end_index in intervals_CH4:
    gei.loc[start_index:end_index, 'CH4_Avg'] = np.nan

for start_index, end_index in intervals_CO:
    gei.loc[start_index:end_index, 'CO_Avg'] = np.nan


def apply_nan_intervals(df, intervals_CO2=None, intervals_CH4=None, intervals_CO=None):
    """
    Aplica np.nan a los intervalos especificados en las columnas 'CO2_Avg', 'CH4_Avg' y 'CO_Avg' del DataFrame.

    Args:
        df: El DataFrame a modificar.
        intervals_CO2: Lista de tuplas con los intervalos para 'CO2_Avg'.
        intervals_CH4: Lista de tuplas con los intervalos para 'CH4_Avg'.
        intervals_CO: Lista de tuplas con los intervalos para 'CO_Avg'.

    Returns:
        El DataFrame modificado.
    """
    if intervals_CO2:
        for start_index, end_index in intervals_CO2:
            df.loc[start_index:end_index, 'CO2_Avg'] = np.nan

    if intervals_CH4:
        for start_index, end_index in intervals_CH4:
            df.loc[start_index:end_index, 'CH4_Avg'] = np.nan

    if intervals_CO:
        for start_index, end_index in intervals_CO:
            df.loc[start_index:end_index, 'CO_Avg'] = np.nan

    return df

