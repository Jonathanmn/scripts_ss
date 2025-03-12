from picarro import *
from picarro_ciclos import *




folder_path = '/home/jmn/L1/minuto/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])





def umbrales_sd(df, CO2_umbral=None, CH4_umbral=None):
  """
 Aplica el umbral a las columnas 'CO2_dry', 'CH4_dry', y 'CO' del DataFrame.

  """
  if CO2_umbral is not None:
    df['CO2_Avg'] = np.where(df['CO2_SD'] > CO2_umbral, np.nan, df['CO2_Avg'])
  if CH4_umbral is not None:
    df['CH4_Avg'] = np.where(df['CH4_SD'] > CH4_umbral, np.nan, df['CH4_Avg'])
  
  df['CO_Avg'] = np.where((df['CO_Avg'] > 0) & (df['CO_Avg'] <= 1), df['CO_Avg'], np.nan)

  return df

gei=umbrales_sd(gei, CO2_umbral=0.2,CH4_umbral=0.002)


plot_1min_avg_sd(gei)

