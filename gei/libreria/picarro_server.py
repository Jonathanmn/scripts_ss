import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import timedelta

# JMN

''' ESTA LIBRERIA TIENE FUNCIONS PARA LEER, LIMPIAR Y GUARDAR LOS ARCHIVOS L0 DE PICARRO'''


def read_raw_gei_folder(folder_path, time):
    """
    Lee los archivos .dat del folder donde se encuentran los archivos raw con subcarpetas
    retorna un solo data frame con los datos de toda la carpeta .
    """
    dataframes = []

    for file_path in Path(folder_path).rglob('*.dat'):
        df = pd.read_csv(file_path, delimiter=r',')
        dataframes.append(df)

    gei = pd.concat(dataframes, ignore_index=True)
    gei[time] = pd.to_datetime(gei[time])
    gei = gei.sort_values(by=time).reset_index(drop=True)


    return gei



def save_gei_l0(df, output_folder):
    """
    guarda el archivo mensual en la carpeta minuto/YYYY/MM/YYYY-MM_CMUL_L0.dat.


    """
    df['Time'] = pd.to_datetime(df['Time'])
    for month, group in df.groupby(pd.Grouper(key='Time', freq='ME')):
        
        year = month.strftime('%Y')
        month_str = month.strftime('%m')

        
        subfolder_path = os.path.join(output_folder, 'minuto', year, month_str)
        os.makedirs(subfolder_path, exist_ok=True)

        
        filename = month.strftime('%Y-%m') + '_CMUL_L0.dat'
        filepath = os.path.join(subfolder_path, filename)

        
        group.to_csv(filepath, sep=',', index=False)


'''correccion de la zona horaria'''

def correccion_utc(df, column_name):
  """
  se llama al data frame con el nombre de la columna con el timestamp ejemplo gei, 'Time'
  Entrega el tiempo con la correccion UTC -6h - 170 s que tarda en llegar a la valvula.
  """
  df[column_name] = df[column_name] - timedelta(hours=6) - timedelta(seconds=170)
  df[column_name] = df[column_name].dt.floor('min') 
  return df

def timestamp_l0(df, timestamp_column):
  'se asegura que cada timestamp sea un minuto exacto'

  start_date = df[timestamp_column].min().floor('D').replace(day=1)  
  end_date = df[timestamp_column].max().floor('D') + pd.offsets.MonthEnd(0)  


  complete_timestamps = pd.date_range(
      start=start_date, 
      end=end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),  
      freq='1min'
  )
  complete_df = pd.DataFrame({timestamp_column: complete_timestamps})
  df_complete = pd.merge(complete_df, df, on=timestamp_column, how='left')

  return df_complete



''' Pruebas umbrales  '''

def umbrales_gei(df, CO2_umbral=None, CH4_umbral=None):
  """
 Aplica el umbral a las columnas 'CO2_dry', 'CH4_dry', y 'CO' del DataFrame.

  """
  if CO2_umbral is not None:
    df['CO2_dry'] = np.where(df['CO2_dry'] <= CO2_umbral, np.nan, df['CO2_dry'])
  if CH4_umbral is not None:
    df['CH4_dry'] = np.where(df['CH4_dry'] <= CH4_umbral, np.nan, df['CH4_dry'])
  df['CO'] = np.where((df['CO'] > 0) & (df['CO'] <= 200), df['CO'], np.nan)

  return df


''' FLAGS SPECIES , CONTEO DE ARCHIVOS VÁLIDOS Y POSCION DE VALUVLA'''

def flags_species_1min(df):
    """
    argumentos: el dataframe y la columna de tiempo
    aplica un filtro donde conserva los valores solo en las especies correspondientes al gas y calcula promedio y desviacion por minuto de muestreo
    retorna el data frame por minuto y se renombran las variables de promedio (avg) y desviacion estandar (sd)
    """

    df.loc[((df["species"] != 2) & (df["species"] != 3)), "CO2_dry"] = None
    df.loc[((df["species"] != 3)), "CH4_dry"] = None
    df.loc[((df["species"] != 1) & (df["species"] != 4)), "CO"] = None

    size_CO2 = df.groupby(pd.Grouper(key='timestamp', freq='1min'))['CO2_dry'].transform('size')
    size_CH4 = df.groupby(pd.Grouper(key='timestamp', freq='1min'))['CH4_dry'].transform('size')
    size_CO = df.groupby(pd.Grouper(key='timestamp', freq='1min'))['CO'].transform('size')
    
    
    df.loc[(size_CO2 < 30), 'CO2_dry'] = None
    df.loc[(size_CH4 < 15), 'CH4_dry'] = None
    df.loc[(size_CO < 30), 'CO'] = None




    df = df.set_index('Time')

    agg_funcs = {
        'MPVPosition': lambda x: x.mode()[0] if not x.empty else np.nan,
        'CO2_dry': ['mean', 'std'],
        'CH4_dry': ['mean', 'std'],
        'CO': ['mean', 'std']
    }


    resampled_df = df.resample('1min').agg(agg_funcs)


    resampled_df.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col
                         for col in resampled_df.columns]
    resampled_df = resampled_df.rename(columns={

    'MPVPosition_<lambda>': 'MPVPosition',
    'CO2_dry_mean': 'CO2_Avg',
    'CO2_dry_std': 'CO2_SD',
    'CH4_dry_mean': 'CH4_Avg',
    'CH4_dry_std': 'CH4_SD',
    'CO_mean': 'CO_Avg',
    'CO_std': 'CO_SD'
    })

    resampled_df = resampled_df.reset_index().rename(columns={'Time': 'Time'})
    resampled_df['CO2_MPVPosition'] = np.nan
    resampled_df['CH4_MPVPosition'] = np.nan
    resampled_df['CO_MPVPosition'] = np.nan

    mask = resampled_df['MPVPosition'].isin([0, 1])

    resampled_df.loc[~mask, 'CO2_MPVPosition'] = resampled_df.loc[~mask, 'CO2_Avg']
    resampled_df.loc[~mask, 'CH4_MPVPosition'] = resampled_df.loc[~mask, 'CH4_Avg']
    resampled_df.loc[~mask, 'CO_MPVPosition'] = resampled_df.loc[~mask, 'CO_Avg']

    resampled_df.loc[~mask, 'CO2_Avg'] = np.nan
    resampled_df.loc[~mask, 'CH4_Avg'] = np.nan
    resampled_df.loc[~mask, 'CO_Avg'] = np.nan
    resampled_df.loc[((resampled_df["MPVPosition"] != 0) & (resampled_df["MPVPosition"] != 1)), "CO2_SD"] = None
    resampled_df.loc[((resampled_df["MPVPosition"] != 0) & (resampled_df["MPVPosition"] != 1)), "CH4_SD"] = None
    return resampled_df



