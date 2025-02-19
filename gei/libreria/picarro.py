import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def umbrales_gei(df, CO2_umbral=None, CH4_umbral=None, CO_umbral=None):
  """
 Aplica el umbral a las columnas 'CO2_dry', 'CH4_dry', y 'CO' del DataFrame.

  """
  if CO2_umbral is not None:
    df['CO2_dry'] = np.where(df['CO2_dry'] <= CO2_umbral, np.nan, df['CO2_dry'])
  if CH4_umbral is not None:
    df['CH4_dry'] = np.where(df['CH4_dry'] <= CH4_umbral, np.nan, df['CH4_dry'])
  if CO_umbral is not None:
    df['CO'] = np.where(df['CO'] <= CO_umbral, np.nan, df['CO'])
  return df


def flags_mvp(df,CO2,CH4,CO):
  df['MPVPosition'] = df['MPVPosition'].fillna(0).astype(int)

  df['MPVPosition'] = df['MPVPosition'].round().astype(int)

  MPVcount = df['MPVPosition'].value_counts(dropna=False)
 
  for value, count in MPVcount.items():
    if value != 0 and value != 1:
      column_name = f'MVP_{value}'

      temp_df = df[df['MPVPosition'] == value][[CO2,CH4,CO]]
      # Rename columns
      temp_df = temp_df.rename(columns={
          CO2: f'{column_name}_CO2_flag',
          CH4: f'{column_name}_CH4_flag',
          CO: f'{column_name}_CO_flag'
      })
      df = pd.merge(df, temp_df, left_index=True, right_index=True, how='left')

  df.loc[((df['MPVPosition'] != 0) & (df['MPVPosition'] != 1)), CO2] = None
  df.loc[((df['MPVPosition'] != 0) & (df['MPVPosition'] != 1)), CH4] = None
  return df

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
    
    # Apply conditions to set values to None based on size
    df.loc[(size_CO2 < 30), 'CO2_dry'] = None
    df.loc[(size_CH4 < 15), 'CH4_dry'] = None
    df.loc[(size_CO < 30), 'CO'] = None




    df = df.set_index('timestamp')

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

    resampled_df = resampled_df.reset_index().rename(columns={'timestamp': 'Time'})

    return resampled_df






