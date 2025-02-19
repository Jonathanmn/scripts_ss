import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
''' estamos haciendo prubeas de git'''


''' Lectura de archivos y guardado de archivos   '''


def raw_gei_folder(folder_path):
    """
    Lee los archivos .dat del folder donde se encuentran los archivos raw con subcarpetas
    retorna un solo data frame con los datos de toda la carpeta .
    """
    dataframes = []

    for file_path in Path(folder_path).rglob('*.dat'):
        df = pd.read_csv(file_path, delimiter=r',')
        dataframes.append(df)

    gei = pd.concat(dataframes, ignore_index=True)
    gei['timestamp'] = pd.to_datetime(gei['timestamp'])
    gei = gei.sort_values(by='timestamp').reset_index(drop=True)


    return gei





''' Correcciones de hora y fecha   '''


def correccion_utc(df, column_name):
  """
  se llama al data frame con el nombre de la columna con el timestamp ejemplo gei, 'Time'
  Entrega el tiempo con la correccion UTC -6h - 170 s que tarda en llegar a la valvula.
  """
  df[column_name] = df[column_name] - timedelta(hours=6) - timedelta(seconds=170)
  df[column_name] = df[column_name].dt.floor('min') 
  return df


def timestamp_l0(df, timestamp_column):


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





''' Pruebas de de Flags   '''

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


def flags_mpv(df,CO2,CH4,CO):
  df['MPVPosition'] = df['MPVPosition'].fillna(0).astype(int)

  df['MPVPosition'] = df['MPVPosition'].round().astype(int)

  MPVcount = df['MPVPosition'].value_counts(dropna=False)
 
  for value, count in MPVcount.items():
    if value != 0 and value != 1:
      column_name = f'MPV_{value}'

      temp_df = df[df['MPVPosition'] == value][[CO2,CH4,CO]]
      
      temp_df = temp_df.rename(columns={
          CO2: f'{column_name}_CO2',
          CH4: f'{column_name}_CH4',
          CO: f'{column_name}_CO'
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















''' Ploteos    '''

def plot_1min_avg_sd(df):
    """
    Ploteo de los valores promedio y desviacion estandar para gei.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # CO2
    ax1 = axes[0]
    ax1.plot(df['Time'], df['CO2_Avg'], label='CO2_Avg', color='#123456')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Time'], df['CO2_SD'], label='CO2_SD', color='#F7883F', alpha=0.8)
    ax1.set_ylabel('CO2_Avg')
    ax1_twin.set_ylabel('CO2_SD')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('CO2 Concentration')

    # CH4
    ax2 = axes[1]
    ax2.plot(df['Time'], df['CH4_Avg'], label='CH4_Avg', color='#123456')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Time'], df['CH4_SD'], label='CH4_SD', color='#F7883F',alpha=0.8)
    ax2.set_ylabel('CH4_Avg')
    ax2_twin.set_ylabel('CH4_SD')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('CH4 Concentration')

    # CO
    ax3 = axes[2]
    ax3.plot(df['Time'], df['CO_Avg'], label='CO_Avg', color='#123456')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['Time'], df['CO_SD'], label='CO_SD', color='#F7883F', alpha=0.8)
    ax3.set_ylabel('CO_Avg')
    ax3_twin.set_ylabel('CO_SD')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.set_title('CO Concentration')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()






def plot_gei_avg_sd_monthly(df):
    """
    Plots average and standard deviation for CO2, CH4, and CO 
    separately for each month in the DataFrame.
    """
    
    # Ensure 'Time' column is datetime type
    df['Time'] = pd.to_datetime(df['Time'])

    # Group data by month
    for month, group in df.groupby(df['Time'].dt.month):
        # Create a figure and subplots for each month
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Month: {month}')  # Set title for the month

        # Plot CO2 data for the month
        ax1 = axes[0]
        ax1.plot(group['Time'], group['CO2_Avg'], label='CO2_Avg', color='#123456')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(group['Time'], group['CO2_SD'], label='CO2_SD', color='#F7883F', alpha=0.8)
        ax1.set_ylabel('CO2_Avg')
        ax1_twin.set_ylabel('CO2_SD')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.set_title('CO2 Concentration')

        # Plot CH4 data for the month
        ax2 = axes[1]
        ax2.plot(group['Time'], group['CH4_Avg'], label='CH4_Avg', color='#123456')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(group['Time'], group['CH4_SD'], label='CH4_SD', color='#F7883F', alpha=0.8)
        ax2.set_ylabel('CH4_Avg')
        ax2_twin.set_ylabel('CH4_SD')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.set_title('CH4 Concentration')

        # Plot CO data for the month
        ax3 = axes[2]
        ax3.plot(group['Time'], group['CO_Avg'], label='CO_Avg', color='#123456')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(group['Time'], group['CO_SD'], label='CO_SD', color='#F7883F', alpha=0.8)
        ax3.set_ylabel('CO_Avg')
        ax3_twin.set_ylabel('CO_SD')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.set_title('CO Concentration')

        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()
