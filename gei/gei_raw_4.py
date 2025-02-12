import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import timedelta
#




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

def flags_species_1min(df):
    """
    argumentos: el dataframe y la columna de tiempo
    aplica un filtro donde conserva los valores solo en las especies correspondientes al gas y calcula promedio y desviacion por minuto de muestreo
    retorna el data frame por minuto y se renombran las variables de promedio (avg) y desviacion estandar (sd)  
    """

    df.loc[((df["species"] != 2) & (df["species"] != 3)), "CO2_dry"] = None
    df.loc[((df["species"] != 3)), "CH4_dry"] = None
    df.loc[((df["species"] != 1) & (df["species"] != 4)), "CO"] = None

    df = df.set_index('timestamp')

    agg_funcs = {
        'MPVPosition': 'mean',
        'CO2_dry': ['mean', 'std'],
        'CH4_dry': ['mean', 'std'],
        'CO': ['mean', 'std']
    }

    
    resampled_df = df.resample('1min').agg(agg_funcs)

     
    resampled_df.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col
                         for col in resampled_df.columns]
    resampled_df = resampled_df.rename(columns={
    
    'MPVPosition_mean': 'MPVPosition',
    'CO2_dry_mean': 'CO2_Avg',
    'CO2_dry_std': 'CO2_SD',
    'CH4_dry_mean': 'CH4_Avg',
    'CH4_dry_std': 'CH4_SD',
    'CO_mean': 'CO_Avg',
    'CO_std': 'CO_SD'
    })
    
    resampled_df = resampled_df.reset_index().rename(columns={'timestamp': 'Time'})

    return resampled_df

def correccion_utc(df, column_name):
  """
  se llama al data frame con el nombre de la columna con el timestamp ejemplo gei, 'Time'
  Entrega el tiempo con la correccion UTC -6h - 170 s que tarda en llegar a la valvula.
  """
  df[column_name] = df[column_name] - timedelta(hours=6) - timedelta(seconds=170)
  df[column_name] = df[column_name].dt.floor('min') 
  return df


def timestamp_l0(df, timestamp_column):
  """
  Completes a DataFrame's timestamp series by adding missing timestamps.

  Args:
    df: The Pandas DataFrame.
    timestamp_column: The name of the timestamp column (default: 'timestamp').

  Returns:
    pandas.DataFrame: The DataFrame with a complete timestamp series.
  """

  # Get the start and end dates for the month
  start_date = df[timestamp_column].min().floor('D').replace(day=1)  # Start of the month
  end_date = df[timestamp_column].max().floor('D') + pd.offsets.MonthEnd(0)  # End of the month

  # Generate the complete timestamp sequence
  complete_timestamps = pd.date_range(
      start=start_date, 
      end=end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1),  # Include 23:59:00 of the last day
      freq='1min'
  )

  # Create a DataFrame with the complete timestamps
  complete_df = pd.DataFrame({timestamp_column: complete_timestamps})

  # Merge the original DataFrame with the complete DataFrame
  df_complete = pd.merge(complete_df, df, on=timestamp_column, how='left')

  return df_complete


def flags_mvp(df):
  """
  Crea una columna llamada MVP_{posicion de MVP}
  por cada posicion MVP que encuentre. 
    """
  

  
  MPVcount = df['MPVPosition'].value_counts(dropna=True)
  df['MPVPosition'] = df['MPVPosition'].round()

  for value, count in MPVcount.items():
    if value != 0:  
      column_name = f'MVP_{value}'
      
      temp_df = df[df['MPVPosition'] == value][['CO2_dry', 'CH4_dry', 'CO']]
      
      temp_df = temp_df.rename(columns={
          'CO2_dry': f'{column_name}_CO2_dry',
          'CH4_dry': f'{column_name}_CH4_dry',
          'CO': f'{column_name}_CO'
      })
      
      df = pd.merge(df, temp_df, left_index=True, right_index=True, how='left')

  return df

def plot_gei_avg_sd(df):
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




folder_path = '/home/jonathan_mn/data'
gei=raw_gei_folder(folder_path)
gei_flags=flags_species_1min(gei)

print(gei_flags.head())
print('ya estuvo')
#plot_gei_avg_sd(gei_flags)
gei_l0=timestamp_l0(gei_flags,'Time')

print(gei_l0.head())

gei_utc=correccion_utc(gei_l0,'Time')

print(gei_utc.head())

plot_gei_avg_sd(gei_utc)
#flags_mvp(gei)
#print(gei.columns)


