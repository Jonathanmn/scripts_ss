import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from datetime import timedelta

''' estamos trabajando con archivos de picarro, por lo que se asume que los datos son de picarro,'''


''' Lectura de archivos y guardado de archivos   '''


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





def save_to(df, time_column, folder):
    """
    Guarda el DataFrame en un archivo .dat con el formato YYYY-MM_CMUL_L0.dat.

        time_column: La columna de tiempo en el DataFrame.
        folder_path: La carpeta donde se guardará el archivo.
    """
    year = df[time_column].dt.strftime('%Y').iloc[0]
    month = df[time_column].dt.strftime('%m').iloc[0]

    filename = f"{year}-{month}_CMUL_L0.dat"
    filepath = os.path.join(folder, filename)

    df.to_csv(filepath, sep=',', index=False)








''' Correcciones de hora y fecha   '''

def resample_to_1min(df, timestamp_column='timestamp'):
    """
    resamplea por un minuto
    """
    df = df.set_index(timestamp_column)
    resampled_df = df.resample('1min').mean()
    
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


''' Filtrado de datos   '''


def limpieza_intervalos(df, start_date, end_date):
  """Sets 'CO2_Avg' to None for a specific date range.

  Args:
    df: The Pandas DataFrame.
    start_date: The start date of the range (inclusive).
    end_date: The end date of the range (inclusive).
  """
  # Convert start_date and end_date to pandas Timestamp objects
  start_date = pd.Timestamp(start_date)
  end_date = pd.Timestamp(end_date)

  # Use loc to filter the DataFrame and set 'CO2_Avg' to None
  df.loc[(df['Time'] >= start_date) & (df['Time'] <= end_date), 'CO2_Avg'] = np.nan
  df.loc[(df['Time'] >= start_date) & (df['Time'] <= end_date), 'CH4_Avg'] = np.nan
  df.loc[(df['Time'] >= start_date) & (df['Time'] <= end_date), 'CO_Avg'] = np.nan

  return df


def apply_nan_intervals(df, intervals_CO2=None, intervals_CH4=None, intervals_CO=None):
    """
    Aplica np.nan a los intervalos especificados en las columnas 'CO2_Avg', 'CH4_Avg' y 'CO_Avg' del DataFrame.

    
    """
    if intervals_CO2:
        for start_index, end_index in intervals_CO2:
            df.loc[start_index:end_index, 'CO2_Avg'] = np.nan
            df.loc[start_index:end_index, 'CO2_SD'] = np.nan

    if intervals_CH4:
        for start_index, end_index in intervals_CH4:
            df.loc[start_index:end_index, 'CH4_Avg'] = np.nan
            df.loc[start_index:end_index, 'CH4_SD'] = np.nan

    if intervals_CO:
        for start_index, end_index in intervals_CO:
            df.loc[start_index:end_index, 'CO_Avg'] = np.nan
            df.loc[start_index:end_index, 'CO_SD'] = np.nan

    return df


# Función para eliminar valores antes y después de encontrar 2, 3, 4 o 5 en 'MPVPosition'
def clean_surrounding_values(df):
    # Identificar las filas donde 'MPVPosition' tiene valores 2, 3, 4 o 5
    condition = df['MPVPosition'].isin([2, 3, 4, 5])
    
    # Crear máscaras para las filas anteriores y posteriores
    previous_mask = condition.shift(5, fill_value=False)  # Fila anterior
    next_mask = condition.shift(-5, fill_value=False)     # Fila posterior
    
    # Combinar las máscaras con la condición actual
    mask = condition | previous_mask | next_mask
    
    # Reemplazar valores en las columnas específicas con NaN donde la máscara es True
    df.loc[mask, ['CO2_Avg', 'CH4_Avg', 'CO_Avg']] = np.nan
    
    return df



def filter_spikes_with_rolling_criteria(df, columns, window_size=10, threshold=1.5):

    for column in columns: 
      
        rolling_median = df[column].rolling(window=window_size, center=True).median()

        
        spikes = (df[column] > rolling_median * threshold) & (df[column].rolling(window=window_size, center=True).count() >= window_size)

       
        df[column] = df[column].copy() 
        df.loc[spikes, column] = np.nan 

    return df  




def filter_by_std(df, columns, num_stds=2):


    for column in columns:
        # Calculate the mean and standard deviation for the column
        mean = df[column].mean()
        std = df[column].std()

        # Calculate the upper and lower bounds for outlier detection
        upper_bound = mean + num_stds * std
        lower_bound = mean - num_stds * std

        # Replace values outside the bounds with np.nan
        df[column] = np.where(
            (df[column] > upper_bound) | (df[column] < lower_bound),
            np.nan,
            df[column]
        )

    return df






''' Flags de MPV y especies   '''


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















''' Ploteos    '''


def plot_comparacion(df1, df2, columns1, columns2, time_column1='Time', time_column2='Time'):

    num_plots = len(columns1)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    for i, (col1, col2) in enumerate(zip(columns1, columns2)):
        ax = axes[i]
        ax.plot(df1[time_column1], df1[col1], color='black', alpha=0.5, label=f'{col1} (df1)')
        ax.plot(df2[time_column2], df2[col2], color='red', alpha=0.8, label=f'{col2} (df2)')
        ax.set_ylabel(col1)
        ax.legend()

    plt.xlabel("Time") 
    plt.title('Comparacion series de versiones gei')
    plt.show()


def plot_comparacion_monthly(df1, df2, columns1, columns2, time_column1='Time', time_column2='Time'):

    # Ensure timestamp columns are in datetime format
    df1[time_column1] = pd.to_datetime(df1[time_column1])
    df2[time_column2] = pd.to_datetime(df2[time_column2])

    # Get unique years and months
    years = df1[time_column1].dt.year.unique()
    months = df1[time_column1].dt.month.unique()

    # Iterate through years and months
    for year in years:
        for month in months:
            # Filter data for the current month
            df1_month = df1[(df1[time_column1].dt.year == year) & (df1[time_column1].dt.month == month)]
            df2_month = df2[(df2[time_column2].dt.year == year) & (df2[time_column2].dt.month == month)]

            # Check if data exists for the current month
            if not df1_month.empty and not df2_month.empty:
                # Create subplots for the current month
                num_plots = len(columns1)
                fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots), sharex=True)
                if num_plots == 1:
                    axes = [axes]

                # Iterate through columns and plot data for the current month
                for i, (col1, col2) in enumerate(zip(columns1, columns2)):
                    ax = axes[i]
                    ax.plot(df1_month[time_column1], df1_month[col1], color='black', alpha=0.5, label=f'{col1} (df1)')
                    ax.plot(df2_month[time_column2], df2_month[col2], color='red', alpha=0.8, label=f'{col2} (df2)')
                    ax.set_ylabel(col1)
                    ax.legend()

                plt.xlabel("Time")
                plt.title(f'Comparacion series de tiempo - {year}-{month}')
                plt.show()


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




def plot_1min_index(df):
    """
    Ploteo de los valores promedio y desviacion estandar para gei.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # CO2
    ax1 = axes[0]
    ax1.plot(df.index, df['CO2_Avg'], label='CO2_Avg', color='#123456')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['CO2_SD'], label='CO2_SD', color='#F7883F', alpha=0.8)
    ax1.set_ylabel('CO2_Avg')
    ax1_twin.set_ylabel('CO2_SD')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('CO2 Concentration')

    # CH4
    ax2 = axes[1]
    ax2.plot(df.index, df['CH4_Avg'], label='CH4_Avg', color='#123456')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df.index, df['CH4_SD'], label='CH4_SD', color='#F7883F', alpha=0.8)
    ax2.set_ylabel('CH4_Avg')
    ax2_twin.set_ylabel('CH4_SD')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('CH4 Concentration')

    # CO
    ax3 = axes[2]
    ax3.plot(df.index, df['CO_Avg'], label='CO_Avg', color='#123456')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df.index, df['CO_SD'], label='CO_SD', color='#F7883F', alpha=0.8)
    ax3.set_ylabel('CO_Avg')
    ax3_twin.set_ylabel('CO_SD')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.set_title('CO Concentration')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()









def plot_avg_sd_month(df):
    """
    Plots average and standard deviation for CO2, CH4, and CO 
    separately for each month in the DataFrame.
    """
    
    # Ensure 'Time' column is datetime type
    df['Time'] = pd.to_datetime(df['Time'])

    # Group data by month
    for (year, month), group in df.groupby([df['Time'].dt.year, df['Time'].dt.month]):
        # Create a figure and subplots for each month
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'GEI {month} - {year} ')  # Set title for the month

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




def plot_raw(df, timestamp_column):


  fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)  

  
  axes[0].plot(df[timestamp_column], df['CO2_dry'], label='CO2_dry')
  axes[0].set_ylabel('CO2_dry')
  axes[0].legend()

  # Plot CH4_dry
  axes[1].plot(df[timestamp_column], df['CH4_dry'], label='CH4_dry')
  axes[1].set_ylabel('CH4_dry')
  axes[1].legend()

  # Plot CO
  axes[2].plot(df[timestamp_column], df['CO'], label='CO')
  axes[2].set_ylabel('CO')
  axes[2].legend()

  plt.xlabel(timestamp_column)  
  plt.suptitle('Raw Data Plot')  
  plt.tight_layout()
  plt.show()


def plot_raw_grouped(df, timestamp_column):
  """
  Plots 'CO2_dry', 'CH4_dry', and 'CO' grouped by year and month.

  Args:
    df: The Pandas DataFrame containing the data.
    timestamp_column: The name of the column containing timestamps.
  """

  # Ensure timestamp column is in datetime format
  df[timestamp_column] = pd.to_datetime(df[timestamp_column])

  # Group data by year and month
  grouped_data = df.groupby([df[timestamp_column].dt.year, df[timestamp_column].dt.month])

  # Iterate through groups and create plots
  for (year, month), group_df in grouped_data:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot CO2_dry
    axes[0].plot(group_df[timestamp_column], group_df['CO2_dry'], label='CO2_dry')
    axes[0].set_ylabel('CO2_dry')
    axes[0].legend()

    # Plot CH4_dry
    axes[1].plot(group_df[timestamp_column], group_df['CH4_dry'], label='CH4_dry')
    axes[1].set_ylabel('CH4_dry')
    axes[1].legend()

    # Plot CO
    axes[2].plot(group_df[timestamp_column], group_df['CO'], label='CO')
    axes[2].set_ylabel('CO')
    axes[2].legend()

    plt.xlabel(timestamp_column)
    plt.suptitle(f'Raw Data Plot - Year: {year}, Month: {month}')
    plt.tight_layout()
    plt.show()





















def filtrar_sd_por_hora(df, columnas_a_filtrar, std):
    """
    Filtra un DataFrame basado en la media y desviación estándar por hora.
    
    Parámetros:
    - df: DataFrame original que contiene los datos.
    - columnas_a_filtrar: Lista de nombres de columnas numéricas a filtrar (ej. ['CO2_Avg', 'CH4_Avg', 'CO_Avg']).
    - std: Número de desviaciones estándar para identificar valores atípicos (default: 2).
    
    Retorna:
    - DataFrame filtrado.
    """
    # Convertir la columna 'timestamp' a formato datetime si no lo está
    df['Time'] = pd.to_datetime(df['Time'])

    # Crear una nueva columna que agrupe los datos por hora
    df['Time'] = df['Time'].dt.floor('H')  # Agrupa por hora (sin minutos ni segundos)

    # Calcular la media y desviación estándar por hora para las columnas numéricas
    stats = df.groupby('hour')[columnas_a_filtrar].agg(['mean', 'std'])

    # Aplanar las columnas del DataFrame de estadísticas
    stats.columns = ['_'.join(col) for col in stats.columns]

    # Unir las estadísticas al DataFrame original
    df = df.merge(stats, left_on='hour', right_index=True)

    # Filtrar los datos que estén dentro de `std` desviaciones estándar para cada columna numérica
    for col in columnas_a_filtrar:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        df = df[(df[col] >= df[mean_col] - std * df[std_col]) & (df[col] <= df[mean_col] + std * df[std_col])]

    # Eliminar columnas auxiliares si ya no son necesarias
    df = df.drop(columns=[col for col in df.columns if '_mean' in col or '_std' in col or col == 'hour'])

    return df










def plot_scatter(df, column):
    """
    Creates a scatter plot using the DataFrame index as the x-axis and the specified column as the y-axis.

    Args:
        df: The Pandas DataFrame containing the data.
        column: The name of the column to plot on the y-axis.
    """
    plt.figure(figsize=(10, 6))  # Adjust figure size if needed
    plt.scatter(df.index, df[column])  # Use df.index for x-axis
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.title(f"Scatter Plot of {column} vs. Index")
    plt.grid(True)
    plt.show()