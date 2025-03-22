import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import timedelta

''' Lectura de archivos y guardado de archivos   '''


def read_raw_gei_folder(folder_path):

    dataframes = []

    for file_path in Path(folder_path).rglob('*.dat'):
    
        df = pd.read_csv(file_path, delimiter=r'\s+')#, usecols=columns_to_read)
        dataframes.append(df)


    gei = pd.concat(dataframes, ignore_index=True)
    gei['Time'] = pd.to_datetime(gei['DATE'] + ' ' + gei['TIME'])
    gei = gei.sort_values(by='Time').reset_index(drop=True)
    gei = gei.drop(['DATE', 'TIME'], axis=1)

    return gei


def read_raw_lite(folder_path, time):
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

def read_L0_or_L1(folder_path,time, header=None):
    """
    Lee los archivos .dat del folder donde se encuentran los archivos raw con subcarpetas
    retorna un solo data frame con los datos de toda la carpeta .
    """
    dataframes = []

    for file_path in Path(folder_path).rglob('*.dat'):
        df = pd.read_csv(file_path, delimiter=r',', header=header)
        dataframes.append(df)

    gei = pd.concat(dataframes, ignore_index=True)
    gei[time] = pd.to_datetime(gei[time])
    gei = gei.sort_values(by=time).reset_index(drop=True)


    return gei








def reverse_rename_columns(df):
    """
    Renombra las columnas del DataFrame basado en su posición para evitar conflictos.
    """
    column_names = [
        'Time', 'MPVPosition', 'Height', 'CO2_Avg', 'CO2_SD', 
        'CH4_Avg', 'CH4_SD', 'CO_Avg', 'CO_SD', 
        'CO2_MPVPosition', 'CH4_MPVPosition', 'CO_MPVPosition'
    ]
    
    # Asegurarse de que el DataFrame tenga al menos tantas columnas como nombres en column_names
    if len(df.columns) >= len(column_names):
        df.columns = column_names + list(df.columns[len(column_names):])
    else:
        raise ValueError("El DataFrame no tiene suficientes columnas para renombrar.")
    
    return df

def save_gei_l0(df, output_folder):
    """
    guarda el archivo mensual en la carpeta minuto/YYYY/MM/YYYY-MM_CMUL_L0.dat.
    """
    for month, group in df.groupby(pd.Grouper(key='Time', freq='ME')):
        
        year = month.strftime('%Y')
        month_str = month.strftime('%m')

        
        subfolder_path = os.path.join(output_folder,'L0' 'minuto', year, month_str)
        os.makedirs(subfolder_path, exist_ok=True)

        
        filename = month.strftime('%Y-%m') + '_CMUL_L0.dat'
        filepath = os.path.join(subfolder_path, filename)

        
        group.to_csv(filepath, sep=',', index=False)





''' headers y variables de sitio'''


def header(file_path, sheet_name='station', header=0, encoding='utf-8'):
    """
    Lee un archivo Excel y retorna un DataFrame.
    
    Argumentos:
    file_path -- La ruta del archivo Excel a leer.
    sheet_name -- El nombre o índice de la hoja a leer (por defecto es la primera hoja).
    header -- La fila que se usará como encabezado (por defecto es la primera fila).
    encoding -- La codificación del archivo Excel (por defecto es 'utf-8').
    
    Retorna:
    Un DataFrame con los datos del archivo Excel.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    return df



def extract_variables(file_path, sheet_name='station', header=0, site_value=None):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
    row = df[df['Site'] == site_value]
    if not row.empty:
        name = row['Name'].values[0]
        state = row['State'].values[0]
        north = row['North'].values[0]
        west = row['West'].values[0]
        masl = row['MASL'].values[0]
        ut = row['UT'].values[0]
        return name, state, north, west, masl, ut
    else:
        return None









''' guardado de archivos  '''



def save_gei_l1_minuto(df, output_folder):
    """
    Guarda el archivo mensual en la carpeta minuto/YYYY/MM/YYYY-MM_CMUL_L1.dat.
    También guarda una versión resampleada a intervalos de 1 hora en la carpeta hora/YYYY/MM/YYYY-MM_CMUL_L1.dat.
    """
    df[['CO2_Avg', 'CO2_SD', 'CH4_Avg', 'CH4_SD', 'CO_Avg', 'CO_SD']].round(4)
    
    descriptive_text = (
        "Red Universitaria de Observatorios Atmosfericos (RUOA)\n"
        "Atmospheric Observatory Calakmul (cmul), Campeche\n"
        "Lat 18.5956 N, Lon 89.4137 W, Alt 275 masl\n"
        "Time UTC-6h + 170 S correction for height position \n"
        "Greenhouse Gas data\n"
        'Concentrations correspond to dry molar fractions\n'
        '\n'
        "Time,MPVPosition,Height,CO2_Avg,CO2_SD,CH4_Avg,CH4_SD,CO_Avg,CO_SD,CO2_MPVPosition,CH4_MPVPosition,CO_MPVPosition\n"
    )

    if 'Height' not in df.columns:
        df.insert(df.columns.get_loc('CO2_Avg'), 'Height', 16)

    # Renombrar las columnas
    df = df.rename(columns={
        'Time': 'yyyy-mm-dd HH:MM:SS',
        'MPVPosition': '1-5',
        'Height': 'm agl',
        'CO2_Avg': 'ppm',
        'CO2_SD': 'ppm',
        'CH4_Avg': 'ppb',
        'CH4_SD': 'ppb',
        'CO_Avg': 'ppb',
        'CO_SD': 'ppm',
        'CO2_MPVPosition': '1-5',
        'CH4_MPVPosition': '1-5',
        'CO_MPVPosition': '1-5'
    })

    for month, group in df.groupby(pd.Grouper(key='yyyy-mm-dd HH:MM:SS', freq='ME')):
        year = month.strftime('%Y')
        month_str = month.strftime('%m')

        subfolder_path = os.path.join(output_folder, 'L1', 'minuto', year, month_str)
        os.makedirs(subfolder_path, exist_ok=True)

        filename = month.strftime('%Y-%m') + '_CMUL_L1.dat'
        filepath = os.path.join(subfolder_path, filename)

        with open(filepath, 'w') as file:
            file.write(descriptive_text)
            group.to_csv(file, sep=',', index=False, mode='a')

def save_gei_l1_hora(df, output_folder):
    """
    Guarda el archivo mensual en la carpeta hora/YYYY/MM/YYYY-MM_CMUL_L1.dat.
    Resamplea los datos a intervalos de 1 hora, calculando la media para las columnas AVG y SD,
    y la moda para las columnas MPVPosition.
    """
    df[['CO2_Avg', 'CO2_SD', 'CH4_Avg', 'CH4_SD', 'CO_Avg', 'CO_SD']].round(4)
    
    descriptive_text = (
        "Red Universitaria de Observatorios Atmosfericos (RUOA)\n"
        "Atmospheric Observatory Calakmul (cmul), Campeche\n"
        "Lat 18.5956 N, Lon 89.4137 W, Alt 275 masl\n"
        "Time UTC-6h + 170 S correction for height position \n"
        "Greenhouse Gas data\n"
        'Concentrations correspond to dry molar fractions\n'
        '\n'
        "Time,MPVPosition,Height,CO2_Avg,CO2_SD,CH4_Avg,CH4_SD,CO_Avg,CO_SD,CO2_MPVPosition,CH4_MPVPosition,CO_MPVPosition\n"
    )

    if 'Height' not in df.columns:
        df.insert(df.columns.get_loc('CO2_Avg'), 'Height', 16)
    df = df.set_index('Time')
    resampled_df = df.resample('h').mean()

    resampled_df = resampled_df.reset_index().rename(columns={'Time': 'yyyy-mm-dd HH:MM:SS'})
    
    resampled_df = resampled_df.rename(columns={
        
        'MPVPosition': '1-5',
        'Height': 'm agl',
        'CO2_Avg': 'ppm',
        'CO2_SD': 'ppm',
        'CH4_Avg': 'ppb',
        'CH4_SD': 'ppb',
        'CO_Avg': 'ppb',
        'CO_SD': 'ppm',
        'CO2_MPVPosition': '1-5',
        'CH4_MPVPosition': '1-5',
        'CO_MPVPosition': '1-5'
    })

    print(resampled_df.head())

    for month, group in resampled_df.groupby(pd.Grouper(key='yyyy-mm-dd HH:MM:SS', freq='ME')):
        year = month.strftime('%Y')
        month_str = month.strftime('%m')

        subfolder_path = os.path.join(output_folder, 'L1', 'hora', year, month_str)
        os.makedirs(subfolder_path, exist_ok=True)

        filename = month.strftime('%Y-%m') + '_CMUL_L1.dat'
        filepath = os.path.join(subfolder_path, filename)

        with open(filepath, 'w') as file:
            file.write(descriptive_text)
            group.to_csv(file, sep=',', index=False, mode='a')




def guardar_raw_lite(df, time_column, folder):
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
    resamplea por un minuto todos los datos.
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
  df['CO'] = np.where((df['CO'] > 0) & (df['CO'] <= 0.8), df['CO'], np.nan)

  return df


def umbrales_sd(df, CO2_umbral=None, CH4_umbral=None):
  """
 Aplica el umbral a las columnas 'CO2_dry', 'CH4_dry', y 'CO' del DataFrame.

  """
  if CO2_umbral is not None:
    df['CO2_Avg'] = np.where(df['CO2_SD'] > CO2_umbral, np.nan, df['CO2_Avg'])
  if CH4_umbral is not None:
    df['CH4_Avg'] = np.where(df['CH4_SD'] > CH4_umbral, np.nan, df['CH4_Avg'])
  
  df['CO_Avg'] = np.where((df['CO_Avg'] > 0) & (df['CO_Avg'] <= 0.8), df['CO_Avg'], np.nan)
  df['CO_SD'] = np.where((df['CO_Avg'] > 0) & (df['CO_Avg'] <= 0.8), df['CO_SD'], np.nan)
  return df


''' Filtrado de datos   '''



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




def plot_1min_avg(df, CO2=True, CH4=True, CO=True):
    """
    Ploteo de los valores promedio para CO2, CH4 y CO según los argumentos proporcionados.
    """
    # Determinar el número de subplots necesarios
    num_plots = sum([CO2, CH4, CO])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    color_avg = '#0569cc'
    plot_index = 0

    # CO2
    if CO2:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO2_Avg'], label='CO2_Avg', color=color_avg)
        ax.set_ylabel('CO2_Avg')
        ax.legend(loc='upper left')
        ax.set_title('CO2 Concentration')
        plot_index += 1

    # CH4
    if CH4:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CH4_Avg'], label='CH4_Avg', color=color_avg)
        ax.set_ylabel('CH4_Avg')
        ax.legend(loc='upper left')
        ax.set_title('CH4 Concentration')
        plot_index += 1

    # CO
    if CO:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO_Avg'], label='CO_Avg', color=color_avg)
        ax.set_ylabel('CO_Avg')
        ax.legend(loc='upper left')
        ax.set_title('CO Concentration')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()


def plot_1min_avg_month(df, CO2=True, CH4=True, CO=True, start_month=None, end_month=None, year=None):
    """
    Ploteo de los valores promedio para CO2, CH4 y CO según los argumentos proporcionados.
    Permite seleccionar un rango de meses y un año para plotear.
    """

    Meses = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}




    # Filtrar por año si se especifica
    if year is not None:
        df = df[df['Time'].dt.year == year]

    # Filtrar por rango de meses si se especifica
    if start_month is not None and end_month is not None:
        df = df[(df['Time'].dt.month >= start_month) & (df['Time'].dt.month <= end_month)]

    # Determinar el número de subplots necesarios
    num_plots = sum([CO2, CH4, CO])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    color_avg = '#0569cc'
    plot_index = 0

    # CO2
    if CO2:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO2_Avg'], label='CO$_{2}$ Avg', color=color_avg)
        ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO$_{2}$')
        plot_index += 1

    # CH4
    if CH4:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CH4_Avg'], label='CH$_{4}$  Avg', color=color_avg)
        ax.set_ylabel('CH$_{4}$ (ppb)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CH$_{4}$')
        plot_index += 1

    # CO
    if CO:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO_Avg'], label='CO Avg', color=color_avg)
        ax.set_ylabel('CO (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO')




#cambios en el titulo 

    if start_month is not None and end_month is not None:
        if start_month == end_month:
            month_str = Meses.get(start_month, f'{start_month}')
        else:
            month_str = f'{Meses.get(start_month, f"{start_month}")}-{Meses.get(end_month, f"{end_month}")}'
    else:
        month_str = ''

    plt.suptitle(f'Concentración de GEI, Estación Calakmul {month_str} {year}' if year is not None else f'Concentración de GEI, Estación Calakmul')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()



def plot_1min_avg_month_scatter(df, CO2=True, CH4=True, CO=True, start_month=None, end_month=None, year=None):
    """
    Ploteo de los valores promedio para CO2, CH4 y CO según los argumentos proporcionados.
    Permite seleccionar un rango de meses y un año para plotear.
    """

    Meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Filtrar por año si se especifica
    if year is not None:
        df = df[df['Time'].dt.year == year]

    # Filtrar por rango de meses si se especifica
    if start_month is not None and end_month is not None:
        df = df[(df['Time'].dt.month >= start_month) & (df['Time'].dt.month <= end_month)]

    # Determinar el número de subplots necesarios
    num_plots = sum([CO2, CH4, CO])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    color_avg = '#0569cc'
    size_scatter = 2
    plot_index = 0

    # CO2
    if CO2:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO2_Avg'], label='CO$_{2}$ Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CO2_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO$_{2}$')
        plot_index += 1

    # CH4
    if CH4:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CH4_Avg'], label='CH$_{4}$ Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CH4_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CH$_{4}$ (ppb)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CH$_{4}$')
        plot_index += 1

    # CO
    if CO:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO_Avg'], label='CO Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CO_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CO (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO')

    # Cambios en el título
    if start_month is not None and end_month is not None:
        if start_month == end_month:
            month_str = Meses.get(start_month, f'{start_month}')
        else:
            month_str = f'{Meses.get(start_month, f"{start_month}")}-{Meses.get(end_month, f"{end_month}")}'
    else:
        month_str = ''

    plt.suptitle(f'Concentración de GEI, Estación Calakmul {month_str} {year}' if year is not None else f'Concentración de GEI, Estación Calakmul')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

def plot_1min_sd(df, CO2=True, CH4=True, CO=True, SD=True, start_month=None, end_month=None, year=None):
    """
    Ploteo de los valores promedio para CO2, CH4 y CO según los argumentos proporcionados.
    Permite seleccionar un rango de meses y un año para plotear.
    """

    Meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Filtrar por año si se especifica
    if year is not None:
        df = df[df['Time'].dt.year == year]

    # Filtrar por rango de meses si se especifica
    if start_month is not None and end_month is not None:
        df = df[(df['Time'].dt.month >= start_month) & (df['Time'].dt.month <= end_month)]

    # Determinar el número de subplots necesarios
    num_plots = sum([CO2, CH4, CO])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    color_avg = '#0569cc'
    color_sd = '#f9631f'
    size_scatter = 2
    plot_index = 0

    # CO2
    if CO2:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO2_Avg'], label='CO$_{2}$ Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CO2_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO$_{2}$')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CO2_SD'], label='CO$_{2}$ SD', color=color_sd, alpha=0.2)
            ax_twin.scatter(df['Time'], df['CO2_SD'], color=color_sd, s=size_scatter)
            ax_twin.set_ylabel('CO$_{2}$ SD')
            ax_twin.legend(loc='upper right')
        plot_index += 1

    # CH4
    if CH4:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CH4_Avg'], label='CH$_{4}$ Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CH4_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CH$_{4}$ (ppb)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CH$_{4}$')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CH4_SD'], label='CH$_{4}$ SD', color=color_sd, alpha=0.2)
            ax_twin.scatter(df['Time'], df['CH4_SD'], color=color_sd, s=size_scatter)
            ax_twin.set_ylabel('CH$_{4}$ SD')
            ax_twin.legend(loc='upper right')
        plot_index += 1

    # CO
    if CO:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO_Avg'], label='CO Avg', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CO_Avg'], color=color_avg, s=size_scatter)
        ax.set_ylabel('CO (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CO_SD'], label='CO SD', color=color_sd, alpha=0.2)
            ax_twin.scatter(df['Time'], df['CO_SD'], color=color_sd, s=size_scatter)
            ax_twin.set_ylabel('CO SD')
            ax_twin.legend(loc='upper right')

    # Cambios en el título
    if start_month is not None and end_month is not None:
        if start_month == end_month:
            month_str = Meses.get(start_month, f'{start_month}')
        else:
            month_str = f'{Meses.get(start_month, f"{start_month}")}-{Meses.get(end_month, f"{end_month}")}'
    else:
        month_str = ''

    plt.suptitle(f'Concentración de GEI, Estación Calakmul {month_str} {year}' if year is not None else f'Concentración de GEI, Estación Calakmul')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()





def plot_1min_sd_comparison(df, df2=None, CO2=True, CH4=True, CO=True, SD=True, start_month=None, end_month=None, year=None):
    """
    Ploteo de los valores promedio para CO2, CH4 y CO según los argumentos proporcionados.
    Permite seleccionar un rango de meses y un año para plotear.
    """

    Meses = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
        5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
        9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Filtrar por año si se especifica
    if year is not None:
        df = df[df['Time'].dt.year == year]
        if df2 is not None:
            df2 = df2[df2['Time'].dt.year == year]

    # Filtrar por rango de meses si se especifica
    if start_month is not None and end_month is not None:
        df = df[(df['Time'].dt.month >= start_month) & (df['Time'].dt.month <= end_month)]
        if df2 is not None:
            df2 = df2[(df2['Time'].dt.month >= start_month) & (df2['Time'].dt.month <= end_month)]

    # Determinar el número de subplots necesarios
    num_plots = sum([CO2, CH4, CO])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

    if num_plots == 1:
        axes = [axes]

    color_avg = 'orange'
    color_avg2 = '#0677d4'
    color_sd = 'green'
    color_sd2 = '#b5250e'
    size_scatter = 1
    plot_index = 0

    # CO2
    if CO2:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO2_Avg'], label='L1', color=color_avg, alpha=0.8,linewidth=0.5)
        ax.scatter(df['Time'], df['CO2_Avg'], color=color_avg, s=size_scatter)
        if df2 is not None:
            ax.plot(df2['Time'], df2['CO2_Avg'], label='L1b', color=color_avg2, alpha=0.8,linewidth=0.5)
            ax.scatter(df2['Time'], df2['CO2_Avg'], color=color_avg2, s=size_scatter)
        ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO$_{2}$')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CO2_SD'], label='SD', color=color_sd, alpha=0.2)
            ax_twin.scatter(df['Time'], df['CO2_SD'], color=color_sd, s=size_scatter)
            ax_twin.legend(loc='upper right')
            '''if df2 is not None:
                ax_twin.plot(df2['Time'], df2['CO2_SD'], label='CO$_{2}$ SD (df2)', color=color_sd2, alpha=0.2)
                ax_twin.scatter(df2['Time'], df2['CO2_SD'], color=color_sd2, s=size_scatter)
            ax_twin.set_ylabel('CO$_{2}$ SD')
            ax_twin.legend(loc='upper right')'''
        plot_index += 1

    # CH4
    if CH4:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CH4_Avg'], label='L1', color=color_avg, alpha=0.2)
        ax.scatter(df['Time'], df['CH4_Avg'], color=color_avg, s=size_scatter)
        if df2 is not None:
            ax.plot(df2['Time'], df2['CH4_Avg'], label='L1b', color=color_avg2, alpha=0.1)
            ax.scatter(df2['Time'], df2['CH4_Avg'], color=color_avg2, s=size_scatter*0.5)
        ax.set_ylabel('CH$_{4}$ (ppb)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CH$_{4}$')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CH4_SD'], label='SD', color=color_sd, alpha=0.1)
            ax_twin.scatter(df['Time'], df['CH4_SD'], color=color_sd, s=size_scatter*0.5)
            '''if df2 is not None:
                ax_twin.plot(df2['Time'], df2['CH4_SD'], label='CH$_{4}$ SD (df2)', color=color_sd2, alpha=0.2)
                ax_twin.scatter(df2['Time'], df2['CH4_SD'], color=color_sd2, s=size_scatter)
            ax_twin.set_ylabel('CH$_{4}$ SD')
            ax_twin.legend(loc='upper right')'''
            ax_twin.legend(loc='upper right')
        plot_index += 1

    # CO
    if CO:
        ax = axes[plot_index]
        ax.plot(df['Time'], df['CO_Avg'], label='L1', color=color_avg, alpha=0.2,linewidth=2)
        ax.scatter(df['Time'], df['CO_Avg'], color=color_avg, s=size_scatter)
        if df2 is not None:
            ax.plot(df2['Time'], df2['CO_Avg'], label='L1b', color=color_avg2, alpha=0.2)
            ax.scatter(df2['Time'], df2['CO_Avg'], color=color_avg2, s=size_scatter)
        ax.set_ylabel('CO (ppm)')
        ax.legend(loc='upper left')
        ax.set_title('Concentración CO')
        if SD:
            ax_twin = ax.twinx()
            ax_twin.plot(df['Time'], df['CO_SD'], label='SD', color=color_sd, alpha=0.2)
            ax_twin.scatter(df['Time'], df['CO_SD'], color=color_sd, s=size_scatter)
            '''if df2 is not None:
                ax_twin.plot(df2['Time'], df2['CO_SD'], label='CO SD (df2)', color=color_sd2, alpha=0.2)
                ax_twin.scatter(df2['Time'], df2['CO_SD'], color=color_sd2, s=size_scatter)
            ax_twin.set_ylabel('CO SD')
            ax_twin.legend(loc='upper right')'''






    # Cambios en el título
    if start_month is not None and end_month is not None:
        if start_month == end_month:
            month_str = Meses.get(start_month, f'{start_month}')
        else:
            month_str = f'{Meses.get(start_month, f"{start_month}")}-{Meses.get(end_month, f"{end_month}")}'
    else:
        month_str = ''

    plt.suptitle(f'Concentración de GEI, Estación Calakmul {month_str} {year}' if year is not None else f'Concentración de GEI, Estación Calakmul 2024')

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()










def plot_1min_avg_sd(df):
    """
    Ploteo de los valores promedio y desviacion estandar para gei.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    color_co2='#0569cc'
    color_sd='#f9631f'
    # CO2
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df['Time'], df['CO2_SD'], label='CO2_SD', color=color_sd)
    ax1.plot(df['Time'], df['CO2_Avg'], label='CO2_Avg', color=color_co2)
    ax1.set_ylabel('CO2_Avg')
    ax1_twin.set_ylabel('CO2_SD')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('CO2 Concentration')

    # CH4
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Time'], df['CH4_SD'], label='CH4_SD', color=color_sd)
    ax2.plot(df['Time'], df['CH4_Avg'], label='CH4_Avg', color=color_co2)
    ax2.set_ylabel('CH4_Avg')
    ax2_twin.set_ylabel('CH4_SD')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('CH4 Concentration')

    # CO
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['Time'], df['CO_SD'], label='CO_SD', color=color_sd)
    ax3.plot(df['Time'], df['CO_Avg'], label='CO_Avg', color=color_co2)
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









def plot_avg_sd_monthly(df):
    """
    grafica por mes los datos de co2, ch4 y co en toda la serie de tiempo, cada mes se grafica al cerrar el anterior
    """
    
    for (year, month), group in df.groupby([df['Time'].dt.year, df['Time'].dt.month]):
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'GEI {month} - {year} ') 

        
        ax1 = axes[0]
        ax1.plot(group['Time'], group['CO2_Avg'], label='CO2_Avg', color='#123456')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(group['Time'], group['CO2_SD'], label='CO2_SD', color='#F7883F', alpha=0.8)
        ax1.set_ylabel('CO2_Avg')
        ax1_twin.set_ylabel('CO2_SD')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.set_title('CO2 Concentration')

        
        ax2 = axes[1]
        ax2.plot(group['Time'], group['CH4_Avg'], label='CH4_Avg', color='#123456')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(group['Time'], group['CH4_SD'], label='CH4_SD', color='#F7883F', alpha=0.8)
        ax2.set_ylabel('CH4_Avg')
        ax2_twin.set_ylabel('CH4_SD')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.set_title('CH4 Concentration')

        
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

  #  CH4_dry
  axes[1].plot(df[timestamp_column], df['CH4_dry'], label='CH4_dry')
  axes[1].set_ylabel('CH4_dry')
  axes[1].legend()

  #  CO
  axes[2].plot(df[timestamp_column], df['CO'], label='CO')
  axes[2].set_ylabel('CO')
  axes[2].legend()

  plt.xlabel(timestamp_column)  
  plt.suptitle('Raw Data Plot')  
  plt.tight_layout()
  plt.show()































def plot_scatter(df, column):
    """

    """
    plt.figure(figsize=(16, 12))
    plt.plot(df.index, df[column], '-', color='black', linewidth=1, alpha=0.2) 
    plt.scatter(df.index, df[column],s=4,color='red' ) 
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.title(f"Scatter Plot of {column} vs. Index")
    plt.grid(True)
    plt.show()