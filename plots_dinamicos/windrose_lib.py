
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
from windrose import WindroseAxes
from datetime import datetime
from pathlib import Path



#datos de GEI

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



def met_cmul(folder_path):

    all_dfs = []  

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            
            df = pd.read_csv(file_path, encoding='ISO-8859-1', header=6,low_memory=False) 
            all_dfs.append(df)

    cmul = pd.concat(all_dfs, ignore_index=True) 
    cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.sort_values(by=['yyyy-mm-dd HH:MM:SS'])
    cmul = cmul.reset_index(drop=True)
    
    cmul.rename(columns={'deg': 'WDir_Avg','m/s': 'WSpeed_Avg'}, inplace=True)

    return cmul 

    



''' PMT64, mismo timestamp hace que coincida la serie de tiempo en ambos archivos '''

def t64_cmul(patht64):
  
  files_t64 = glob.glob(os.path.join(patht64, "*.txt"))
  df_t64 = []
  for file in files_t64:
    df = pd.read_csv(file,delimiter=',')  
    df_t64.append(df)
    t64 = pd.concat(df_t64, ignore_index=True)
  
  t64['Date & Time (Local)'] = pd.to_datetime(t64['Date & Time (Local)'])
  t64 = t64.sort_values(by=['Date & Time (Local)'])
  t64 = t64.reset_index().rename(columns={'index': 'yyyy-mm-dd HH:MM:SS'})

  t64.rename(columns={'  PM10 Conc': 'PM10 Conc','  PM2.5 Conc':'PM2.5 Conc'}, inplace=True)
  
  return t64




#linea del tiempo merge



def merge_df(dfs, timestamp_columns):
    """
   Une dos o más dataframe coindiciendo con la fecha 
   se une hasta donde tenga la fecha con datos en ese momento 
   """
    for i, df in enumerate(dfs):
        df.rename(columns={timestamp_columns[i]: 'Time'}, inplace=True)
    
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Time', how='inner' )  #inner
    
    return merged_df












'''PLOTEOS DE ROSAS DE VIENTO     MET Y PM '''





#ploteo
def rosa_pm(wr_cmul):

    plt.figure(figsize=(18, 9))  

    plt.subplot(2, 3, (1, 3))  
    plt.subplot(2, 3, (1, 3))  
    plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM10 Conc'], 
             label='PM10 Conc', color='#4C4B16')
    plt.plot(wr_cmul['yyyy-mm-dd HH:MM:SS'], wr_cmul['PM2.5 Conc'], 
             label='PM2.5 Conc', alpha=0.9, color='#E6C767')
    
    
    plt.title('Observatorio Atmosférico Calakmul Concentraciones de PM2.5 y PM10 \n')
    plt.ylabel('\nConcentración μg/m^3')
    
    #datos estadisticos.
    
    max_ws = wr_cmul['WSpeed_Avg'].max()
    max_pm10=wr_cmul['PM10 Conc'].max()
    max_pm25=wr_cmul['PM2.5 Conc'].max()
    mean_pm10=wr_cmul['PM10 Conc'].mean()
    mean_pm25=wr_cmul['PM2.5 Conc'].mean()
    
    plt.figtext(0.1, 0.8, f'Velocidad máxima: {max_ws:.1f}m/s \nMáx PM10: {max_pm10:.1f} \nMáx PM 2.5: {max_pm25:.1f}\nConcentración promedio\nPM10: {mean_pm10:.1f}\nPM 2.5: {mean_pm25:.1f}',  # Adjust position as needed
            fontsize=10, color='black',bbox=dict(facecolor='white', alpha=1, boxstyle='round'))  

    plt.legend()
    plt.grid(True)



    '''Rosas de viento           '''

    # DIRECCION DE VIENTO
    ax = plt.subplot(2, 3, 4, projection="windrose")  
    wind_data = wr_cmul[['WSpeed_Avg', 'WDir_Avg']].dropna()
    ax.bar(wind_data['WDir_Avg'], wind_data['WSpeed_Avg'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Velocidad (m/s)", title_fontsize=8,
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title('\nVelocidad y dirección del viento')

    # PM10 CONC
    ax = plt.subplot(2, 3, 5, projection="windrose")  
    wind_data = wr_cmul[['PM10 Conc', 'WDir_Avg']].dropna()

    max_pm10 = wr_cmul['PM10 Conc'].max()
    PM10_umbral=round(max_pm10-(max_pm10*0.50),2)
    

    filtered_wind_data = wind_data[wind_data['PM10 Conc'] >= PM10_umbral]
    ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM10 Conc'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Concentración", title_fontsize=8, 
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title(f'PM 10 mayor a {PM10_umbral} μg/m^3')

    # PM2.5
    ax = plt.subplot(2, 3, 6, projection="windrose")  
    wind_data = wr_cmul[['PM2.5 Conc', 'WDir_Avg']].dropna()

    max_pm25 = wr_cmul['PM2.5 Conc'].max()
    PM25_umbral=round(max_pm25-(max_pm25*0.5),2)

    filtered_wind_data = wind_data[wind_data['PM2.5 Conc'] >= PM25_umbral]
    ax.bar(filtered_wind_data['WDir_Avg'], filtered_wind_data['PM2.5 Conc'], 
           normed=True, opening=0.8, edgecolor='white',bins=4)
    ax.legend(title="Concentración", title_fontsize=8, 
              loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title(f'PM 2.5 mayor a {PM25_umbral} μg/m^3')

    plt.tight_layout()  
    plt.show()






def met_windrose(wr_cmul, timestamp='yyyy-mm-dd HH:MM:SS', column='WSpeed_Avg'):
    """
    Esta función toma un DataFrame y plotea rosas de viento en un subplot de 4x3 para cada mes de 2024.
    
    Parameters:
        wr_cmul (pd.DataFrame): DataFrame con datos de viento
        timestamp (str): Nombre de la columna que contiene los datos de fecha y hora (por defecto: 'yyyy-mm-dd HH:MM:SS')
        column (str): Nombre de la columna a plotear en las rosas de viento (por defecto: 'WSpeed_Avg')
    """

    # Filtrar datos para el año 2024
    wr_cmul_2024 = wr_cmul[wr_cmul[timestamp].dt.year == 2024]

    # Crear una figura con subplots 4x3
    fig, axs = plt.subplots(4, 3, figsize=(9, 15), subplot_kw={'projection': 'windrose'})
    
    # Título dinámico basado en la columna elegida
    if column == 'WSpeed_Avg':
        fig.suptitle('Rosas de Viento Mensuales para 2024', fontsize=16)
    else:
        fig.suptitle(f'Rosas de Viento Mensuales para {column} - 2024', fontsize=16)

    # Nombres de los meses
    month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    legend_title = "Velocidad (m/s)" if column == 'WSpeed_Avg' else f"{column}"
    # Iterar sobre cada mes y crear una rosa de viento
    for month in range(1, 13):
        ax = axs[(month-1)//3, (month-1)%3]
        monthly_data = wr_cmul_2024[wr_cmul_2024[timestamp].dt.month == month]
        wind_data = monthly_data[['WDir_Avg', column]].dropna()

        if not wind_data.empty:
            ax.bar(wind_data['WDir_Avg'], wind_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
            ax.set_title(f'\n{month_names[month]}', fontsize=10, pad=10)  # Ajustar la posición del título
            # Configurar las etiquetas cardinales
            ax.xaxis.set_ticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
            ax.xaxis.set_tick_params(labelsize=5)  
            ax.yaxis.set_tick_params(labelsize=5)
            ax.tick_params(axis='both', which='major', pad=-5)
            ax.legend(title=legend_title, title_fontsize=8, loc="lower right", 
                     bbox_to_anchor=(0.5, -0.1), prop={'size': 5})
        else:
            ax.set_title(month_names[month], fontsize=10, pad=10)  # Ajustar la posición del título
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            # Configurar las etiquetas cardinales también para gráficos vacíos
            ax.xaxis.set_ticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
            ax.xaxis.set_tick_params(labelsize=4)  
            ax.yaxis.set_tick_params(labelsize=4)
            ax.tick_params(axis='both', which='major', pad=-5)


    # Ajustar el espaciado entre los subplots
    plt.subplots_adjust(hspace=0.9, wspace=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()




# rosas de viento de gei pm


def plot_windrose(df, column):
    """
  
        None
    """
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' is not in the DataFrame.")

    # Filter the DataFrame to include only rows with non-null values for the specified column and 'WDir_Avg'
    windrose_data = df[['WDir_Avg', column]].dropna()

    # Plotting the windrose
    ax = WindroseAxes.from_ax()
    ax.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=15)
    ax.legend(title=f"{column} (units)", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax.set_title(f'Windrose for {column}')
    plt.show()





def plot_windrose_subplots_intervalos(df, columns, intervals=None):
    """
    Prepares and plots windrose subplots for the specified columns, with optional filtering by value intervals.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing wind data.
        columns (list of str): List of columns to plot (e.g., wind speed or pollutant concentrations).
        intervals (dict, optional): A dictionary where keys are column names and values are tuples specifying
                                     the (min, max) range to filter the data for that column.

    Returns:
        None
    """
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    
    # Create subplots dynamically based on the number of columns (1 row, n columns)
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8), subplot_kw={'projection': 'windrose'})
    
    # Ensure axes is iterable (even if there's only one column)
    if n == 1:
        axes = [axes]
    
    for ax, column in zip(axes, columns):
        if column not in df.columns:
            raise ValueError(f"The specified column '{column}' is not in the DataFrame.")
        
        # Filter the DataFrame to include only rows with non-null values for the specified column and 'WDir_Avg'
        windrose_data = df[['WDir_Avg', column]].dropna()
        
        # Apply interval filtering if specified
        if intervals and column in intervals:
            min_val, max_val = intervals[column]
            windrose_data = windrose_data[(windrose_data[column] >= min_val) & (windrose_data[column] <= max_val)]
        
        # Plotting the windrose for the current column
        ax.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        ax.legend(title=f"{column}", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
        ax.set_title(f'Windrose for {column}')
    
    plt.tight_layout()
    plt.show()


def plot_windrose_subplots(df, columns):
    """
    Prepares and plots windrose subplots for the specified columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing wind data.
        columns (list of str): List of columns to plot (e.g., wind speed or pollutant concentrations).

    Returns:
        None
    """
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    
    # Create subplots dynamically based on the number of columns (1 row, n columns)
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8), subplot_kw={'projection': 'windrose'})
    
    # Ensure axes is iterable (even if there's only one column)
    if n == 1:
        axes = [axes]
    
    for ax, column in zip(axes, columns):
        if column not in df.columns:
            raise ValueError(f"The specified column '{column}' is not in the DataFrame.")
        
        # Filter the DataFrame to include only rows with non-null values for the specified column and 'WDir_Avg'
        windrose_data = df[['WDir_Avg', column]].dropna()
        
        # Plotting the windrose for the current column
        ax.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=15)
        ax.legend(title=f"{column} (units)", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
        ax.set_title(f'Windrose for {column}')
    
    plt.tight_layout()
    plt.show()





def plot_wr_timeseries(df, column):
    """
    Creates a 1x3 subplot layout:
    - The first subplot spans positions (1, 2) and plots a time series of the specified column.
    - The third subplot (position 3) plots a windrose using 'WDir_Avg' and the specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing wind data.
        column (str): The column to plot (e.g., wind speed or pollutant concentration).

    Returns:
        None
    """
    if 'Time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Time' column for the x-axis of the time series.")
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' is not in the DataFrame.")

    # Create a 1x3 subplot layout
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Time series plot (spanning positions 1 and 2)
    ax1 = plt.subplot(1, 3, (1, 2))
    ax1.plot(df['Time'], df[column], label=column, color='blue')
    ax1.set_title(f"Time Series of {column}")
    ax1.set_xlabel("Time")
    ax1.set_ylabel(column)
    ax1.grid(True)
    ax1.legend()

    # Subplot 2: Windrose plot (position 3)
    ax2 = plt.subplot(1, 3, 3, projection="windrose")
    windrose_data = df[['WDir_Avg', column]].dropna()
    ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
    ax2.legend(title=f"{column} (units)", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
    ax2.set_title(f"Windrose for {column}")

    plt.tight_layout()
    plt.show()

def plot_wr_timeseries_dynamic(df, columns):
    """
    Creates a dynamic subplot layout:
    - For one column: 1x3 layout (time series spanning positions 1 and 2, and windrose in position 3).
    - For multiple columns: n x 3 layout, where n is the number of columns.
    - Custom labels for CO2 and CH4, and windrose title set to 'Dirección del viento'.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing wind data.
        columns (list of str): List of columns to plot (e.g., wind speed or pollutant concentrations).

    Returns:
        None
    """
    

    if 'Time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Time' column for the x-axis of the time series.")
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"The specified column '{column}' is not in the DataFrame.")

    # Determine the number of rows based on the number of columns
    n = len(columns)
    fig = plt.figure(figsize=(15, 5 * n))  # Adjust figure height dynamically

    for i, column in enumerate(columns):
        # Calculate the starting position for the current row
        row_start = i * 3 + 1

        # Subplot 1: Time series plot for the current column (spanning positions 1 and 2 in the row)
        ax1 = plt.subplot(n, 3, (row_start, row_start + 1))
        ax1.plot(df['Time'], df[column], label=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, color='blue')
        ax1.set_title(f"Time Series of {column}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column)
        ax1.grid(True)
        ax1.legend()

        # Subplot 2: Windrose plot for the current column (position 3 in the row)
        ax2 = plt.subplot(n, 3, row_start + 2, projection="windrose")
        windrose_data = df[['WDir_Avg', column]].dropna()
        ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        ax2.legend(title=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
        ax2.set_title("Dirección del viento")

    plt.tight_layout()
    plt.show()



def plot_wr_timeseries_date(df, columns, inicio=None, fin=None):
    """
    Plotea con direccion de viento y serie de tiempo de GEI, puedes ingresar un intervalo de tiempo 
    para filtrar los datos.

    """

    #filtrado de fecha 
    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        df = df[df['Time'] >= inicio]
    if fin is not None:
        fin = pd.to_datetime(fin)
        df = df[df['Time'] <= fin]

    if df.empty:
        raise ValueError("The filtered DataFrame is empty. Check your 'inicio' and 'fin' values.")

    # se ajusta el numero de subplots
    n = len(columns)
    fig = plt.figure(figsize=(13, 3 * n))  # Adjust figure height dynamically


    fig.suptitle('Serie de tiempo y dirección de viento Calakmul', fontsize=16, y=0.95)  # Adjust 'y' to position the suptitle


    for i, column in enumerate(columns):
        # Calculate the starting position for the current row
        row_start = i * 3 + 1

        # Subplot 1: Time series plot for the current column (spanning positions 1 and 2 in the row)
        ax1 = plt.subplot(n, 3, (row_start, row_start + 1))
        ax1.plot(df['Time'], df[column], label=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, color='blue')
        ax1.set_title(f"Time Series of {column}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column)
        ax1.grid(True)
        ax1.legend()

        # Subplot 2: Windrose plot for the current column (position 3 in the row)
        ax2 = plt.subplot(n, 3, row_start + 2, projection="windrose")
        windrose_data = df[['WDir_Avg', column]].dropna()
        ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        ax2.legend(title=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
        ax2.set_title("Dirección del viento")

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust 'rect' to leave space for the suptitle
    plt.show()




def plot_wr_timeseries_date1(df, columns, inicio=None, fin=None):
  
    # Filter the DataFrame based on the 'inicio' and 'fin' arguments
    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        df = df[df['Time'] >= inicio]
    if fin is not None:
        fin = pd.to_datetime(fin)
        df = df[df['Time'] <= fin]

    if df.empty:
        raise ValueError("The filtered DataFrame is empty. Check your 'inicio' and 'fin' values.")

    # Determine the number of rows based on the number of columns
    n = len(columns)
    fig = plt.figure(figsize=( 5* n, 5))  # Adjust figure height dynamically

    # Add a fixed suptitle
    fig.suptitle('Serie de tiempo y dirección de viento Calakmul', fontsize=16, y=0.95)  # Adjust 'y' to position the suptitle

    # Initialize the first axis for sharing x-axis
    shared_ax = None

    for i, column in enumerate(columns):
        # Calculate the starting position for the current row
        row_start = i * 3 + 1

        # Subplot 1: Time series plot for the current column (spanning positions 1 and 2 in the row)
        ax1 = plt.subplot(n, 3, (row_start, row_start + 1), sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax1  # Set the first axis as the shared x-axis
        ax1.plot(df['Time'], df[column], label=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, color='blue')
        ax1.set_xlabel("Time")
        ax1.set_ylabel(f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column)
        ax1.grid(True)
        ax1.legend()

        # Subplot 2: Windrose plot for the current column (position 3 in the row)
        ax2 = plt.subplot(n, 3, row_start + 2, projection="windrose")
        windrose_data = df[['WDir_Avg', column]].dropna()
        ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        ax2.legend(title=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})

    # Adjust layout to fit the plots below the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust 'rect' to leave space for the suptitle
    plt.show()


    
def plot_wr_timeseries_date2(df, columns, inicio=None, fin=None, rmax=None):
    if 'Time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Time' column for the x-axis of the time series.")
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"The specified column '{column}' is not in the DataFrame.")

    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        df = df[df['Time'] >= inicio]
    if fin is not None:
        fin = pd.to_datetime(fin)
        df = df[df['Time'] <= fin]

    if df.empty:
        raise ValueError("The filtered DataFrame is empty. Check your 'inicio' and 'fin' values.")

    n = len(columns)
    fig = plt.figure(figsize=(10, 3 * n))
    fig.suptitle('Serie de tiempo y dirección de viento, Calakmul', fontsize=16, y=0.95)

    shared_ax = None

    for i, column in enumerate(columns):
        row_start = i * 3 + 1

        ax1 = plt.subplot(n, 3, (row_start, row_start + 1), sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax1
        ax1.plot(df['Time'], df[column], label=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, color='blue')
        
        ax1.set_ylabel(f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column)
        ax1.grid(True)
        ax1.legend()

        ax2 = plt.subplot(n, 3, row_start + 2, projection="windrose")
        windrose_data = df[['WDir_Avg', column]].dropna()
        ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        ax2.legend(title=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, title_fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 7})

        # labels windrose
        ax2.xaxis.set_ticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
        ax2.xaxis.set_tick_params(labelsize=8)  
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.tick_params(axis='both', which='major', pad=-20)  

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()






def plot_wr_timeseries_date3(df, columns, inicio=None, fin=None, rmax=None, cuartil=None):
    if 'Time' not in df.columns:
        raise ValueError("The DataFrame must contain a 'Time' column for the x-axis of the time series.")
    if 'WDir_Avg' not in df.columns:
        raise ValueError("The DataFrame must contain a 'WDir_Avg' column for wind direction.")
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"The specified column '{column}' is not in the DataFrame.")

    if inicio is not None:
        inicio = pd.to_datetime(inicio)
        df = df[df['Time'] >= inicio]
    if fin is not None:
        fin = pd.to_datetime(fin)
        df = df[df['Time'] <= fin]

    if df.empty:
        raise ValueError("The filtered DataFrame is empty. Check your 'inicio' and 'fin' values.")

    n = len(columns)
    fig = plt.figure(figsize=(10, 3 * n))
    fig.suptitle('Serie de tiempo y dirección de viento, Calakmul', fontsize=16, y=0.95)

    shared_ax = None

    for i, column in enumerate(columns):
        row_start = i * 3 + 1

        ax1 = plt.subplot(n, 3, (row_start, row_start + 1), sharex=shared_ax)
        if shared_ax is None:
            shared_ax = ax1
        ax1.plot(df['Time'], df[column], label=f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column, color='blue')
        
        ax1.set_ylabel(f"CO$_{{2}}$ (ppm)" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm)" if column == 'CH4_Avg' else column)
        ax1.grid(True)
        ax1.legend()

        ax2 = plt.subplot(n, 3, row_start + 2, projection="windrose")
        windrose_data = df[['WDir_Avg', column]].dropna()
        
        # Filter data based on quartile if provided
        if cuartil is not None:
            if cuartil <= 0 or cuartil >= 100:
                raise ValueError("Cuartil must be a value between 0 and 100")
            
            threshold = np.percentile(windrose_data[column], cuartil)
            windrose_data = windrose_data[windrose_data[column] >= threshold]
            quartile_title = f" (>{cuartil}%)"
        else:
            quartile_title = ""
            
        ax2.bar(windrose_data['WDir_Avg'], windrose_data[column], normed=True, opening=0.8, edgecolor='white', bins=4)
        
        title_text = f"CO$_{{2}}$ (ppm){quartile_title}" if column == 'CO2_Avg' else f"CH$_{{4}}$ (ppm){quartile_title}" if column == 'CH4_Avg' else f"{column}{quartile_title}"
        ax2.legend(title=title_text, title_fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5), prop={'size': 7})

        # labels windrose
        ax2.xaxis.set_ticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
        ax2.xaxis.set_tick_params(labelsize=8)  
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.tick_params(axis='both', which='major', pad=-20)  

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()