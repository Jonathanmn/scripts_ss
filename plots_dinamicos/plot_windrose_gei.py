
from windrose_lib import *


''' Aqui se van a plotear los ciclos horarios y nocturnos'''

''' folders laptop 
folder_cmul = '/home/jmn/DATA/met/L2/hora' 
folder_gei = '/home/jmn/L1/minuto/2024'
'''
#folders lab pc
folder_met = '/home/jonathan_mn/Descargas/data/met/L2/minuto'
folder_gei = '/home/jonathan_mn/gei-l1/minuto/2024' 




gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


met = met_cmul(folder_met)   

met_winddata = met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei_winddata = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]



gei_met=pd.concat([gei_winddata, met_winddata], axis=1, join='inner')

print (gei_met.columns)





def plot_windrose(df, column):
    """
    Prepares and plots a windrose using the specified column.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing wind data.
        column (str): The column to plot (e.g., wind speed or pollutant concentration).

    Returns:
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

# Example usage
# Assuming `gei_met` is your DataFrame
 
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




#plot_windrose_subplots(gei_met, columns=['CO2_Avg', 'CH4_Avg'])


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
        ax.legend(title=f"{column} (units)", title_fontsize=8, loc="lower right", bbox_to_anchor=(0.5, 0.1), prop={'size': 7})
        ax.set_title(f'Windrose for {column}')
    
    plt.tight_layout()
    plt.show()

# Example usage
# Assuming `gei_met` is your DataFrame
intervals = {
    'CO2_Avg': (500, 600),
    'CH4_Avg': (2.3, 3)
}
#plot_windrose_subplots_intervalos(gei_met, columns=['CO2_Avg', 'CH4_Avg'], intervals=intervals)



import matplotlib.pyplot as plt
from windrose import WindroseAxes

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

# Example usage
# Assuming `gei_met` is your DataFrame
plot_wr_timeseries(gei_met, column='CO2_Avg')
