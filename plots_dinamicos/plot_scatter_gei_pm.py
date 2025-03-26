from windrose_lib import *





folder_met = './DATOS/met/L2/minuto'
folder_t64 = './DATOS/pm/L0/minuto'
folder_gei = './DATOS/gei/L1/minuto/2024' 






gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)






t64 = t64_cmul(folder_t64)

met = met_cmul(folder_met)   
t64=t64[['Date & Time (Local)','PM10 Conc', 'PM2.5 Conc']]
met= met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]



print(t64.columns)

#gei_met=pd.concat([gei_winddata, met_winddata], axis=1, join='inner')





import pandas as pd

def merge_on_timestamp(dfs, timestamp_columns):
    """
    Merges multiple DataFrames on a common timestamp column, even if the column names differ.

    Args:
        dfs (list of pd.DataFrame): List of DataFrames to merge.
        timestamp_columns (list of str): List of timestamp column names corresponding to each DataFrame.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    # Rename timestamp columns to a common name
    for i, df in enumerate(dfs):
        df.rename(columns={timestamp_columns[i]: 'Timestamp'}, inplace=True)
    
    # Merge DataFrames on the common 'Timestamp' column
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Timestamp', how='inner' )  #inner
    
    return merged_df


# Merge the DataFrames
merged_data = merge_on_timestamp(
    [met,gei,t64],
    ['yyyy-mm-dd HH:MM:SS', 'Time', 'Date & Time (Local)'])

print(merged_data)






import matplotlib.pyplot as plt

def analisis_variables(df, column1, column2, timestamp_column='Timestamp'):
    """
    Creates a scatter plot with two variables from a DataFrame, using a secondary y-axis (twinx).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column1 (str): The name of the first column to plot on the primary y-axis.
        column2 (str): The name of the second column to plot on the secondary y-axis.
        timestamp_column (str): The name of the column to use as the x-axis (timestamps).

    Returns:
        None
    """
    fig, ax1 = plt.subplots()
    marker_size=1
    # Plot column1 on the primary y-axis
    ax1.scatter(df[timestamp_column], df[column1], color='blue', label=column1, alpha=0.7,s=marker_size)
    ax1.plot(df[timestamp_column], df[column1], color='blue', alpha=0.2, linewidth=.5)
    ax1.set_xlabel(timestamp_column)
    ax1.set_ylabel(column1, color='blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.tick_params(axis='x', rotation=45)

    # Create a secondary y-axis (twinx)
    ax2 = ax1.twinx()
    ax2.scatter(df[timestamp_column], df[column2], color='red', label=column2, alpha=0.7,s=marker_size)
    ax2.plot(df[timestamp_column], df[column2], color='red', alpha=0.3, linewidth=.5)
    ax2.set_ylabel(column2, color='red')
    ax2.tick_params(axis='y', colors='red')

    # Add a legend
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

    # Show the plot
    plt.title(f'Scatter Plot: {column1} and {column2} over {timestamp_column}')
    plt.tight_layout()
    plt.show()

# Example usage
analisis_variables(merged_data, 'CO2_Avg', 'PM2.5 Conc', timestamp_column='Timestamp')