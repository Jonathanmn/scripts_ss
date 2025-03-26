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

