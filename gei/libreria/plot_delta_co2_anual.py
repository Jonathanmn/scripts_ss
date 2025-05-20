from picarro import *
from picarro_ciclos import *


folder_path='DATOS Sensores/gei/L1b/minuto/2024'


gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])

# Create a new DataFrame with max, min, and delta CO2_Avg per month
gei['month'] = gei['Time'].dt.month

# Get max and min for each month (regardless of year)
monthly_max = gei.groupby('month')['CO2_Avg'].max().reset_index()
monthly_min = gei.groupby('month')['CO2_Avg'].min().reset_index()

# Create the gei_delta DataFrame
gei_delta = pd.DataFrame({'month': monthly_max['month'],
                         'max': monthly_max['CO2_Avg'],
                         'min': monthly_min['CO2_Avg']})

# Calculate delta (max - min)
gei_delta['delta'] = gei_delta['max'] - gei_delta['min']

# Sort by month for better readability
gei_delta = gei_delta.sort_values('month')

print(gei_delta.head())