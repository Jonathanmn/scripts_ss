''' Este archivo va a intertar plotear todos los archivos de raw lite y diversas fuentes miscelaneas.'''




#ojo piojo sd umbrales   co2 =>2 ch4=>0.002

from picarro import *

#folder_path = '/home/jmn/L1/minuto/2024'

folder_path = 'DATOS Sensores/gei/L1b/minuto/2024' 

#gei = read_raw_lite(folder_path, 'Time')
gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)

print(gei.head())
#gei=umbrales_sd(gei)


folder_pathb='./DATOS/gei/L1b/minuto/2024'



plot_1min_avg(gei, CO2=True, CH4=False, CO=False)


'''


geib = read_L0_or_L1(folder_pathb, 'yyyy-mm-dd HH:MM:SS', header=7)
geib = reverse_rename_columns(geib)
geib['Time'] = pd.to_datetime(geib['Time'])




# Llamada a la función plot_1min_avg con argumentos dinámicos, rango de meses y año
#plot_1min_avg_month_scatter(gei, CO2=True, CH4=True, CO=True, start_month=1, end_month=12, year=2024)




plot_1min_sd(gei, CO2=True, CH4=True, CO=False, SD=True,start_month=1, end_month=12, year=2024)

#plot_1min_sd_comparison(gei, df2=geib, CO2=True, CH4=False, CO=False, SD=True, start_month=None, end_month=None, year=2024)

'''

def plot_describe_table(df):
    """
    Plots the descriptive statistics of the 'CO2_Avg' and 'CH4_Avg' columns of a DataFrame as a table.
    """
    # Filter the DataFrame to include only the 'CO2_Avg' and 'CH4_Avg' columns
    df_filtered = df[['CO2_Avg', 'CH4_Avg']]
    desc = df_filtered.describe(percentiles=[]).transpose().round(3)
    
    fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(desc)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=desc.values, colLabels=desc.columns, rowLabels=desc.index, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.title('Estadisticas para CO$_2$ y CH$_4$')
    plt.show()
#plot_describe_table(geib)

