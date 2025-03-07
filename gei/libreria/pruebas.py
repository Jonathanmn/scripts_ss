
from picarro import *
from picarro_clean import *


'''Lab

folder_path = '/home/jonathan_mn/l0-1/minuto/2024/03'


// local


folder_path = '/home/jmn/L1/minuto/2024/03'
'''

folder_path = '/home/jmn/L1/minuto/2024'


gei=read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei=reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])


gei_nocturno=gei[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()
gei_diario=gei[['Time', 'CH4_Avg', 'CO2_Avg', 'CO_Avg']].copy()

gei_nocturno['Time'] = pd.to_datetime(gei_nocturno['Time'])


ciclo_filtrado=gei_nocturno[((gei_nocturno['Time'].dt.hour >= 19) | (gei_nocturno['Time'].dt.hour <= 5))].copy().reset_index(drop=True)


print(ciclo_filtrado.head())

ciclo_filtrado['Time'] = ciclo_filtrado['Time'] - timedelta(hours=5)

print(ciclo_filtrado.head())

ciclo_filtrado=ciclo_filtrado.set_index('Time')

ciclo_dia=ciclo_filtrado.resample('1D').agg(['mean','std'])
# Rename columns
ciclo_dia.columns = ['_'.join(col).replace('_mean', '').replace('_std', '_SD') for col in ciclo_dia.columns] 
ciclo_dia=ciclo_dia.reset_index()


print(ciclo_dia.head())




import matplotlib.pyplot as plt

# Assuming ciclo_dia has columns like 'CO2_Avg', 'CO2_Avg_SD', 'CH4_Avg', 'CH4_Avg_SD', etc.
gas_cols = ['CO2_Avg', 'CH4_Avg']  # List of gas columns

fig, axes = plt.subplots(len(gas_cols), 1, sharex=True, figsize=(10, 6))  # Create subplots

for i, gas in enumerate(gas_cols):
    ax = axes[i]  # Get the current subplot
    ax.plot(ciclo_dia['Time'], ciclo_dia[gas], label='Mean', color='blue')  # Plot mean
    
    # Create a secondary y-axis for SD
    ax2 = ax.twinx()  
    ax2.plot(ciclo_dia['Time'], ciclo_dia[gas + '_SD'], label='SD', color='red')  # Plot SD on secondary axis
    
    ax.set_ylabel(gas, color='blue')  # Set y-axis label for mean
    ax2.set_ylabel(gas + ' SD', color='red')  # Set y-axis label for SD
    
    # Combine legends from both axes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)

plt.xlabel('Time')  # Set x-axis label for the entire figure
plt.suptitle('Mean and Standard Deviation of Gases')  # Set overall title
plt.tight_layout()  # Adjust layout for better spacing
plt.show()















#ciclo_diurno_plottly_6(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')


#ciclo_diurno_mensual_matplot2(gei, 'CH4_Avg', 'CO2_Avg')


#ciclo_diurno_3(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
'''
save_data = input("revisamos la linea de tiempo? (yes/no): ")

if save_data.lower() == 'yes':

    plot_1min_avg_sd(gei)
else:
    print("Los datos no se han guardado.")




#plot_hourly_subplots(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')


'''




'''
gei_clean = clean_plotly_gei(gei, 'CH4_Avg', 'CO2_Avg', 'CO_Avg')
plot_scatter(gei_clean, 'CH4_Avg')

limpieza usando el scatter 


save_data = input("¿Desea guardar los datos limpios? (yes/no): ")

if save_data.lower() == 'yes':
    folder = '/home/jmn'
    save_to(gei_clean, 'Time', folder)
    print("Datos guardados en la carpeta:", folder)
else:
    print("Los datos no se han guardado.")

#save_to(gei, 'Time', folder)  #home/jonathan_mn/clean_prueba

#home/jonathan_mn/clean_prueba
'''