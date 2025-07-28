import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import os
import numpy as np




def met_cmul_L1_L2(folder_path):
    """
    Reads multiple CSV files with inconsistent headers.

    Args:
        folder_path: The path to the folder containing the CSV files.

    Returns:
        A pandas DataFrame containing the data from all the files.
    """
    all_dfs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Check if it's a CSV file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='ISO-8859-1') as f:
                lines = f.readlines()
                header_row = None
                for i, line in enumerate(lines):
                    if line.strip() == '':
                        header_row = 6
                        break
                    if 'yyyy-mm-dd HH:MM:SS' in line:
                        header_row = i
                        break
                
            df = pd.read_csv(file_path, header=header_row, encoding='ISO-8859-1')
            all_dfs.append(df)

    
    cmul = pd.concat(all_dfs, ignore_index=True)
    return cmul


folder_path = r'DATOS Sensores\met\L2\hora' 




cmul = met_cmul_L1_L2(folder_path)
cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
cmul = cmul.sort_values(by=['yyyy-mm-dd HH:MM:SS'])
cmul = cmul.reset_index(drop=True)










cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
mes = cmul['yyyy-mm-dd HH:MM:SS'].dt.month_name().iloc[0]
year = int(cmul['yyyy-mm-dd HH:MM:SS'].dt.year.iloc[0])
month = cmul['yyyy-mm-dd HH:MM:SS'].dt.month.iloc[0]
month_number = "{:02d}".format(int(month))

''' estadistica '''
mean_temp = cmul['°C'].mean()
max_temp = cmul['°C'].max()
min_temp = cmul['°C'].min()
min_hPa = cmul['hPa'].min()
max_hPa = cmul['hPa'].max()
rain_tot = cmul['mm'].sum()
percen_null = (cmul['°C'].isnull().sum() / len(cmul)) * 100
print(f"Porcentaje de valores nulos : {percen_null:.2f}%")
print(f'valores válidos: {100 - percen_null:.2f}%')
print(f'Valores nulos: {cmul["°C"].isnull().sum()}')
print(f'estadistica \npromedio: {cmul["°C"].mean():.2f}\nValor max: {cmul["°C"].max()}\nValor min: {cmul["°C"].min()}')

selected_columns = ['°C', '%', 'm/s', 'deg', 'mm', 'hPa', 'W/m^2']
cmul_stats = cmul[selected_columns].agg(['mean', 'max', 'min'])
print(cmul_stats)


''' Gráficas '''

fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(10,6))

ax1 = axes[0]
ax2 = ax1.twinx()
ax1.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['°C'], marker='', linestyle='-', color='red')
ax1.set_ylabel('Temp (°C)', color='red'), ax1.yaxis.set_minor_locator(MultipleLocator(2))
ax2.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['%'], marker='', linestyle='-', color='blue')
ax2.set_ylabel('RH (%)', color='blue'), ax2.set_yticks(np.arange(25,101,25))

ax3 = axes[1]
ax4 = ax3.twinx()
ax3.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['deg'], marker='', linestyle='-', color='green')
ax3.set_ylabel('WDIR (deg)', color='green'), ax3.set_yticks(np.arange(0,361,90))
ax4.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['m/s'], marker='', linestyle='-', color='black',alpha=0.5)
ax4.set_ylabel('WSpeed (m/s)', color='black'), ax4.set_yticks(np.arange(0,16,5))
ax4.yaxis.set_minor_locator(MultipleLocator(1))

ax5=axes[2]
ax5.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['W/m^2'], marker='', linestyle='-', color='orange')
ax5.set_ylabel('Rad (W/m^2)', color='orange'), ax5.yaxis.set_minor_locator(MultipleLocator(100))


ax7 = axes[3]
ax8 = ax7.twinx()
ax7.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['hPa'], marker='', linestyle='-', color='black')
ax7.set_ylabel('Pres (hPa)', color='black'); ax7.set_ylim((min_hPa - 1, max_hPa + 1))
ax8.plot(cmul['yyyy-mm-dd HH:MM:SS'], cmul['mm'], marker='', linestyle='-', color='blue')
# ax8.bar(cmul['yyyy-mm-dd HH:MM:SS'], cmul['mm'], color='blue', alpha=0.8, width=0.1)
ax8.set_ylabel('Rain (mm)', color='blue'), ax8.yaxis.set_minor_locator(MultipleLocator(0.1))

axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d" "\n" "%Hh"))
axes[0].xaxis.set_minor_locator(MultipleLocator(1))

# Add text to the figure
fig.text(0.20, 0.71, f"Temp avg: {mean_temp:.1f} °C", fontsize=10, color='red')
fig.text(0.45, 0.71, f"Max Temp: {max_temp:.1f} °C", fontsize=10, color='red')
fig.text(0.75, 0.71, f"Min Temp: {min_temp:.1f} °C", fontsize=10, color='red')
fig.text(0.75, 0.27, f"Rain Tot: {rain_tot:.1f} mm", fontsize=10, color='blue')

plt.suptitle(f'Observatorio Atmosférico Calakmul: condiciones atmosféricas {year}')
plt.tight_layout()
#plt.savefig(f'cmul_l2_{mes}.png')
plt.show()