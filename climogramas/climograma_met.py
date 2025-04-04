import os
import pandas as pd
import matplotlib.pyplot as plt
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


folder_path = './DATOS/met/L2/hora' 




cmul = met_cmul_L1_L2(folder_path)
cmul['yyyy-mm-dd HH:MM:SS'] = pd.to_datetime(cmul['yyyy-mm-dd HH:MM:SS'])
cmul = cmul.sort_values(by=['yyyy-mm-dd HH:MM:SS'])
cmul = cmul.reset_index(drop=True)


#mes


cmul = cmul.set_index('yyyy-mm-dd HH:MM:SS')

monthly_stats = cmul.groupby(cmul.index.month).agg({'°C': ['mean', 'max', 'min'], 'mm': 'sum'})
monthly_stats.columns = ['_'.join(col) for col in monthly_stats.columns]


fig, ax1 = plt.subplots(figsize=(10, 6))


ax1.bar(monthly_stats.index, monthly_stats['mm_sum'], color='none',
         edgecolor='blue',hatch='///', alpha=0.5, label='Precipitation (mm)', zorder=1)


ax1.set_xlabel('    ')
ax1.set_ylabel('Precipitación (mm)', color='blue')
ax1.tick_params('y', labelcolor='blue')
ax1.set_xticks(np.arange(1, 13))
ax1.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])


ax2 = ax1.twinx()
ax2.plot(monthly_stats.index, monthly_stats['°C_mean'], color='green', marker='o', label='Avg T (°C)', zorder=5)
ax2.plot(monthly_stats.index, monthly_stats['°C_max'], color='red', linestyle='--', marker='o', label='Max T (°C)', zorder=5)
ax2.plot(monthly_stats.index, monthly_stats['°C_min'], color='blue', linestyle='--',marker='o', label='Min T (°C)', zorder=5)

ax2.set_ylabel('T (°C)', color='red')
ax2.tick_params('y', labelcolor='red')



plt.title('Climograma Calakmul ')

fig.legend(loc='lower center', ncol=4)

plt.tight_layout()
plt.show()
