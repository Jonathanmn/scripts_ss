from picarro import *
from picarro_ciclos import *




import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Create a 3x2 grid of subplots
fig, axs = plt.subplots(3, 2, sharex='col', sharey='row')

# Plotting data
axs[0, 0].plot(x, y1)
axs[1, 0].plot(x, y2)
axs[2, 0].plot(x, y3)

# Set y-labels only on the left side
axs[0, 0].set_ylabel('Sine')
axs[1, 0].set_ylabel('Cosine')
axs[2, 0].set_ylabel('Tangent')

# Set x-labels only on the bottom row
axs[2, 0].set_xlabel('X-axis')
axs[2, 1].set_xlabel('X-axis')

# Hide inner x-labels
for ax in axs[:-1, :].flatten():
    ax.label_outer()

plt.tight_layout()
plt.show()





'''

folder_path = '/home/jmn/L1b/hora/2024'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])







#gei=umbrales_sd(gei, CO2_umbral=0.2,CH4_umbral=0.002)
#gei=umbrales_sd(gei)




plot_1min_avg_sd(gei)

output_folder='/home/jmn/L1_2'


save_gei_l1_minuto(gei,output_folder)

save_gei_l1_hora(gei,output_folder)'
''
''
''
'''


