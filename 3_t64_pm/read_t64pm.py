import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes



folder_t64 = 'C:/git_mn/scripts_ss/_files/pm/L0/minuto'


txt_files = glob.glob(os.path.join(folder_t64, '*.txt'))

dataframes = []
for file in txt_files:
    usecols = ['Date & Time (Local)', '  PM2.5 Conc', '  PM10 Conc']
    df = pd.read_csv(file, sep=',', header=0, usecols=usecols)
    dataframes.append(df)

t64_=pd.concat(dataframes,ignore_index=True)
t64_['Date & Time (Local)']=pd.to_datetime(t64_['Date & Time (Local)'])
t64_=t64_.sort_values(by='Date & Time (Local)')
t64_=t64_.reset_index(drop=True)


print(t64_.columns)


#rename columns
t64_.rename(columns={
    'Date & Time (Local)': 'datetime',
    '  PM2.5 Conc': 'pm25',
    '  PM10 Conc': 'pm10'
}, inplace=True)


print(t64_.columns)

t64_['datetime']=pd.to_datetime(t64_['datetime'], format='%Y-%m-%d %H:%M:%S')

#plot
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))


plt.scatter(t64_['datetime'], t64_['pm25'], label='PM2.5 Concentration', color='blue')
plt.scatter(t64_['datetime'], t64_['pm10'], label='PM10 Concentration', color='orange')
plt.xlabel('Date and Time')
plt.ylabel('Concentration (µg/m³)')
plt.title('PM2.5 and PM10 Concentrations Over Time')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()