import pandas as pd
from pathlib import Path
import time
import matplotlib.pyplot as plt

def raw_gei_folder(folder_path):
    """
    Lee los archivos .dat del folder donde se encuentran los archivos raw
    retorna un data frame con todos los archivos. 
    """
    dataframes = []
    start_time = time.time()

    for file_path in Path(folder_path).rglob('*.dat'):
        df = pd.read_csv(file_path, delimiter=r',') 
        dataframes.append(df)

    gei = pd.concat(dataframes, ignore_index=True)
    
    
    gei['timestamp'] = pd.to_datetime(gei['timestamp'])
    gei = gei.sort_values(by='timestamp').reset_index(drop=True)
    
    
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to read and process the files: {time_taken:.2f} seconds")

    return gei

def flags_1min(df):
    """
    aplica un filtro donde conserva los valores solo en las especies correspondientes al gas y calcula promedio y desviacion por minuto de muestreo
    retorna el data frame por minuto ya sin la columna de especies. 
    """

    df.loc[((df["species"] != 2) & (df["species"] != 3)), "CO2_dry"] = None
    df.loc[((df["species"] != 3)), "CH4_dry"] = None
    df.loc[((df["species"] != 1) & (df["species"] != 4)), "CO"] = None

    df = df.set_index('timestamp')

    agg_funcs = {
        'MPVPosition': 'mean',
        'CO2_dry': ['mean', 'std'],
        'CH4_dry': ['mean', 'std'],
        'CO': ['mean', 'std']
    }

    
    resampled_df = df.resample('1min').agg(agg_funcs)

     
    resampled_df.columns = [f'{col[0]}_{col[1]}' if isinstance(col, tuple) else col
                         for col in resampled_df.columns]
    resampled_df = resampled_df.rename(columns={
    
    'MPVPosition_mean': 'MPVPosition',
    'CO2_dry_mean': 'CO2_Avg',
    'CO2_dry_std': 'CO2_SD',
    'CH4_dry_mean': 'CH4_Avg',
    'CH4_dry_std': 'CH4_SD',
    'CO_mean': 'CO_Avg',
    'CO_std': 'CO_SD'
    })
    
    resampled_df = resampled_df.reset_index().rename(columns={'timestamp': 'Time'})

    return resampled_df



def plot_gei_avg_sd(df):
    """
    Ploteo de los valores promedio y desviacion estandar para gei.
    """

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True) 

    # CO2
    ax1 = axes[0]
    ax1.plot(df['Time'], df['CO2_Avg'], label='CO2_Avg', color='blue')
    ax1_twin = ax1.twinx()  # Create a twin axis for CO2_SD
    ax1_twin.plot(df['Time'], df['CO2_SD'], label='CO2_SD', color='red')
    ax1.set_ylabel('CO2_Avg')
    ax1_twin.set_ylabel('CO2_SD')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.set_title('CO2 Concentration')

    # CH4
    ax2 = axes[1]
    ax2.plot(df['Time'], df['CH4_Avg'], label='CH4_Avg', color='blue')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df['Time'], df['CH4_SD'], label='CH4_SD', color='red')
    ax2.set_ylabel('CH4_Avg')
    ax2_twin.set_ylabel('CH4_SD')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.set_title('CH4 Concentration')

    # CO
    ax3 = axes[2]
    ax3.plot(df['Time'], df['CO_Avg'], label='CO_Avg', color='blue')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['Time'], df['CO_SD'], label='CO_SD', color='red')
    ax3.set_ylabel('CO_Avg')
    ax3_twin.set_ylabel('CO_SD')
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    ax3.set_title('CO Concentration')

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()




folder_path = '/home/jonathan_mn/data'

gei=raw_gei_folder(folder_path)

gei_flags=flags_1min(gei)

print(gei_flags.head())
print('ya estuvo')



plot_gei_avg_sd(gei_flags)
