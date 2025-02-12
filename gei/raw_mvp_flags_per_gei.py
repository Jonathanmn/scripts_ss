import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import timedelta
import matplotlib.dates as mdates
import re




gei = pd.read_csv('C:/Users/malag/Downloads/03_raw_mvp.dat',delimiter=",")

gei['timestamp']=pd.to_datetime(gei['timestamp'])

for col in ['CO2_dry', 'CH4_dry', 'CO']:
        
        gei[col] = gei[col].round(3)







def flags_mvp(df):
  
  #df = df.dropna(subset=['MPVPosition'])
  #df['MPVPosition'] = df['MPVPosition'].round().astype(int)
  df['MPVPosition'] = df['MPVPosition'].fillna(-1).round().astype(int)
  MPVcount = df['MPVPosition'].value_counts(dropna=True)
  

  
  for value, count in MPVcount.items():
    if value != 0 and value != 1: 
      column_name = f'MVP_{value}'
      
      temp_df = df[df['MPVPosition'] == value][['CO2_dry', 'CH4_dry', 'CO']]
      # Rename columns
      temp_df = temp_df.rename(columns={
          'CO2_dry': f'{column_name}_CO2_flag',
          'CH4_dry': f'{column_name}_CH4_flag',
          'CO': f'{column_name}_CO_flag'
      })
      
      df = pd.merge(df, temp_df, left_index=True, right_index=True, how='left')

  return df


def plot_co2_flags(df):
    """
    Plots all columns ending with '_CO2-flag' in the given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
    """
    df.loc[((df['MPVPosition'] != 1) & (df['MPVPosition'] != 1)), "CO2_dry"] = None

    # Get all columns ending with '_CO2-flag'
    co2_flag_cols = [col for col in df.columns if col.endswith('_CO2_flag')]

    # If no such columns are found, print a message and return
    if not co2_flag_cols:
        print("No columns ending with '_CO2-flag' found in the DataFrame.")
        return

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each CO2 flag column
    for col in co2_flag_cols:
        ax.plot(df['timestamp'], df[col], label=col)

    plt.plot(df['timestamp'], df['CO2_dry'], label='CO2_dry')
    # Set plot labels and title
    ax.set_xlabel('Index')
    ax.set_ylabel('CO2 Values')
    ax.set_title('CO2 Flag Columns')
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


gei=flags_mvp(gei)



def plot_flags_subplots(df):
    """
    Plots 'CH4_dry', 'CO', and 'CO2_dry' with their corresponding flag columns
    using subplots. Applies conditional filtering based on 'MPVPosition'.
    Rounds 'CO2_dry', 'CH4_dry', and 'CO' to 3 decimal places.
    Places legend outside the plot area.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
    """

    # Round columns to 3 decimal places
    for col in ['CO2_dry', 'CH4_dry', 'CO']:
        df[col] = df[col].round(3)

    # Apply conditional filtering for all relevant columns
    for base_col in ['CO2_dry', 'CH4_dry', 'CO']:
        df.loc[((df['MPVPosition'] != 1) & (df['MPVPosition'] != 1)), base_col] = None

    # Get flag columns for each base column
    flag_cols_dict = {
        'CO2_dry': [col for col in df.columns if col.endswith('_CO2_flag')],
        'CH4_dry': [col for col in df.columns if col.endswith('_CH4_flag')],
        'CO': [col for col in df.columns if col.endswith('_CO_flag')]
    }

    # Create subplots
    fig, axes = plt.subplots(len(flag_cols_dict), 1, figsize=(10, 6 * len(flag_cols_dict)), sharex=True)

    # Iterate and plot for each base column
    for i, (base_col, flag_cols) in enumerate(flag_cols_dict.items()):
        ax = axes[i]  # Get the current subplot axes

        # Plot base column
        ax.plot(df['timestamp'], df[base_col], label=base_col, color='#273746', alpha=0.4)

        # Plot corresponding flag columns
        for flag_col in flag_cols:
            ax.plot(df['timestamp'], df[flag_col], label=flag_col)

        # Set subplot labels and title
        ax.set_ylabel(f'{base_col} Values')
        ax.set_title(f'{base_col} and Flag Columns')
        
        # Place legend outside the plot area
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 

    # Set overall x-axis label
    plt.xlabel('Timestamp')

    # Show the plot
    plt.tight_layout()
    plt.show()







plot_flags_subplots(gei)
#plot_co2_flags(gei)









