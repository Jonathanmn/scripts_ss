''' 



En este archivo se van a comparar los valores de CO2 y NDVI


'''


from picarro import *
from picarro_clean import *




folder_path = 'DATOS Sensores/gei/L1b/minuto/2024'

#ndvi_folder_path = 'gei/CO2_NDVI/files/MOD13Q1_NDVI_2024.txt'

ndvi_folder_path = r'C:\git_mn\MODIS_NDVI\MOD13Q1_NDVI_2024-cmul-9pix.txt'

gei = read_L0_or_L1(folder_path, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)
gei['Time'] = pd.to_datetime(gei['Time'])






def resample24_month(df):


    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month
    df_resampled['Dia'] = df_resampled.index.day

    
    df_monthly_avg_24h = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    df_month=df_monthly_avg_24h.groupby('Mes').mean().reset_index()
    df_month['Mes'] = df_month['Mes']
    df_month['Mes'] = pd.to_datetime(df_month['Mes'], format='%m').dt.strftime('%m')
    df_month['CO2_Avg'] = df_month['CO2_Avg'].round(3)
    df_month= df_month[['Mes', 'CO2_Avg']]
    return df_month



co2_delta=timeseries_delta_per_day(gei, CO2='CO2_Avg', start_month=1, end_month=12)

print(co2_delta)

co2_delta['Mes'] = pd.to_datetime(co2_delta['Mes'], format='%m').dt.strftime('%m')
co2_delta=co2_delta[['Mes', 'CO2_Avg_delta']]
co2_delta['CO2_Avg_delta'] = co2_delta['CO2_Avg_delta'].round(3)


print(co2_delta)



ndvi_avg = pd.read_csv(ndvi_folder_path, sep=',', header=0,)

ndvi_avg['Mes'] = pd.to_datetime(ndvi_avg['Mes'], format='%m').dt.strftime('%m')
ndvi_avg = ndvi_avg[['Mes', 'NDVI_Mean','NDVI_StdErr']]
ndvi_avg['NDVI_Mean'] = ndvi_avg['NDVI_Mean'].round(3)
ndvi_avg['NDVI_StdErr'] = ndvi_avg['NDVI_StdErr'].round(3)
print(ndvi_avg)




# Unimos los dataframes por mes
ndvi_co2 = pd.merge(co2_delta, ndvi_avg, left_on='Mes', right_on='Mes', how='inner')
ndvi_co2 = ndvi_co2[['Mes', 'CO2_Avg_delta', 'NDVI_Mean']]

ndvi_co2['Mes'] = pd.to_datetime(ndvi_co2['Mes'], format='%m').dt.strftime('%b')

print(ndvi_co2.info())









def plot_co2_delta_ndvi(ndvi_co2):
    """
    Plot CO2_Avg_delta and NDVI_Mean on twin y-axes with month as x-axis.
    
    Parameters:
    -----------
    ndvi_co2 : pandas.DataFrame
        DataFrame containing 'Mes', 'CO2_Avg_delta', and 'NDVI_Mean' columns
    """
    # Create figure and primary axis with extra space at bottom
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot CO2_Avg_delta on primary y-axis
    line_delta = ax1.plot(ndvi_co2['Mes'], ndvi_co2['CO2_Avg_delta'], '^--', 
                         color='purple', label='Δ CO₂ (ppm)', markersize=8,
                         zorder=5)  # Lower than annotations but higher than grid
    
    # Add annotations for CO2_Avg_delta
    for i, row in ndvi_co2.iterrows():
        ax1.annotate(f"{row['CO2_Avg_delta']:.1f}", 
                   (row['Mes'], row['CO2_Avg_delta']),
                   textcoords="offset points", 
                   xytext=(10, 10), 
                   ha='left',
                   fontsize=9,
                   color='purple',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=.8),
                   zorder=10)  # Higher zorder ensures it's on top
    
    # Configure primary y-axis
    ax1.set_ylabel('Δ CO₂ (max-min)', fontsize=12, color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_ylim(0, 80)  # Set y-axis limits for CO2_Avg_delta
    # Use the existing month abbreviations directly
    ax1.set_xticks(range(len(ndvi_co2['Mes'])))
    ax1.set_xticklabels(ndvi_co2['Mes'])
    ax1.set_xlabel('Mes', fontsize=12)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot NDVI_Mean on secondary y-axis
    line_ndvi = ax2.plot(ndvi_co2['Mes'], ndvi_co2['NDVI_Mean'], 'o-', 
                        color='green', label='NDVI promedio', markersize=8,
                        zorder=5)  # Lower than annotations but higher than grid
    
    # Add annotations for NDVI_Mean
    for i, row in ndvi_co2.iterrows():
        ax2.annotate(f"{row['NDVI_Mean']:.2f}", 
                   (row['Mes'], row['NDVI_Mean']),
                   textcoords="offset pixels", 
                   xytext=(-10, 10), 
                   ha='right',
                   fontsize=9,
                   color='green',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=.8),
                   zorder=10)  # Higher zorder ensures it's on top
    
    # Configure secondary y-axis
    ax2.set_ylabel('NDVI', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0.5, 1)  # Set y-axis limits for NDVI
    # Add grid
    ax1.grid(True, alpha=0.3, zorder=1)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add combined legend to the lower right corner
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='lower right',
              frameon=True,
              fontsize=10)
    
    # Add title
    plt.suptitle(f'Observatorio Atmosférico Calakmul 2024\nDelta Δ CO₂ e índice NDVI', fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for title only
    
    plt.show()


plot_co2_delta_ndvi(ndvi_co2)



def error_plot(df):
    """
    Plot NDVI_Mean with error bars showing NDVI_StdErr on one axis
    and CO2_Avg_delta on another axis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Mes', 'CO2_Avg_delta', 'NDVI_Mean', and 'NDVI_StdErr' columns
    """
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot CO2_Avg_delta on primary y-axis (similar to previous function)
    line_delta = ax1.plot(df['Mes'], df['CO2_Avg_delta'], '^--', 
                         color='purple', label='Δ CO₂ (ppm)', markersize=8,
                         zorder=5)
    
    # Add annotations for CO2_Avg_delta
    for i, row in df.iterrows():
        ax1.annotate(f"{row['CO2_Avg_delta']:.1f}", 
                   (row['Mes'], row['CO2_Avg_delta']),
                   textcoords="offset points", 
                   xytext=(10, 10), 
                   ha='left',
                   fontsize=9,
                   color='purple',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=1),
                   zorder=15)
    
    # Configure primary y-axis
    ax1.set_ylabel('Δ CO₂ (max-min)', fontsize=12, color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.set_ylim(0, 80)
    ax1.set_xticks(range(len(df['Mes'])))
    ax1.set_xticklabels(df['Mes'])
    ax1.set_xlabel('Mes', fontsize=12)
    
    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Plot NDVI_Mean with error bars on secondary y-axis
    x_numeric = np.arange(len(df['Mes']))
    line_ndvi = ax2.errorbar(x_numeric, df['NDVI_Mean'], yerr=df['NDVI_StdErr'], 
                           fmt='o-', color='green', label='NDVI promedio', 
                           markersize=8, capsize=5, elinewidth=1.5, capthick=1.5,
                           zorder=5)
    
    # Add annotations for NDVI_Mean with error values
    for i, row in df.iterrows():
        ax2.annotate(f"{row['NDVI_Mean']:.2f}±{row['NDVI_StdErr']:.2f}", 
                   (i, row['NDVI_Mean']),
                   textcoords="offset pixels", 
                   xytext=(0, 10), 
                   ha='center',
                   fontsize=9,
                   color='green',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=1),
                   zorder=15)
    
    # Configure secondary y-axis
    ax2.set_ylabel('NDVI', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(0.5, 1)
    
    # Add grid behind everything
    ax1.grid(True, alpha=0.3, zorder=1)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Add combined legend
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='lower right',
              frameon=True,
              fontsize=10)
    
    # Add title
    plt.suptitle(f'Observatorio Atmosférico Calakmul 2024\nDelta Δ CO₂ e índice NDVI con barras de error', 
               fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()


def ndvi_error_plot(df):
    """
    Plot NDVI_Mean with error bars showing NDVI_StdErr.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Mes', 'NDVI_Mean', and 'NDVI_StdErr' columns
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot NDVI_Mean with error bars
    x_numeric = np.arange(len(df['Mes']))
    line_ndvi = ax.errorbar(x_numeric, df['NDVI_Mean'], yerr=df['NDVI_StdErr'], 
                          fmt='o-', color='green', label='NDVI promedio', 
                          markersize=10, capsize=6, elinewidth=1.5, capthick=1.5,
                          zorder=5)
    
    # Add annotations for NDVI_Mean with error values
    for i, row in df.iterrows():
        ax.annotate(f"{row['NDVI_Mean']:.1f}±{row['NDVI_StdErr']:.1f}", 
                  (i, row['NDVI_Mean']),
                  textcoords="offset pixels", 
                  xytext=(-50, 0), 
                  ha='center',
                  fontsize=7,
                  color='green',
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=1),
                  zorder=15)
    
    # Configure axis
    ax.set_ylabel('NDVI', fontsize=14, color='green')
    ax.tick_params(axis='y', labelcolor='green')
    ax.set_ylim(0.5, 1)  # Adjust as needed
    ax.set_xticks(range(len(df['Mes'])))
    ax.set_xticklabels(df['Mes'])
    ax.set_xlabel('Mes', fontsize=14)
    
    # Add grid behind everything
    ax.grid(True, alpha=0.3, zorder=1)
    
    # Add legend
    ax.legend(loc='lower right', frameon=True, fontsize=12)
    
    # Add title
    plt.suptitle(f'Observatorio Atmosférico Calakmul 2024\nÍndice NDVI con barras de error', 
               fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()


ndvi_error_plot(ndvi_avg)






















