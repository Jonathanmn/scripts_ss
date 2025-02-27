import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_plotly(df, column):
    """ un scatter que selecciona puntos y los elimina del data frame
    """
    selected_indices = []

    
    fig, ax = plt.subplots()
    ax.plot(df.index, df[column], '-', color='black', linewidth=1, alpha=0.2)
    scatter = ax.scatter(df.index, df[column], s=4, picker=True, color='red')
     
    ax.set_title(f'Interactive Plot for {column}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column)

    
    def onpick(event):
          if event.artist != scatter:
            return True

          ind = event.ind
          if not len(ind):
            return True

    
          for i in ind:

            selected_indices.append(df.index[i])
            df.at[df.index[i], column] = np.nan

    # Actualizar el gráfico
          scatter.set_offsets(np.c_[df.index, df[column]])
          fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

    
    print("Selected indices:", selected_indices)
    df.loc[selected_indices, column] = np.nan 
    
    
    sd_column = column[:-3] + 'SD'  
   
    if sd_column in df.columns:
        df.loc[selected_indices, sd_column] = np.nan

    return df 


def clean_plotly_gei(df, CH4, CO2, CO):
    """ Un scatter que selecciona puntos y los elimina del data frame para CH4_Avg, CO2_Avg y CO_Avg """
    selected_indices_CH4 = []
    selected_indices_CO2 = []
    selected_indices_CO = []
    month_year = df['Time'].dt.to_period('M').unique()[0]
    fig, axs = plt.subplots(3, 1, figsize=(20, 15), sharex=True, sharey=False)

    # Plot for CH4_Avg
    axs[0].plot(df.index, df[CH4], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CH4 = axs[0].scatter(df.index, df[CH4], s=4, picker=True, color='red')
    axs[0].set_title(f'{CH4}')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(CH4)

    # Plot for CO2_Avg
    axs[1].plot(df.index, df[CO2], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CO2 = axs[1].scatter(df.index, df[CO2], s=4, picker=True, color='red')
    axs[1].set_title(f'{CO2}')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(CO2)

    # Plot for CO_Avg
    axs[2].plot(df.index, df[CO], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CO = axs[2].scatter(df.index, df[CO], s=4, picker=True, color='red')
    axs[2].set_title(f'{CO}')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel(CO)

    def onpick(event):
        if event.artist == scatter_CH4:
            ind = event.ind
            for i in ind:
                selected_indices_CH4.append(df.index[i])
                df.at[df.index[i], CH4] = np.nan
        elif event.artist == scatter_CO2:
            ind = event.ind
            for i in ind:
                selected_indices_CO2.append(df.index[i])
                df.at[df.index[i], CO2] = np.nan
        elif event.artist == scatter_CO:
            ind = event.ind
            for i in ind:
                selected_indices_CO.append(df.index[i])
                df.at[df.index[i], CO] = np.nan

        # Actualizar los gráficos
        scatter_CH4.set_offsets(np.c_[df.index, df[CH4]])
        scatter_CO2.set_offsets(np.c_[df.index, df[CO2]])
        scatter_CO.set_offsets(np.c_[df.index, df[CO]])
        fig.canvas.draw_idle()
    fig.tight_layout()
    fig.canvas.mpl_connect('pick_event', onpick)
    fig.suptitle(f'Picarro {month_year}', fontsize=16)

    plt.show()

    
    df.loc[selected_indices_CH4, CH4] = np.nan
    df.loc[selected_indices_CO2, CO2] = np.nan
    df.loc[selected_indices_CO, CO] = np.nan

    
    for gas, selected_indices in zip([CH4, CO2, CO], [selected_indices_CH4, selected_indices_CO2, selected_indices_CO]):
        sd_column = gas[:-3] + 'SD'
        if sd_column in df.columns:
            df.loc[selected_indices, sd_column] = np.nan

    return df



def clean_plotly_gei_1(df, CH4, CO2, CO):
    """ Un scatter que selecciona puntos y los elimina del data frame para CH4_Avg, CO2_Avg y CO_Avg """
    selected_indices_CH4 = []
    selected_indices_CO2 = []
    selected_indices_CO = []

    # Obtener el mes y el año del DataFrame
    month_year = df['Time'].dt.to_period('M').unique()[0]

    # Crear subplots con tamaño grande
    fig, axs = plt.subplots(3, 1, figsize=(20, 15), sharex=True)

    # Plot for CH4_Avg
    axs[0].plot(df.index, df[CH4], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CH4 = axs[0].scatter(df.index, df[CH4], s=4, picker=True, color='red')
    ax2_0 = axs[0].twiny()
    ax2_0.set_xlim(axs[0].get_xlim())
    valid_ticks_0 = [tick for tick in axs[0].get_xticks() if 0 <= tick < len(df)]
    ax2_0.set_xticks(valid_ticks_0)
    ax2_0.set_xticklabels(df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[valid_ticks_0])
    axs[0].set_title(f'Interactive Plot for {CH4}')
    #axs[0].set_xlabel('Index')
    axs[0].set_ylabel(CH4)

    # Plot for CO2_Avg
    axs[1].plot(df.index, df[CO2], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CO2 = axs[1].scatter(df.index, df[CO2], s=4, picker=True, color='red')
    ax2_1 = axs[1].twiny()
    ax2_1.set_xlim(axs[1].get_xlim())
    valid_ticks_1 = [tick for tick in axs[1].get_xticks() if 0 <= tick < len(df)]
    ax2_1.set_xticks(valid_ticks_1)
    ax2_1.set_xticklabels(df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[valid_ticks_1])
    axs[1].set_title(f'Interactive Plot for {CO2}')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(CO2)

    # Plot for CO_Avg
    axs[2].plot(df.index, df[CO], '-', color='black', linewidth=1, alpha=0.2)
    scatter_CO = axs[2].scatter(df.index, df[CO], s=4, picker=True, color='red')
    ax2_2 = axs[2].twiny()
    ax2_2.set_xlim(axs[2].get_xlim())
    valid_ticks_2 = [tick for tick in axs[2].get_xticks() if 0 <= tick < len(df)]
    ax2_2.set_xticks(valid_ticks_2)
    ax2_2.set_xticklabels(df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S').iloc[valid_ticks_2])
    axs[2].set_title(f'Interactive Plot for {CO}')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel(CO)

    def onpick(event):
        if event.artist == scatter_CH4:
            ind = event.ind
            for i in ind:
                selected_indices_CH4.append(df.index[i])
                df.at[df.index[i], CH4] = np.nan
        elif event.artist == scatter_CO2:
            ind = event.ind
            for i in ind:
                selected_indices_CO2.append(df.index[i])
                df.at[df.index[i], CO2] = np.nan
        elif event.artist == scatter_CO:
            ind = event.ind
            for i in ind:
                selected_indices_CO.append(df.index[i])
                df.at[df.index[i], CO] = np.nan

        # Actualizar los gráficos
        scatter_CH4.set_offsets(np.c_[df.index, df[CH4]])
        scatter_CO2.set_offsets(np.c_[df.index, df[CO2]])
        scatter_CO.set_offsets(np.c_[df.index, df[CO]])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', onpick)

    # Ajustar la figura para que se adapte al tamaño de la ventana
    fig.tight_layout()
    fig.suptitle(f'Interactive Plots for {month_year}', fontsize=16)
    plt.show()

    print("Selected indices for CH4:", selected_indices_CH4)
    print("Selected indices for CO2:", selected_indices_CO2)
    print("Selected indices for CO:", selected_indices_CO)

    # Aplicar np.nan a los índices seleccionados en cada gas
    df.loc[selected_indices_CH4, CH4] = np.nan
    df.loc[selected_indices_CO2, CO2] = np.nan
    df.loc[selected_indices_CO, CO] = np.nan

    # Aplicar np.nan a las columnas de desviación estándar si existen
    for gas, selected_indices in zip([CH4, CO2, CO], [selected_indices_CH4, selected_indices_CO2, selected_indices_CO]):
        sd_column = gas[:-3] + 'SD'
        if sd_column in df.columns:
            df.loc[selected_indices, sd_column] = np.nan

    return df