import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_plotly(df, column):
    """ un scatter que selecciona puntos y los elimina del data frame
    """
    selected_indices = []

    
    fig, ax = plt.subplots()
    scatter = ax.scatter(df.index, df[column], s=4, picker=True)
    ax.plot(df.index, df[column], '-', color='blue', linewidth=2) 
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