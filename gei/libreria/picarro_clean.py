import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from plot_ciclo_horas import intervalo_horas, ciclo_1d_avg
'''herramienta para desplegar el plotly y eliminar datos de forma visual '''



def clean_plotly_gei(df, CH4, CO2, CO):
 

    """ Un scatter que selecciona puntos y los elimina del data frame para CH4_Avg, CO2_Avg y CO_Avg """
    selected_indices_CH4 = []
    selected_indices_CO2 = []
    selected_indices_CO = []

    month_year = df['Time'].dt.to_period('M').unique()[0]
    fig, axs = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
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






def plot_scatter(df, column):
    """
    """
    plt.figure(figsize=(16, 12))
    plt.plot(df.index, df[column], '-', color='black', linewidth=1, alpha=0.2)
    plt.scatter(df.index, df[column], s=4, color='red')
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.title(f"Scatter Plot of {column} vs. Index")
    plt.grid(True)
    plt.show()



def ciclo_diurno_3(df, CO2, CH4, CO):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por día,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas.
    """
    # Asegurarse de que 'Time' esté en el índice
    df = df.set_index('Time')

    # Resamplear el DataFrame a intervalos de 1 hora
    df_resampled = df.resample('1H').mean()

    # Crear una nueva columna 'Hora' que contiene solo la hora del día
    df_resampled['Hora'] = df_resampled.index.hour

    # Calcular el promedio mensual para cada hora del día
    df_monthly_avg = df_resampled.groupby('Hora').mean()

    # Crear subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Iterar sobre cada día y plotear las 24 horas en el mismo subplot
    for day, group in df_resampled.groupby(df_resampled.index.date):
        # Plot para CO2_Avg
        axs[0].plot(group['Hora'], group[CO2], '-', linewidth=1, alpha=0.7) #label=str(day))
        # Plot para CH4_Avg
        axs[1].plot(group['Hora'], group[CH4], '-', linewidth=1, alpha=0.7) #label=str(day))
        # Plot para CO_Avg
        axs[2].plot(group['Hora'], group[CO], '-', linewidth=1, alpha=0.7) #label=str(day))

    # Plotear el promedio mensual como referencia
    axs[0].plot(df_monthly_avg.index, df_monthly_avg[CO2], 'k--', linewidth=2, label='Promedio Mensual')
    axs[1].plot(df_monthly_avg.index, df_monthly_avg[CH4], 'k--', linewidth=2, label='Promedio Mensual')
    axs[2].plot(df_monthly_avg.index, df_monthly_avg[CO], 'k--', linewidth=2, label='Promedio Mensual')

    # Configurar los subplots
    axs[0].set_title(f'Ciclo Diurno de {CO2}')
    axs[0].set_ylabel(CO2)
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize='small')

    axs[1].set_title(f'Ciclo Diurno de {CH4}')
    axs[1].set_ylabel(CH4)
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize='small')

    axs[2].set_title(f'Ciclo Diurno de {CO}')
    axs[2].set_xlabel('Hora del Día')
    axs[2].set_ylabel(CO)
    axs[2].grid(True)
    axs[2].legend(loc='upper right', fontsize='small')

    # Ajustar la figura para que se adapte al tamaño de la ventana
    fig.tight_layout()
    fig.suptitle('Ciclo Diurno de Gases de Efecto Invernadero', fontsize=16)
    plt.show()




def ciclo_diurno_plottly_6(df, CO2, CH4, CO):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por día,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas.
    También grafica los outliers junto al promedio mensual.
    """
    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour

    df_monthly_avg = df_resampled.groupby('Hora').mean().reset_index()
    df_monthly_std = df_resampled.groupby('Hora').std().reset_index()

    year_month = df.index.to_period('M')[0].strftime('%Y-%m')
    first_date = df.index.min().strftime('%Y-%m-%d')
    last_date = df.index.max().strftime('%Y-%m-%d')

 
    colors = [
        'rgba(30, 105, 221, 1)',  
        'rgba(59, 102, 167, 1)',  
        'rgba(70, 96, 135, 1)',
        'rgba(32, 189, 236, 1)' 
    ]

    # Función para crear el plot de cada gas
    def plot_gas(df_resampled, df_monthly_avg, df_monthly_std, gas, title):
        fig = go.Figure()

        # Iterar sobre cada día y plotear las 24 horas en el mismo plot
        for i, (day, group) in enumerate(df_resampled.groupby(df_resampled.index.date)):
            group = group.reset_index()
            color = colors[i % len(colors)]  # Asignar color del colormap
            fig.add_trace(go.Scatter(x=group['Hora'], y=group[gas], mode='lines', line=dict(width=3, color=color), opacity=0.9, name=str(day)))

        # Plotear el promedio mensual como referencia
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas], mode='lines', line=dict(color='red', dash='dash', width=6), name='Promedio Mensual'))

        # Plotear los outliers
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas] + 2 * df_monthly_std[gas], mode='lines', line=dict(color='red', dash='dot', width=3), name='Outliers Sup'))
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas] - 2 * df_monthly_std[gas], mode='lines', line=dict(color='red', dash='dot', width=3), name='Outliers Inf'))

        # Agregar anotaciones de texto para cada hora
        for i, row in df_monthly_avg.iterrows():
            fig.add_annotation(x=row['Hora'], y=row[gas], text=f"{row[gas]:.2f}", showarrow=True, yshift=8, bgcolor="rgba(255, 255, 255, 0.8)")


        # Configurar el plot
        fig.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_yaxes(title_text=gas, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_layout(title_text=f'{title} ({first_date} al {last_date})', showlegend=True, autosize=True, height=780, width=1520, plot_bgcolor='white')
        fig.show()

    # Plotear cada gas individualmente
    plot_gas(df_resampled, df_monthly_avg, df_monthly_std, CO, f'Ciclo Diurno de {CO}')
    plot_gas(df_resampled, df_monthly_avg, df_monthly_std, CO2, f'Ciclo Diurno de {CO2}')
    plot_gas(df_resampled, df_monthly_avg, df_monthly_std, CH4, f'Ciclo Diurno de {CH4}')



def ciclo_diurno_plottly_7(df, CO2, CH4, CO):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por día,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas.
    También grafica los outliers junto al promedio mensual.
    """
    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour

    df_monthly_avg = df_resampled.groupby('Hora').mean().reset_index()
    df_monthly_std = df_resampled.groupby('Hora').std().reset_index()

    year_month = df.index.to_period('M')[0].strftime('%Y-%m')
    first_date = df.index.min().strftime('%Y-%m-%d')
    last_date = df.index.max().strftime('%Y-%m-%d')

    # Definir un colormap personalizado de azul a violeta
    colors = [
        'rgba(0, 0, 255, 1)',  
        'rgba(75, 0, 130, 1)',  
        'rgba(138, 43, 226, 1)',  
    ]

    # Crear la figura
    fig = go.Figure()

    # Función para crear el plot de cada gas
    def plot_gas(fig, df_resampled, df_monthly_avg, df_monthly_std, gas, title):
        # Iterar sobre cada día y plotear las 24 horas en el mismo plot
        for i, (day, group) in enumerate(df_resampled.groupby(df_resampled.index.date)):
            group = group.reset_index()
            color = colors[i % len(colors)]  # Asignar color del colormap
            fig.add_trace(go.Scatter(x=group['Hora'], y=group[gas], mode='lines', line=dict(width=3, color=color), opacity=0.9, name=str(day), visible=False))

        # Plotear el promedio mensual como referencia
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas], mode='lines', line=dict(color='red', dash='dash', width=6), name='Promedio Mensual', visible=False))

        # Plotear los outliers
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas] + 2 * df_monthly_std[gas], mode='lines', line=dict(color='grey', dash='dot', width=2), name='Outliers Superior', visible=False))
        fig.add_trace(go.Scatter(x=df_monthly_avg['Hora'], y=df_monthly_avg[gas] - 2 * df_monthly_std[gas], mode='lines', line=dict(color='grey', dash='dot', width=2), name='Outliers Inferior', visible=False))

        # Agregar anotaciones de texto para cada hora
        for i, row_data in df_monthly_avg.iterrows():
            fig.add_annotation(x=row_data['Hora'], y=row_data[gas], text=f"{row_data[gas]:.2f}", showarrow=True, yshift=8, bgcolor="rgba(255, 255, 255, 0.8)", visible=False)

    # Plotear cada gas
    plot_gas(fig, df_resampled, df_monthly_avg, df_monthly_std, CO2, f'Ciclo Diurno de {CO2}')
    plot_gas(fig, df_resampled, df_monthly_avg, df_monthly_std, CH4, f'Ciclo Diurno de {CH4}')
    plot_gas(fig, df_resampled, df_monthly_avg, df_monthly_std, CO, f'Ciclo Diurno de {CO}')

    # Crear botones para cambiar entre los gráficos
    buttons = []
    gases = [CO2, CH4, CO]
    for i, gas in enumerate(gases):
        visible = [False] * len(fig.data)
        visible[i*4:(i+1)*4] = [True] * 4
        buttons.append(dict(label=gas, method='update', args=[{'visible': visible}, {'title': f'Ciclo Diurno de {gas} ({first_date} al {last_date})'}]))

    # Agregar los botones a la figura
    fig.update_layout(updatemenus=[dict(active=0, buttons=buttons, x=1.15, y=1.15)])

    # Configurar el layout
    fig.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='grey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='grey')
    fig.update_layout(title_text=f'Ciclo Diurno de {CO2} ({first_date} al {last_date})', showlegend=True, autosize=True, height=780, width=1520, plot_bgcolor='white')
    fig.show()




def ciclo_diurno_mensual_anual(df, CO2, CH4, CO):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por mes y hora,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas.
    """
    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month

    # Calcular el promedio mensual para cada hora del día
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()

    # Obtener el año del DataFrame
    year = df.index.year[0]

    # Diccionario para mapear los números de los meses a sus nombres
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Definir un colormap personalizado
    colors = [
        'rgba(30, 105, 221, 1)',  
        'rgba(59, 102, 167, 1)',  
        'rgba(70, 96, 135, 1)',
        'rgba(32, 189, 236, 1)' 
    ]




    # Función para crear el plot de cada gas
    def plot_gas(df_monthly_avg, gas, title):
        fig = go.Figure()

        # Iterar sobre cada mes y plotear las 24 horas en el mismo plot
        for i, mes in enumerate(df_monthly_avg['Mes'].unique()):
            group = df_monthly_avg[df_monthly_avg['Mes'] == mes]
            color = colors[i % len(colors)]  # Asignar color del colormap
            month_name = month_names[mes]  # Obtener el nombre del mes
            fig.add_trace(go.Scatter(x=group['Hora'], y=group[gas], mode='lines', line=dict(width=3, color=color), opacity=0.9, name=f'{month_name}'))

        # Configurar el plot
        fig.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_yaxes(title_text=gas, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_layout(title_text=f'{title} ({year})', showlegend=True, autosize=True, height=780, width=1520, plot_bgcolor='white')
        fig.show()

    # Plotear cada gas individualmente
    plot_gas(df_monthly_avg, CO, f'Ciclo Diurno de {CO}')
    plot_gas(df_monthly_avg, CO2, f'Ciclo Diurno de {CO2}')
    plot_gas(df_monthly_avg, CH4, f'Ciclo Diurno de {CH4}')





def ciclo_diurno_mensual_matplot(df, CO2=None, CH4=None, CO=None,start_month=1, end_month=12):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por mes y hora,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas utilizando matplotlib.
    También grafica el promedio de todo el DataFrame y los outliers.
    """
    if start_month & end_month is not None:
        df = df[df['Time'].dt.month.between(start_month, end_month)]

    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month

    # Calcular el promedio mensual para cada hora del día
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()

    # Calcular el promedio y la desviación estándar para todo el DataFrame
    df_avg = df_resampled.groupby('Hora').mean().reset_index()
    df_std = df_resampled.groupby('Hora').std().reset_index()

    # Obtener el año del DataFrame
    year = df.index.year[0]

    # Diccionario para mapear los números de los meses a sus nombres
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Definir el colormap   plasma  gist_rainbow summer
    sequential2_cmap = cm.get_cmap('viridis', 12)

    # Contar el número de argumentos no None
    gases = [(CO2, 'CO2'), (CH4, 'CH4'), (CO, 'CO')]
    gases = [(gas, label) for gas, label in gases if gas is not None]
    num_gases = len(gases)

    # Crear subplots dinámicamente en función del número de gases
    fig, axs = plt.subplots(num_gases, 1, figsize=(16, 4 * num_gases), sharex=True)

    # Asegurarse de que axs sea una lista incluso si hay solo un subplot
    if num_gases == 1:
        axs = [axs]

    # Función para crear el plot de cada gas
    def plot_gas(ax, df_monthly_avg, df_avg, df_std, gas, label):
        # Iterar sobre cada mes y plotear las 24 horas en el mismo plot
        for i, mes in enumerate(df_monthly_avg['Mes'].unique()):
            group = df_monthly_avg[df_monthly_avg['Mes'] == mes]
            color = sequential2_cmap(i / 11)  # Asignar color del colormap
            month_name = month_names[mes]  # Obtener el nombre del mes
            ax.plot(group['Hora'], group[gas], '-', linewidth=2, color=color, label=f'{month_name}')

        # Plotear el promedio de todo el DataFrame
        ax.plot(df_avg['Hora'], df_avg[gas], 'k--', linewidth=4, label='Promedio Anual')

        # Plotear los outliers
        ax.plot(df_avg['Hora'], df_avg[gas] + 2 * df_std[gas], 'k:', linewidth=3, label='Outliers Superior')
        ax.plot(df_avg['Hora'], df_avg[gas] - 2 * df_std[gas], 'k:', linewidth=3, label='Outliers Inferior')


        # Encontrar los valores máximos y mínimos
        max_value = df_monthly_avg[gas].max()
        min_value = df_monthly_avg[gas].min()
        max_month = df_monthly_avg.loc[df_monthly_avg[gas].idxmax(), 'Mes']
        min_month = df_monthly_avg.loc[df_monthly_avg[gas].idxmin(), 'Mes']

        bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.8)
        ax.text(0.95, 0.95, f'Valores máx y mínimos:\nMáx: {max_value:.2f} ({month_names[max_month]})\nMín: {min_value:.2f} ({month_names[min_month]})', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right', bbox=bbox_props)


        # Configurar el plot
        ax.set_xlabel('Hora del Día')
        if label == 'CH4':
            ax.set_ylabel('CH$_{4}$ (ppb)')
        else:
            ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.grid(True)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
        ax.set_xticks(range(24))
        ax.set_xticklabels([f'{hour:02d}:00' for hour in range(24)])  # Formato de hora hh:mm

          # Formato de hora hh:mm
    # Plotear cada gas individualmente
    for ax, (gas, label) in zip(axs, gases):
        plot_gas(ax, df_monthly_avg, df_avg, df_std, gas, label)

    # Ajustar la figura para que se adapte al tamaño de la ventana
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.suptitle(f'Ciclo Diurno de Gases de Efecto Invernadero, Estación Calakmul ({year})', fontsize=16)
    plt.show()



def plot_24h_anual_subplot(df, CO2=None, CH4=None, CO=None, start_month=1, end_month=12):
    """
    Esta función resamplea el DataFrame a intervalos de 1 hora, agrupa los datos por mes y hora,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas utilizando matplotlib.
    Cada mes se plotea en un subplot separado en una cuadrícula de 3x4.
    """
    if start_month & end_month is not None:
        df = df[df['Time'].dt.month.between(start_month, end_month)]

    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month


    # Calcular el promedio mensual para cada hora del día
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    # Calcular el promedio para todo el DataFrame
    df_avg = df_resampled.groupby('Hora').mean().reset_index()
    # Obtener el año del DataFrame
    year = df.index.year[0]

    # Diccionario para mapear los números de los meses a sus nombres
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Contar el número de argumentos no None
    gases = [(CO2, 'CO2'), (CH4, 'CH4'), (CO, 'CO')]
    gases = [(gas, label) for gas, label in gases if gas is not None]

    # Crear subplots 3x4
    fig, axs = plt.subplots(4, 3, figsize=(6, 10), sharey=True,sharex=True)

    # Función para crear el plot de cada gas
    def plot_gas(ax, df_monthly_avg, df_avg, gas, label, month):
        group = df_monthly_avg[df_monthly_avg['Mes'] == month]
        color= 'green'
        month_name = month_names[month]  # Obtener el nombre del mes
        ax.plot(group['Hora'], group[gas], '-', linewidth=2, color=color, label='L1b')

        # Plotear el promedio de todo el DataFrame
        ax.plot(df_avg['Hora'], df_avg[gas], 'r--', linewidth=1, label='Promedio Anual')

        ylim_min=df_monthly_avg[gas].min() - 5
        ylim_max=df_monthly_avg[gas].max() + 5


        # Configurar el plot
        ax.set_title(f'{month_name}', size=10)
        #ax.set_xlabel('Hora del Día')
        if label == 'CH4':
            ax.set_ylabel('CH$_{4}$ (ppb)')
        else:
            ax.set_ylabel(f'CO$_{2}$ (ppm)')
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='xx-small')
        #ax.set_xticks(range(24))
        #ax.set_xticklabels([f'{hour:02d}' for hour in range(24)], rotation=90)

        ax.set_ylim([ylim_min,ylim_max])

    # Plotear cada gas individualmente en cada subplot
    for month in range(start_month, end_month + 1):
        row = (month - 1) // 3
        col = (month - 1) % 3
        for gas, label in gases:
            plot_gas(axs[row, col], df_monthly_avg, df_avg, gas, label, month)

    for ax in axs.flat:
        ax.label_outer()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))



    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if label == 'CH4':
        fig.suptitle('Valores promedio de CH$_{4}$ 2024', fontsize=16)
    else:
        fig.suptitle('Valor Anual promedio de CO$_{2}$ (ppb)', fontsize=16)
    
    ax.grid(True)    

    plt.show()


def plot_24h_anual_subplot_comp(df, df2, CO2=None, CH4=None, CO=None, start_month=1, end_month=12):
    """
    Esta función resamplea dos DataFrames a intervalos de 1 hora, agrupa los datos por mes y hora,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas utilizando matplotlib.
    Cada mes se plotea en un subplot separado en una cuadrícula de 3x4, comparando los dos DataFrames.
    """
    if start_month & end_month is not None:
        df = df[df['Time'].dt.month.between(start_month, end_month)]
        df2 = df2[df2['Time'].dt.month.between(start_month, end_month)]

    df = df.set_index('Time')
    df2 = df2.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df2_resampled = df2.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df2_resampled['Hora'] = df2_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month
    df2_resampled['Mes'] = df2_resampled.index.month

    # Calcular el promedio mensual para cada hora del día
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    df2_monthly_avg = df2_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    # Calcular el promedio para todo el DataFrame
    df_avg = df_resampled.groupby('Hora').mean().reset_index()
    df2_avg = df2_resampled.groupby('Hora').mean().reset_index()
    # Obtener el año del DataFrame
    year = df.index.year[0]

    # Diccionario para mapear los números de los meses a sus nombres
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Contar el número de argumentos no None
    gases = [(CO2, 'CO2'), (CH4, 'CH4'), (CO, 'CO')]
    gases = [(gas, label) for gas, label in gases if gas is not None]

    # Crear subplots 3x4
    fig, axs = plt.subplots(4, 3, figsize=(7, 10), sharey=True, sharex=True)

    # Función para crear el plot de cada gas
    def plot_gas(ax, df_monthly_avg, df2_monthly_avg, df_avg, df2_avg, gas, label, month):
        group = df_monthly_avg[df_monthly_avg['Mes'] == month]
        group2 = df2_monthly_avg[df2_monthly_avg['Mes'] == month]
        color1 = 'orange'
        color2 = '#1062b4'
        month_name = month_names[month]  # Obtener el nombre del mes
        ax.plot(group['Hora'], group[gas], '-', linewidth=2, color=color1, label=f'L1')
        ax.plot(group2['Hora'], group2[gas], '-', linewidth=2, color=color2, alpha=0.8, label=f'L1b')

        # Plotear el promedio de todo el DataFrame
        #ax.plot(df_avg['Hora'], df_avg[gas], 'r--', linewidth=1, label='k')
        ax.plot(df2_avg['Hora'], df2_avg[gas], 'r--', linewidth=1, label='Promedio Anual L1b')

        ylim_min=df_monthly_avg[gas].min() - 5
        ylim_max=df_monthly_avg[gas].max() + 5

        # Configurar el plot
        ax.set_title(f'{month_name}', size=10)
        #ax.set_xlabel('2024')
        if label == 'CH4':
            ax.set_ylabel('CH$_{4}$ (ppb)')
        else:
            ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.grid(True)
        #ax.legend(loc='upper right', fontsize='x-small')

        ax.set_xticks(range(0, 30, 6))
        ax.set_xticklabels([f'{hour:02d}' for hour in range(0, 30, 6)], rotation=90)
        ax.set_ylim([ylim_min, ylim_max])

    # Plotear cada gas individualmente en cada subplot
    for month in range(start_month, end_month + 1):
        row = (month - 1) // 3
        col = (month - 1) % 3
        for gas, label in gases:
            plot_gas(axs[row, col], df_monthly_avg, df2_monthly_avg, df_avg, df2_avg, gas, label, month)


    for ax in axs.flat:
        ax.label_outer()

    # Crear una leyenda única para todos los subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small')
    fig.text(0.5, 0.04, '2024', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle('Valores mensuales de CO$_{2}$', fontsize=16)
    fig.subplots_adjust(top=0.88, bottom=0.1)  # Ajustar espacio para la leyenda
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small', bbox_to_anchor=(0.5, 0.95))
    plt.show()
   















def plot_24h_anual_subplot_comp_delta(df, df2=None, CO2=None, CH4=None, CO=None, start_month=1, end_month=12):
    """
    Esta función resamplea dos DataFrames a intervalos de 1 hora, agrupa los datos por mes y hora,
    y plotea los valores promedio de CO2, CH4 y CO en subplots para un período de 24 horas utilizando matplotlib.
    Cada mes se plotea en un subplot separado en una cuadrícula de 3x4.
    
    Si df2 está presente, compara los dos DataFrames.
    Si df2 es None, solo grafica el primero.
    
    También muestra el delta (diferencia máxima) de cada mes.
    """


    if start_month & end_month is not None:
        df = df[df['Time'].dt.month.between(start_month, end_month)]
        if df2 is not None:
            df2 = df2[df2['Time'].dt.month.between(start_month, end_month)]

    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month
    
    # Procesar el segundo DataFrame solo si está presente
    if df2 is not None:
        df2 = df2.set_index('Time')
        df2_resampled = df2.resample('1h').mean()
        df2_resampled['Hora'] = df2_resampled.index.hour
        df2_resampled['Mes'] = df2_resampled.index.month
        df2_monthly_avg = df2_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
        df2_avg = df2_resampled.groupby('Hora').mean().reset_index()
    else:
        # Crear dataframes vacíos para evitar errores cuando df2 es None
        df2_monthly_avg = None
        df2_avg = None

    # Calcular el promedio mensual para cada hora del día para df
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    
    # Calcular el promedio para todo el DataFrame para df
    df_avg = df_resampled.groupby('Hora').mean().reset_index()
    
    # Obtener el año del DataFrame
    year = df.index.year[0]

    # Diccionario para mapear los números de los meses a sus nombres
    month_names = {
        1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
        7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
    }

    # Contar el número de argumentos no None
    gases = [(CO2, 'CO2'), (CH4, 'CH4'), (CO, 'CO')]
    gases = [(gas, label) for gas, label in gases if gas is not None]

    # Crear subplots 3x4
    fig, axs = plt.subplots(4, 3, figsize=(7, 10), sharey=True, sharex=True)

    # Función para crear el plot de cada gas
    def plot_gas(ax, df_monthly_avg, df2_monthly_avg, df_avg, df2_avg, gas, label, month):
        group = df_monthly_avg[df_monthly_avg['Mes'] == month]
        
        # Configurar colores
        color1 = 'green' if df2 is None else 'orange'
        month_name = month_names[month]  # Obtener el nombre del mes
        
        # Plotear el primer dataset
        ax.plot(group['Hora'], group[gas], '-', linewidth=2, color=color1, label='Datos' if df2 is None else 'L1')
        
        # Calcular el delta (diferencia) entre los valores máximo y mínimo para df
        delta_df = group[gas].max() - group[gas].min()
        
        # Si df2 existe, plotear la comparación
        if df2_monthly_avg is not None:
            group2 = df2_monthly_avg[df2_monthly_avg['Mes'] == month]
            color2 = '#1062b4'
            ax.plot(group2['Hora'], group2[gas], '-', linewidth=2, color=color2, alpha=0.8, label='L1b')
            
            # Plotear el promedio anual del segundo dataset
            ax.plot(df2_avg['Hora'], df2_avg[gas], 'r--', linewidth=1, label='Promedio Anual L1b')
            
            # Calcular delta para df2
            delta_df2 = group2[gas].max() - group2[gas].min()
            
            # Añadir texto con ambos deltas
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.7)
            ax.text(0.95, 0.05, f'ΔL1: {delta_df:.2f}\nΔL1b: {delta_df2:.2f}', 
                    transform=ax.transAxes, fontsize=8, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=bbox_props)
        else:
            # Plotear el promedio anual del primer dataset
            ax.plot(df_avg['Hora'], df_avg[gas], 'r--', linewidth=1, label='Promedio Anual')
            
            # Añadir texto solo con el delta del primer dataset
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="white", alpha=0.7)
            ax.text(0.95, 0.05, f'Δ: {delta_df:.2f}', 
                    transform=ax.transAxes, fontsize=8, 
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=bbox_props)

        # Definir límites del eje Y
        ylim_min = df_monthly_avg[gas].min() - 5
        ylim_max = df_monthly_avg[gas].max() + 6

        # Configurar el plot
        ax.set_title(f'{month_name}', size=10)
        if label == 'CH4':
            ax.set_ylabel('CH$_{4}$ (ppb)')
        else:
            ax.set_ylabel('CO$_{2}$ (ppm)')
        ax.grid(True)

        ax.set_xticks(range(0, 30, 6))
        ax.set_xticklabels([f'{hour:02d}' for hour in range(0, 30, 6)], rotation=90)
        ax.set_ylim([ylim_min, ylim_max])

    # Plotear cada gas individualmente en cada subplot
    for month in range(start_month, end_month + 1):
        row = (month - 1) // 3
        col = (month - 1) % 3
        for gas, label in gases:
            plot_gas(axs[row, col], df_monthly_avg, df2_monthly_avg, df_avg, df2_avg, gas, label, month)

    for ax in axs.flat:
        ax.label_outer()

    # Crear una leyenda única para todos los subplots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.text(0.5, 0.04, f'{year}', ha='center', fontsize=12)
    
    # Calcular el delta promedio anual para cada gas
    for gas, label in gases:
        annual_delta_df = df_monthly_avg.groupby('Mes')[gas].apply(lambda x: x.max() - x.min()).mean()
        
        if df2 is not None:
            annual_delta_df2 = df2_monthly_avg.groupby('Mes')[gas].apply(lambda x: x.max() - x.min()).mean()
            fig.text(0.5, 0.01, f'Delta promedio anual {label}: ΔL1: {annual_delta_df:.2f}, ΔL1b: {annual_delta_df2:.2f}', 
                    ha='center', fontsize=10)
        else:
            fig.text(0.5, 0.01, f'Delta promedio anual {label}: Δ: {annual_delta_df:.2f}', 
                    ha='center', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Configurar el título según los gases utilizados
    if any('CO2' in gas for gas, _ in gases):
        title = 'Valores mensuales de CO$_{2}$'
    elif any('CH4' in gas for gas, _ in gases):
        title = 'Valores mensuales de CH$_{4}$'
    else:
        title = 'Valores mensuales'
        
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.88, bottom=0.1)  # Ajustar espacio para la leyenda
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small', bbox_to_anchor=(0.5, 0.95))
    plt.show()








def timeseries_delta_per_day(df, CO2=None, start_month=1, end_month=12):

    if start_month and end_month is not None:
        df = df[df['Time'].dt.month.between(start_month, end_month)]

    df = df.set_index('Time')
    df_resampled = df.resample('1h').mean()
    df_resampled['Hora'] = df_resampled.index.hour
    df_resampled['Mes'] = df_resampled.index.month
    df_resampled['Dia'] = df_resampled.index.day

    
    df_monthly_avg = df_resampled.groupby(['Mes', 'Hora']).mean().reset_index()
    
   
    timeseries_delta = pd.DataFrame()
    
    if CO2 is not None:
        

        #df_resampled    or df_monthly_avg
        #monthly_stats = df_resampled.groupby('Mes').agg({
        monthly_stats = df_monthly_avg.groupby('Mes').agg({
            CO2: ['max', 'min', 'mean']
        })
        
        
        monthly_stats.columns = ['_'.join(col).strip() for col in monthly_stats.columns.values]
        
        # Calculate delta (max - min)
        monthly_stats[f'{CO2}_delta'] = monthly_stats[f'{CO2}_max'] - monthly_stats[f'{CO2}_min']
        
       
        timeseries_delta = monthly_stats.reset_index()

        
    # Plot



        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        
        month_names = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        
        
        ax1.plot(timeseries_delta['Mes'], timeseries_delta[f'{CO2}_max'], 'o-', color='red', label='Máximo')
        ax1.plot(timeseries_delta['Mes'], timeseries_delta[f'{CO2}_min'], 'o-', color='blue', label='Mínimo')
        ax1.plot(timeseries_delta['Mes'], timeseries_delta[f'{CO2}_mean'], 'o-', color='green', label='Avg')
        
        # valor por mes
        for i, row in timeseries_delta.iterrows():
            # max
            ax1.annotate(f"{row[f'{CO2}_max']:.1f}", 
                        (row['Mes'], row[f'{CO2}_max']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7))
            
            #min
            ax1.annotate(f"{row[f'{CO2}_min']:.1f}", 
                        (row['Mes'], row[f'{CO2}_min']),
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.7))
            
            # avg
            ax1.annotate(f"{row[f'{CO2}_mean']:.1f}", 
                        (row['Mes'], row[f'{CO2}_mean']),
                        textcoords="data", 
                        
                        ha='right',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.7))
        
        
        ax1.set_xticks(timeseries_delta['Mes'])
        ax1.set_xticklabels([month_names[m] for m in timeseries_delta['Mes']])
        ax1.set_ylabel('CO$_2$ (ppm)', fontsize=12)
        
       


        ax1.set_xlabel('Mes', fontsize=12)
        
       
        ax2 = ax1.twinx()
        

        line_delta = ax2.plot(timeseries_delta['Mes'], timeseries_delta[f'{CO2}_delta'], '^--', 
                             color='purple', label='Δ (max-min)', markersize=8)
        
        ax2.set_ylabel(f'Δ {CO2} (max-min)', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # delta
        for i, row in timeseries_delta.iterrows():
            ax2.annotate(f"{row[f'{CO2}_delta']:.1f}", 
                       (row['Mes'], row[f'{CO2}_delta']),
                       textcoords="offset points", 
                       xytext=(10,0), 
                       ha='left',
                       fontsize=8,
                       color='black',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="purple", alpha=0.7))
        
        ax1.grid(True, alpha=0.3)
        
   
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
        
        
        year = df.index.year[0] if len(df.index) > 0 else ''
        plt.title(f'Observatorio Atmosférico Calakmul 2024\nEstadísticas de CO₂', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        timeseries_delta['CO2_Avg_delta'].round(3) # Redondear a 2 decimales
        return timeseries_delta







def plot_comparacion(*dfs, column='CO2_Avg'):
    plt.figure(figsize=(10, 6))
    
    for i, df in enumerate(dfs):
        if column in df.columns:
            plt.plot(df['Time'], df[column], label=f' {df}')
        else:
            print(f"Column '{column}' not found in DataFrame {i+1}")
    
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.title(f'Comparison of {column} across DataFrames')
    plt.legend()
    plt.show()



def plot_comparacion(*dfs, column='CO2_Avg'):
    month_names = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }
    
    plt.figure(figsize=(10, 6))
    
    for df in dfs:
        if isinstance(df, tuple) and len(df) == 2:
            df_name, df_data = df
            if column in df_data.columns:
                plt.plot(df_data['Time'], df_data[column], label=df_name)
            else:
                print(f"Column '{column}' no se encontro en '{df_name}'")
        else:
            print("se deben meter valores en tupla ('df_name', df_data)")
    
 
    plt.ylabel('CO$_{2}$ ppm')
    plt.title('Comparación de intervalos de tiempo de CO$_{2}$')
    plt.legend()
    
    # Set month names as x-tick labels
    months = dfs[0][1]['Time'].dt.month.unique()
    month_starts = [dfs[0][1][dfs[0][1]['Time'].dt.month == month]['Time'].iloc[0] for month in months]
    month_labels = [month_names[month] for month in months]

    plt.xticks(month_starts, month_labels, rotation=45)
    plt.grid()
    plt.show()








def plot_intervalos_subplot_4x1(df1, df2, column='CO2_Avg', intervalos=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')]):
    """
    Esta función toma dos DataFrames y una lista de intervalos de tiempo, filtra los datos según los intervalos,
    aplica ciclo_1d_avg y plotea los resultados en un subplot de 4x1.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        Primer DataFrame con datos de gases de efecto invernadero
    df2 : pandas.DataFrame
        Segundo DataFrame con datos de gases de efecto invernadero
    column : str, default='CO2_Avg'
        Nombre de la columna a graficar
    intervalos : list of tuples, default=[('19:00', '23:59'), ('00:00', '05:00'), ('09:00', '16:00')]
        Lista de tuplas con intervalos de tiempo en formato ('HH:MM', 'HH:MM')
    
    Returns:
    --------
    None
        Muestra el gráfico con 4 subplots mostrando diferentes intervalos horarios
    """
    fig, axs = plt.subplots(4, 1, figsize=(6, 10), sharex=True)

    '''
    df2_interval_full = intervalo_horas(df2, intervalos[0][0], intervalos[0][1])
    df2_avg_full = ciclo_1d_avg(df2_interval_full)
    df2_monthly_avg = df2_avg_full.set_index('Time').resample('ME').mean().reset_index()
    df2_monthly_avg['Time'] = df2_monthly_avg['Time'] + pd.offsets.MonthBegin(1) - pd.offsets.Day(15)
'''
    for i, (h0, hf) in enumerate(intervalos):
        df1_interval = intervalo_horas(df1, h0, hf)
        df2_interval = intervalo_horas(df2, h0, hf)

        df1_avg = ciclo_1d_avg(df1_interval)
        df2_avg = ciclo_1d_avg(df2_interval)

        axs[i].plot(df1_avg['Time'], df1_avg[column], label='L1', color='orange', alpha=1)
        axs[i].plot(df2_avg['Time'], df2_avg[column], label='L1b', color='#1062b4',alpha=1)

        df2_monthly_avg = df2_avg.set_index('Time').resample('ME').mean().reset_index()
        df2_monthly_avg['Time'] = df2_monthly_avg['Time'] + pd.offsets.MonthBegin(1) - pd.offsets.Day(15)
        
        axs[i].scatter(df2_monthly_avg['Time'], df2_monthly_avg[column], color='red', label='Promedio Mensual', s=30, zorder=5)
        axs[i].plot(df2_monthly_avg['Time'], df2_monthly_avg[column], color='red', linestyle='--', linewidth=1, zorder=4)
        axs[i].set_title(f'Horario:{h0}-{hf}', fontsize=10)
        axs[i].set_ylabel('CO$_{2}$ ppm')
        #axs[i].legend(loc='upper right',fontsize='x-small')
        axs[i].grid(True)
        #axs[i].set_ylim(405, 580)

    axs[-1].set_xlabel('2024')
    fig.suptitle('Promedios diarios de CO$_{2}$ para diferentes horarios', fontsize=14)
    # Set month names as x-tick labels
    months = df1['Time'].dt.month.unique()
    month_starts = [df1[df1['Time'].dt.month == month]['Time'].iloc[0] for month in months]
    month_names = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
    month_labels = [month_names[month] for month in months]

    for ax in axs:
        ax.set_xticks(month_starts)
        ax.set_xticklabels(month_labels, rotation=45)
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Crear una leyenda única para todos los subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small', bbox_to_anchor=(0.5, 0.95))



    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()