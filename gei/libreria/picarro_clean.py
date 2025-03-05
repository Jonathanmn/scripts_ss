import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors



'''herramienta para desplegar el plotly y eliminar datos de forma visual '''

def clean_plotly(df, column):
    """ un scatter que selecciona puntos y los elimina del data frame """
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

def ciclo_diurno(df, CO2, CH4, CO):
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

    # Agrupar por la columna 'Hora' para obtener el promedio de cada hora del día
    df_grouped = df_resampled.groupby('Hora').mean()

    # Crear subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Plot para CO2_Avg
    axs[0].plot(df_grouped.index, df_grouped[CO2], '-', color='black', linewidth=1, alpha=0.7)
    axs[0].set_title(f'Ciclo Diurno de {CO2}')
    axs[0].set_ylabel(CO2)
    axs[0].grid(True)

    # Plot para CH4_Avg
    axs[1].plot(df_grouped.index, df_grouped[CH4], '-', color='black', linewidth=1, alpha=0.7)
    axs[1].set_title(f'Ciclo Diurno de {CH4}')
    axs[1].set_ylabel(CH4)
    axs[1].grid(True)

    # Plot para CO_Avg
    axs[2].plot(df_grouped.index, df_grouped[CO], '-', color='black', linewidth=1, alpha=0.7)
    axs[2].set_title(f'Ciclo Diurno de {CO}')
    axs[2].set_xlabel('Hora del Día')
    axs[2].set_ylabel(CO)
    axs[2].grid(True)

    # Ajustar la figura para que se adapte al tamaño de la ventana
    fig.tight_layout()
    fig.suptitle('Ciclo Diurno de Gases de Efecto Invernadero', fontsize=16)
    plt.show()




def ciclo_diurno_2(df, CO2, CH4, CO):
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

    # Crear subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Iterar sobre cada día y plotear las 24 horas en el mismo subplot
    for day, group in df_resampled.groupby(df_resampled.index.date):
        # Plot para CO2_Avg
        axs[0].plot(group['Hora'], group[CO2], '-', linewidth=1, alpha=0.7, label=str(day))
        # Plot para CH4_Avg
        axs[1].plot(group['Hora'], group[CH4], '-', linewidth=1, alpha=0.7, label=str(day))
        # Plot para CO_Avg
        axs[2].plot(group['Hora'], group[CO], '-', linewidth=1, alpha=0.7, label=str(day))

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

    # Definir un colormap personalizado de azul a violeta
    colors = px.colors.qualitative.Plotly

    # Función para crear el plot de cada gas
    def plot_gas(df_monthly_avg, gas, title):
        fig = go.Figure()

        # Iterar sobre cada mes y plotear las 24 horas en el mismo plot
        for mes in df_monthly_avg['Mes'].unique():
            group = df_monthly_avg[df_monthly_avg['Mes'] == mes]
            color = colors[mes % len(colors)]  # Asignar color del colormap
            fig.add_trace(go.Scatter(x=group['Hora'], y=group[gas], mode='lines', line=dict(width=3, color=color), opacity=0.9, name=f'Mes {mes}'))

        # Configurar el plot
        fig.update_xaxes(title_text='Hora del Día', tickmode='linear', dtick=1, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_yaxes(title_text=gas, showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_layout(title_text=f'{title} ({year})', showlegend=True, autosize=True, height=780, width=1520, plot_bgcolor='white')
        fig.show()

    # Plotear cada gas individualmente
    plot_gas(df_monthly_avg, CO, f'Ciclo Diurno de {CO}')
    plot_gas(df_monthly_avg, CO2, f'Ciclo Diurno de {CO2}')
    plot_gas(df_monthly_avg, CH4, f'Ciclo Diurno de {CH4}')









