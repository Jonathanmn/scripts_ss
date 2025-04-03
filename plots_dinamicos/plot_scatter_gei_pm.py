from windrose_lib import *





folder_met = './DATOS/met/L2/minuto'
folder_t64 = './DATOS/pm/L0/minuto'
folder_gei = './DATOS/gei/L1/minuto/2024' 





gei = read_L0_or_L1(folder_gei, 'yyyy-mm-dd HH:MM:SS', header=7)
gei = reverse_rename_columns(gei)






t64 = t64_cmul(folder_t64)


met = met_cmul(folder_met)   
t64=t64[['Date & Time (Local)','PM10 Conc', 'PM2.5 Conc']]
met= met[['yyyy-mm-dd HH:MM:SS', 'WDir_Avg', 'WSpeed_Avg']]
gei = gei[['Time', 'CO2_Avg', 'CH4_Avg','CO_Avg']]





merged_data = merge_df(
    [met,gei,t64],
    ['yyyy-mm-dd HH:MM:SS', 'Time', 'Date & Time (Local)'])







import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def correlacion(df, column1, column2):
    """
    Realiza un scatter plot para mostrar la correlación entre dos columnas
    y añade el valor de la correlación de Pearson en el gráfico.
    """
    # Eliminar filas con valores NaN en cualquiera de las dos columnas
    filtered_df = df[[column1, column2]].dropna()

    # Calcular la correlación de Pearson
    correlation, _ = pearsonr(filtered_df[column1], filtered_df[column2])

    # Crear el scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x=filtered_df[column1], y=filtered_df[column2], alpha=0.7, color='red', s=1)

    # Añadir el valor de la correlación en el gráfico
    plt.title(f'Scatter Plot: {column1} vs {column2}\nPearson Correlation: {correlation:.2f}', fontsize=14)
    plt.xlabel(column1, fontsize=12)
    plt.ylabel(column2, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()




def correlacion1(df, column1, column2):
    """
    Realiza un scatter plot para mostrar la correlación entre dos columnas,
    añade el valor de la correlación de Pearson, R² y una línea de regresión.
    """
    # Eliminar filas con valores NaN en cualquiera de las dos columnas
    filtered_df = df[[column1, column2]].dropna()

    # Calcular la correlación de Pearson
    correlation, _ = pearsonr(filtered_df[column1], filtered_df[column2])

    # Calcular el valor de R²
    r_squared = correlation ** 2

    # Calcular la línea de regresión
    slope, intercept = np.polyfit(filtered_df[column1], filtered_df[column2], 1)
    regression_line = slope * filtered_df[column1] + intercept

    # Crear el scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x=filtered_df[column1], y=filtered_df[column2], alpha=1, color='red', s=1, label='Datos')
    
    # Añadir la línea de regresión
    plt.plot(filtered_df[column1], regression_line, color='black', linewidth=1, label='Línea de regresión')

    # Añadir el valor de la correlación y R² en el gráfico
    plt.title(
        f'Scatter Plot: {column1} vs {column2}\n'
        f'Pearson Correlation: {correlation:.2f}\n'
        f'R²: {r_squared:.2f}',
        fontsize=14
    )
    plt.xlabel(column1, fontsize=12)
    plt.ylabel(column2, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()






correlacion1(merged_data, 'CH4_Avg', 'CO2_Avg')












'''

# Excluir la columna 'timestamp' del DataFrame para el heatmap
columns_to_exclude = ['Timestamp']  # Ajusta el nombre exacto si es diferente
filtered_data = merged_data.drop(columns=columns_to_exclude, errors='ignore')

sns.heatmap(filtered_data.corr(), vmin=-1, vmax=1, annot=True, cmap="rocket_r")
plt.show()'
''
''
'''