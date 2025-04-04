from picarro import *



'''

file_path='/home/jmn/Headers.xlsx'

headers=header(file_path, sheet_name='stations', header=0)

print(headers.columns)




site_value = 'altz'
variables = extract_variables(file_path, sheet_name='stations', header=0, site_value=site_value)
if variables:
    name, state, north, west, masl, ut = variables
    print(f"Name: {name}")
    print(f"State: {state}")
    print(f"North: {north}")
    print(f"West: {west}")
    print(f"MASL: {masl}")
    print(f"UT: {ut}")
else:
    print(f"No data found for site: {site_value}")

'''
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from windrose import WindroseAxes
import numpy as np

import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from windrose import WindroseAxes
import numpy as np
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory

# Crear un GeoDataFrame con la ubicación de la rosa de viento
data = {'name': ['Calakmul'], 'lat': [18.5], 'lon': [-89.5]}
gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']), crs="EPSG:4326")

# Convertir a la proyección Web Mercator (necesaria para contextily)
gdf = gdf.to_crs(epsg=3857)

# Crear una figura cuadrada
fig, ax = plt.subplots(figsize=(8, 8))

# Calcular los límites para centrar en Calakmul y hacer zoom
buffer_distance = 20000  # Buffer de 20 km alrededor del punto (ajusta según necesidad)
bounds = gdf.geometry.buffer(buffer_distance).total_bounds
ax.set_xlim(bounds[0], bounds[2])
ax.set_ylim(bounds[1], bounds[3])


'''
'''


# Agregar un mapa base con zoom
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap, zoom=10)
#usgs_url = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
#ctx.add_basemap(ax, source=usgs_url, zoom=10)

# Plotear el punto de interés
gdf.plot(ax=ax, color='red', markersize=100, alpha=0.7, label="Ubicación")

# Crear datos de ejemplo para la rosa de viento
wind_directions = np.random.uniform(0, 360, 100)  # Direcciones del viento (grados)
wind_speeds = np.random.uniform(0, 10, 100)       # Velocidades del viento (m/s)

# Coordenadas de la rosa de viento (en proyección Web Mercator)
x, y = gdf.geometry.x[0], gdf.geometry.y[0]  # Coordenadas de Calakmul en Web Mercator

# Obtener las coordenadas del punto en los ejes
x_ax, y_ax = x, y

# Convertir las coordenadas del mapa a coordenadas relativas de la figura
def axis_to_fig(ax, coord_x, coord_y):
    # Obtener la transformación del eje a la figura
    fig_x = (coord_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
    fig_y = (coord_y - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    # Ajustar por el tamaño relativo del eje en la figura
    bbox = ax.get_position()
    fig_x = bbox.x0 + fig_x * bbox.width
    fig_y = bbox.y0 + fig_y * bbox.height
    
    return fig_x, fig_y

# Obtener coordenadas relativas
fig_x, fig_y = axis_to_fig(ax, x_ax, y_ax)

# Tamaño relativo de la rosa de viento (ajusta según necesidad)
windrose_size = 0.3  # 30% del tamaño de la figura

# Calcular las coordenadas de la esquina inferior izquierda y tamaño
rect = [fig_x - windrose_size/2, fig_y - windrose_size/2, windrose_size, windrose_size]

# Crear una rosa de viento usando WindroseAxes.from_ax
ax_inset = WindroseAxes.from_ax(fig=fig, rect=rect)
ax_inset.bar(wind_directions, wind_speeds, normed=True, opening=0.8, edgecolor='white', bins=4)
ax_inset.set_title("Rosa de viento", fontsize=10)

# Personalizar el gráfico
plt.legend(loc="lower left")
plt.title("Mapa con Rosa de Viento (OpenStreetMap)", fontsize=14)

# Asegurarse de que la figura sea cuadrada incluso después de agregar elementos
plt.tight_layout()
plt.show()


