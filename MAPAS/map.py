import folium
import geopandas as gpd
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import contextily as ctx
import pandas as pd
import osmnx as ox

'''
# Define the ZMVM ROI polygon coordinates
zmvm_coords = [
    [-99.63973291477484, 20.20023328768361],
    [-99.63973291477484, 18.83340172041972],
    [-98.55208643039984, 18.83340172041972],
    [-98.55208643039984, 20.20023328768361],
    [-99.63973291477484, 20.20023328768361]  # Close the polygon
]

# Define the points
unam_point = [-99.1761, 19.3262]
vallejo_point = [-99.1470, 19.4830]

# Calculate center of the area for map centering
center_lat = (20.20023328768361 + 18.83340172041972) / 2
center_lon = (-99.63973291477484 + -98.55208643039984) / 2

# Create a folium map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=9,
    tiles='OpenStreetMap'
)

# Add the ZMVM ROI polygon
folium.Polygon(
    locations=[[coord[1], coord[0]] for coord in zmvm_coords],  # Note: folium uses [lat, lon]
    color='white',
    weight=2,
    fillColor='blue',
    fillOpacity=0.3,
    popup='ZMVM ROI'
).add_to(m)

# Add UNAM point
folium.Marker(
    location=[unam_point[1], unam_point[0]],  # [lat, lon]
    popup='UNAM Point',
    tooltip='UNAM',
    icon=folium.Icon(color='red', icon='info-sign')
).add_to(m)

# Add Vallejo point
folium.Marker(
    location=[vallejo_point[1], vallejo_point[0]],  # [lat, lon]
    popup='Vallejo Point',
    tooltip='Vallejo',
    icon=folium.Icon(color='green', icon='info-sign')
).add_to(m)

# Add different basemap options
folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('Stamen Toner').add_to(m)
folium.TileLayer('CartoDB positron').add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
m.save('zmvm_map.html')

# Display the map (if running in Jupyter notebook)
'''

# Create the polygon
zmvm_polygon = Polygon([
    [-99.63973291477484, 20.20023328768361],
    [-99.63973291477484, 18.83340172041972],
    [-98.55208643039984, 18.83340172041972],
    [-98.55208643039984, 20.20023328768361]
])

# Create points
points_data = {
    'name': ['UNAM', 'Vallejo'],
    'geometry': [
        Point(-99.1761, 19.3262),
        Point(-99.1470, 19.4830)
    ]
}

# Create GeoDataFrames
polygon_gdf = gpd.GeoDataFrame([1], geometry=[zmvm_polygon], crs='EPSG:4326')
points_gdf = gpd.GeoDataFrame(points_data, crs='EPSG:4326')

# Convert to Web Mercator for basemap
polygon_gdf = polygon_gdf.to_crs('EPSG:3857')
points_gdf = points_gdf.to_crs('EPSG:3857')

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot the polygon with label for legend
polygon_gdf.boundary.plot(ax=ax, color='red', linewidth=0.8, alpha=0, )
polygon_gdf.plot(ax=ax, alpha=0, color='blue')

# Plot the points with labels for legend
points_gdf[points_gdf['name'] == 'UNAM'].plot(ax=ax, color='red', markersize=100, alpha=1, label='UNAM Station')
points_gdf[points_gdf['name'] == 'Vallejo'].plot(ax=ax, color='green', markersize=100, alpha=1, label='Vallejo Station')

# Add point labels
for idx, row in points_gdf.iterrows():
    ax.annotate(row['name'], 
                (row.geometry.x, row.geometry.y),
                xytext=(10, 10), textcoords='offset points',
                fontsize=12, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.5))

# Get Mexico City administrative boundary
try:
    # Download Mexico City boundary from OpenStreetMap
    mexico_city = ox.geocode_to_gdf('Mexico City, Mexico')
    
    # Convert to Web Mercator for consistency   
    mexico_city = mexico_city.to_crs('EPSG:3857')
    
    # Plot Mexico City boundary with label
    mexico_city.boundary.plot(ax=ax, color='yellow', linewidth=2, alpha=0.8, label='Mexico City')

    
except Exception as e:
    print(f"Could not download Mexico City boundary: {e}")

try:
    # Try different queries for ZMVM
    zmvm_boundary = ox.geocode_to_gdf('Zona Metropolitana del Valle de México, Mexico')
    
    # Convert to Web Mercator for consistency
    zmvm_boundary = zmvm_boundary.to_crs('EPSG:3857')
    
    # Plot ZMVM boundary with label
    zmvm_boundary.boundary.plot(ax=ax, color='orange', linewidth=3, alpha=0.9, linestyle='-', label='ZMVM')
    zmvm_boundary.plot(ax=ax, alpha=0, color='cyan')
    
except Exception as e:
    print(f"Could not download ZMVM boundary: {e}")
    
ctx.add_basemap(ax, crs=polygon_gdf.crs, source=ctx.providers.Esri.WorldImagery, alpha=0.9)

#add basemap esri terrain
ctx.add_basemap(ax, crs=polygon_gdf.crs, source=ctx.providers.Esri.WorldTerrain, alpha=0.4)

# Add legend in upper left
ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True, 
          fontsize=10, title='Legend', title_fontsize=12, facecolor='white', edgecolor='black', framealpha=0.9)

# Set title and labels
ax.set_title('Study Areas', fontsize=16, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
#save figure
plt.savefig('study_areas.png', dpi=300)

plt.tight_layout()
plt.show()
