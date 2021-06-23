#%%
import pandas as pd, json
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import os
import openpyxl
import gmaps

# %%
df_map=pd.read_csv('/Users/jhasneha/Documents/DOE/summer2021/DOE_ag/Soil data/SEPAC/J4/j4_2007_J4_nobuffer_harvest.csv')
df_map['Latitude']=df_map['Latitude'].astype(float)
df_map['Longitude']=df_map['Longitude'].astype(float)


#%%

m = folium.Map(location=[45.5236, -122.6750], zoom_start=12)
m
#.save("/Users/jhasneha/Documents/Spring2021/SPR_indot/SPRprojectcodes/html_files/map.html")
# %%
df_elev=df_map[["Latitude","Longitude","ELEVATION_"]].copy()
# %%
def df_to_geojson(df, properties, lat='Latitude', lon='Longitude'):
    # create a new python dict to contain our geojson data, using geojson format
    geojson = {'type':'FeatureCollection', 'features':[]}

    # loop through each row in the dataframe and convert each row to geojson format
    for _, row in df.iterrows():
        # create a feature template to fill in
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point',
                               'coordinates':[]}}

        # fill in the coordinates
        feature['geometry']['coordinates'] = [row[lon],row[lat]]

        # for each column, get the value and add it as a new feature property
        for prop in properties:
            feature['properties'][prop] = row[prop]
        
        # add this feature (aka, converted dataframe row) to the list of features inside our dict
        geojson['features'].append(feature)
    
    return geojson


#%%
props=["ELEVATION_"]
geojson = df_to_geojson(df_elev, properties=props)

#%%
# save the geojson result to a file
# save the geojson result to a file
output_filename = 'dataset.js'
with open(output_filename, 'w') as output_file:
    output_file.write('var dataset = {};'.format(json.dumps(geojson)))
    
# how many features did we save to the geojson file?
print('{} geotagged features saved to file'.format(len(geojson['features'])))# %%


#%%
overlay= os.path.join('data','dataset.js')
folium.GeoJson(overlay, name="trial").add_to(m)

# %%
import geojson

# %%
def data2geojson(df):
    features = []
    insert_features = lambda X: features.append(
            geojson.Feature(geometry=geojson.Point((X["long"],
                                                    X["lat"],
                                                    X["elev"])),
                            properties=dict(name=X["name"],
                                            description=X["description"])))
    df.apply(insert_features, axis=1)
    with open('map1.geojson', 'w', encoding='utf8') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False)

col = ['lat','long','elev','name','description']
data = [[-29.9953,-70.5867,760,'A','Place ñ'],
        [-30.1217,-70.4933,1250,'B','Place b'],
        [-30.0953,-70.5008,1185,'C','Place c']]

df = pd.DataFrame(data, columns=col)

data2geojson(df)
# %%

#%%
#%%