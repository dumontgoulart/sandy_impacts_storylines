#!/usr/bin/env python3
""" Add city names to the list and obtain the corresponding coords (lat/lon)

If running from terminal: <python cities_storms_creation_script.py>
If running as a module imported in another script:
    <import cities_storms_creation_script
    df_test = cities_storms_creation_script.city_coords_retrieval()> """

import os
import numpy as np 
import pandas as pd
from geopy.geocoders import Nominatim
os.chdir('D:/paper_3/code')

# Storm locations to explore - add the cities wanted
list_cities = ['Whitestrand', 'Port Logan', 'La Rochelle', 'Rotterdam','Lowestoft', 
        'Cuxhaven', 'Aalborg', 'Hamburg', 'Vigo', 'Edinburgh', 'La Coruna',
        'Newcastle', 'Dover', 'Bilbao', 'Venice', 'New York', 'Tacloban', 'Santiago de Cuba', 'Townsville']

# Function
def city_coords_retrieval(cities = list_cities):
    geolocator = Nominatim(user_agent="EU_extratropical_storms")

    cities_coords =[geolocator.geocode(city) for city in cities]
    list_lat = [item.latitude for item in cities_coords]
    list_lon = [item.longitude for item in cities_coords]

    df_cities_coords = pd.DataFrame(data = zip(cities, list_lat, list_lon), columns = ['cities','lat','lon'])
    return df_cities_coords

if __name__ == '__main__':
    df_cities_coords = city_coords_retrieval(list_cities)
    df_cities_coords.to_csv('../data/list_cities_storm.csv')