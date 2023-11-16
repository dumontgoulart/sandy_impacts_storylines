"""
To activate virtual environment: ctrl+shift+p >> "Select Interpreter" >> "hydromt"

Created on Tue Nov  9 16:39:47 2021
@author: morenodu
"""
import os
os.chdir('D:/paper_3/code')
import xarray as xr 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import storm_functions as sf
import seaborn as sns
import sn_storm_preprocessing as prep
from shapely.geometry import Point, MultiPolygon, Polygon
import geopandas as gpd
# from geopy.distance import distance
from haversine import haversine
from scipy.spatial.distance import cdist
from datetime import datetime

import metpy.calc as mpcalc
import metpy.xarray as mpxarray
from scipy.signal import savgol_filter
from scipy.interpolate import BSpline
from scipy.ndimage import laplace

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

def legend_without_duplicate_labels(ax, loc = 'center left', bbox_to_anchor=(1, 0.5)):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc=loc,bbox_to_anchor=bbox_to_anchor, frameon = False)

def find_min_pressure_track(ds):
    if 'member' in ds.keys():
        df_min = ds['mslp'].where(ds['mslp']==ds['mslp'].min(dim=['lat','lon']), drop=True).to_dataframe().dropna().reset_index(['lat','lon','member'])
    else:
        df_min = ds['mslp'].where(ds['mslp']==ds['mslp'].min(dim=['lat','lon']), drop=True).to_dataframe().dropna().reset_index(['lat','lon'])
    return df_min

def plot_minimum_track_single(df_factual, df_ws, storm_name):
    fig = plt.figure(figsize=(6, 8))
    central_lon = df_factual.lon.min() + (df_factual.lon.max() - df_factual.lon.min())/2
    central_lat = df_factual.lat.min() + (df_factual.lat.max() - df_factual.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
    ax.plot(df_factual['lon'], df_factual['lat'],transform=ccrs.PlateCarree(), color = 'orange', label = 'factual')
    plt.title(f'{storm_name} track for mean GW levels')
    legend_without_duplicate_labels(ax,loc = 'center left')
    plt.tight_layout()
    plt.show()
    # plt.close()
    return fig

def plot_minimum_track(df_counter, df_factual, df_plus2, df_ws, storm_name):
    fig = plt.figure(figsize=(6, 8))
    central_lon = df_factual.lon.min() + (df_factual.lon.max() - df_factual.lon.min())/2
    central_lat = df_factual.lat.min() + (df_factual.lat.max() - df_factual.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    if 'member' in df_counter.columns:
        sns.lineplot(ax=ax, data = df_counter, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member', color = 'blue', label = 'counter') 
        sns.lineplot(ax=ax, data = df_factual, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member',color = 'orange', label = 'factual') 
        sns.lineplot(ax=ax, data = df_plus2, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member',color = 'green', label = 'plus2') 
        plt.title(f'{storm_name} track for ens. GW levels')
    else:
        ax.plot(df_counter['lon'], df_counter['lat'],transform=ccrs.PlateCarree(), color = 'blue', label = 'counter')
        ax.plot(df_factual['lon'], df_factual['lat'],transform=ccrs.PlateCarree(), color = 'orange', label = 'factual')
        ax.plot(df_plus2['lon'], df_plus2['lat'],transform=ccrs.PlateCarree(), color = 'green', label = 'plus2')
        plt.title(f'{storm_name} track for mean GW levels')
    legend_without_duplicate_labels(ax,loc = 'center left')
    plt.tight_layout()
    plt.show()
    # plt.close()
    return fig

def plot_minimum_track_validation(df_hist, df_counter, df_factual, df_plus2, df_ws, storm_name):
    fig = plt.figure(figsize=(6, 8))
    central_lon = df_factual.lon.min() + (df_factual.lon.max() - df_factual.lon.min())/2
    central_lat = df_factual.lat.min() + (df_factual.lat.max() - df_factual.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    # Plot historical track
    ax.plot(df_hist['lon'], df_hist['lat'],transform=ccrs.PlateCarree(), color = 'black', label = 'ECHAM hist')
    # Plot counterfactual trackS
    sns.lineplot(ax=ax, data = df_counter, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member', color = 'blue', label = 'counter') 
    sns.lineplot(ax=ax, data = df_factual, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member',color = 'orange', label = 'factual') 
    sns.lineplot(ax=ax, data = df_plus2, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, sort=False, style = 'member',color = 'green', label = 'plus2') 
    plt.title(f'{storm_name} track for ens. GW levels')

    legend_without_duplicate_labels(ax,loc = 'center left')
    plt.tight_layout()
    plt.show()
    return fig

# Storm tracker inspired by Nadia
def storm_tracker_mslp(ds, df_ws, smooth_step = None, grid_conformity = False, large_box = 6, small_box = 3):
    '''
    This function has as purpose to create a storm track based on a dataset (ds) that has mean sea level pressure (mslp) and wind speed (u10m, v10m).
    Steps are: First we find the observed position available of the storm eye in the IBTraCS database. Then we draw a box of 5 degrees per 5 degrees and
    locate the point with maximum vorticity. If this point has lower MSLP, the storm eye centre is updated. 
    After that, we draw a box of 2.5 x 2.5 degrees around the storm eye and detect the lowest MSLP point. If it corresponds to a new position, update it.
    '''    
    # Calculate the vorticity for all timesteps
    dx, dy = mpxarray.grid_deltas_from_dataarray(ds['u10m'])
    ds_vorticity = mpcalc.vorticity(u = ds['u10m'], v = ds['v10m'], dx = dx, dy = dy)
    # add ds_vorticity to ds
    ds['vorticity'] = ds_vorticity


    # time = '2010-02-28T00:00:00.000000000' 
    # ds_test = ds['mslp'].sel(time =time).copy()
    
    # # Find laplacian operator for MSLP
    # mslp_laplacian = laplace(ds_test)
    # da_mslp_laplacian = xr.DataArray(mslp_laplacian, dims=ds_test.dims, coords=ds_test.coords)
    # da_mslp_laplacian_mask = da_mslp_laplacian.where(da_mslp_laplacian > da_mslp_laplacian.quantile(q=0.90).values)

    # # Calculate gradient along lat and lon using xarray.DataArray.differentiate
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines(resolution='50m', linewidth=0.5) 
    # ds['vorticity'].sel(time = time ).plot(ax=ax, robust = True)
    # ds_test.plot.contour(ax=ax, colors='k', levels=10)
    # ax.set_title('Vorcitity')
    # plt.show()

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # ax.coastlines(resolution='50m', linewidth=0.5) 
    # da_mslp_laplacian_mask.plot(ax=ax, cmap='RdBu_r', robust = True)
    # ds_test.plot.contour(ax=ax, colors='k', levels=10)
    # ax.set_title('Laplacian of MSLP')
    # plt.show()

    # Find common time steps between dataset and historical track
    common_times = np.intersect1d(ds['time'].values, df_ws['time'].values)  # Find common time values
    ds = ds.sel(time=common_times)  # Select common time values in dataset

    ds_relative_track = []
    for time_step in ds.sel(time=slice(df_ws['time'].iloc[0],df_ws['time'].iloc[-1])).time.values:
        lat_ini = df_ws.loc[df_ws['time'] == time_step, 'lat'].values[0]
        lon_ini = df_ws.loc[df_ws['time'] == time_step, 'lon'].values[0]
        # Initial dataset at time T
        ds_box = ds.sel(time = time_step)
        # 1) Determine eye position with respect to IBTracs historical records
        da_eye_track = ds_box.sel(lat = lat_ini, lon = lon_ini, method = 'nearest')
        
        # 2) Draw a 5x5 degree box around 1) and find maximum vorticity position:
        ds_box5 = ds_box.sel(lat = slice(da_eye_track['lat'] + large_box, da_eye_track['lat'] - large_box),
                             lon = slice(da_eye_track['lon'] - large_box, da_eye_track['lon'] + large_box))
        coords_max_vort = ds_box5.where(ds_box5['vorticity'] == ds_box5['vorticity'].max(dim=['lat','lon'])).to_dataframe()['vorticity'].dropna().reset_index(['lat','lon'])
        
        # 2-a) update values if vorticity position indicates lower pressure:
        if ds_box5['mslp'].sel(lat = coords_max_vort['lat'].values, lon = coords_max_vort['lon'].values, method = 'nearest').values < da_eye_track['mslp'].values:
            print('Step 2: update location minimum pressure for vorticity maximum')
            da_eye_track = ds_box5.sel(lat = coords_max_vort['lat'].values, lon = coords_max_vort['lon'].values, method = 'nearest')

        # 3) Establish minimum pressure within 2.5 degrees from eye of the storm:
        ds_box_small = ds_box.sel(lat = slice(da_eye_track['lat'].values.item() + small_box, da_eye_track['lat'].values.item() - small_box), lon = slice(da_eye_track['lon'].values.item() - small_box, da_eye_track['lon'].values.item() + small_box))
        coords_minimum_small = ds_box_small.where(ds_box_small['mslp'] == ds_box_small['mslp'].min(dim=['lat','lon'])).to_dataframe()['mslp'].dropna().reset_index(['lat','lon'])
        coords_minimum_small = coords_minimum_small.mean(numeric_only=True).to_frame().T
        # 3-a) update values if new position indicates lower pressure:
        if ds_box_small['mslp'].sel(lat = coords_minimum_small['lat'].values, lon = coords_minimum_small['lon'].values, method = 'nearest').values <= da_eye_track['mslp'].values:
            print('Step 3: update location minimum pressure for minimum MSLP')
            da_eye_track = ds_box_small.sel(lat = coords_minimum_small['lat'].values, lon = coords_minimum_small['lon'].values, method = 'nearest')

        ds_relative_track.append(da_eye_track.to_dataframe()) # check out the to_dataframe. Keep it as dataset. Concatenate 
    df_relative_track = pd.concat(ds_relative_track).reset_index(['lat','lon'])
    df_relative_track = df_relative_track.set_index('time')

    # df_relative_track_og = df_relative_track.copy()
    # df_relative_track_smooth = df_relative_track.copy()
    # df_relative_track_bs = df_relative_track.copy()

    # time_snapshot = '2013-12-05 12:00:00'

    # fig = plt.figure(figsize=(6, 8))
    # central_lon = df_relative_track.lon.min() + (df_relative_track.lon.max() - df_relative_track.lon.min())/2
    # central_lat = df_relative_track.lat.min() + (df_relative_track.lat.max() - df_relative_track.lat.min())/2
    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    # ax.coastlines(resolution='10m') 
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    # ax.plot(df_relative_track['lon'], df_relative_track['lat'],transform=ccrs.PlateCarree(), color = 'black', label = 'og')
    # ax.plot(df_relative_track_og['lon'], df_relative_track_og['lat'],transform=ccrs.PlateCarree(), color = 'blue', label = 'savgol')
    # ax.plot(df_relative_track_smooth['lon'], df_relative_track_smooth['lat'],transform=ccrs.PlateCarree(), color = 'red', label = 'smooth 4')
    # # ax.plot(df_relative_track_bs['lon'], df_relative_track_bs['lat'],transform=ccrs.PlateCarree(), color = 'yellow', label = 'bs')

    # ax.scatter(df_relative_track.loc[time_snapshot]['lon'], df_relative_track.loc[time_snapshot]['lat'], s=50, marker='*',transform=ccrs.PlateCarree())
    # ax.scatter(df_relative_track_og.loc[time_snapshot]['lon'], df_relative_track_og.loc[time_snapshot]['lat'], s=50, marker='*',transform=ccrs.PlateCarree())
    # ax.scatter(df_relative_track_smooth.loc[time_snapshot]['lon'], df_relative_track_smooth.loc[time_snapshot]['lat'], s=50, marker='*',transform=ccrs.PlateCarree())
               
    # plt.legend()
    # plt.show()

    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat+10))    # 
    # ax.coastlines(resolution='10m') 
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ds['vorticity'].sel(time = time_snapshot).plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', robust=True)
    # plt.show()


    # FILTERING THE TRACKS TO MAKE THEM SMOOTHER
    if smooth_step == 'savgol':
        print(f'smooth: {smooth_step}')
        # define window length and polynomial order for Savitzky-Golay filter, or use window_length to 9 because that's the minimum for one day
        window_length = max(int(np.ceil(len(df_relative_track)/10) // 2 * 2 + 1 ), 9)
        poly_order = 2
        # smooth lat and lon coordinates separately using Savitzky-Golay filter
        df_relative_track['lat'] = savgol_filter(df_relative_track['lat'], window_length, poly_order)
        df_relative_track['lon'] = savgol_filter(df_relative_track['lon'], window_length, poly_order)

    elif smooth_step == 'bspline':
        print(f'smooth: {smooth_step}') # define degree and knots for B-spline 
        k = 3 
        t = np.linspace(0, 1, len(df_relative_track)) # construct B-spline for lat and lon coordinates separately 
        spl_lat = BSpline(t, df_relative_track['lat'], k) 
        spl_lon = BSpline(t, df_relative_track['lon'], k) # evaluate B-spline at original points 
        df_relative_track['lat'] = spl_lat(t) 
        df_relative_track['lon'] = spl_lon(t)
    
    elif (smooth_step != None) and (smooth_step != 0):
        print(f'smooth: {smooth_step}')
        df_relative_track = df_relative_track.rolling(window=smooth_step, closed='both',min_periods=1).mean()
    
    if grid_conformity == True:
        for i, row in df_relative_track.iterrows():
            # select the nearest grid cell in the dataset to the (lat, lon) pair
            nearest_latlon = ds.sel(lat=row['lat'], lon=row['lon'], method='nearest')
            # store the closest lat, lon values in the dataframe
            df_relative_track.at[i, 'lat'] = nearest_latlon['lat'].values
            df_relative_track.at[i, 'lon'] = nearest_latlon['lon'].values

    return df_relative_track

def storm_tracker_mslp_ens(ds, df_ws, large_box, small_box, smooth_step = None):
    storm_tracker_list = []
    for members in ds.member:
        print(f'member: {members}')
        storm_tracker_list.append(storm_tracker_mslp(ds.sel(member = members), df_ws, smooth_step = smooth_step,  large_box = large_box, small_box = small_box))
    return pd.concat(storm_tracker_list)

def closest_point_to_ref(df_ws, dict_storm):
    # Calculate the distances between each lat-lon pair and the reference point using geopy
    global storm_name
    storm_df = pd.read_csv('D:/paper_3/data/list_cities_storm.csv', index_col=0)
    
    city_ref = storm_df.query('cities == @dict_storm[@storm_name]["target_city"]')
    distances = [haversine((city_ref['lat'].item(), city_ref['lon'].item()), (lat, lon)) for lat, lon in zip(df_ws['lat'], df_ws['lon'])]
    distances_arr = np.array(distances).reshape(-1, 1)
    # Calculate the Euclidean distances between each distance and the reference distance
    euc_dists = cdist(distances_arr, np.array([0]).reshape(-1, 1))
    # Find the index of the closest lat-lon pair
    closest_idx = np.argmin(euc_dists)
    return df_ws.iloc[[closest_idx]]

def landfall(track_syn, country = 'USA'):
    '''
    Find the index of the point in the storm track that is closest to the coastline.
    '''
    # Define a dictionary of countries and corresponding CRSs      
    country_crs = {
        'USA': 'EPSG:4326',  # replace XXXX with the appropriate EPSG code for USA
        'France': 'EPSG:2154',  # replace YYYY with the appropriate EPSG code for France
        'Germany': 'EPSG:25832'  # replace ZZZZ with the appropriate EPSG code for Germany
    }
    
    # Load coastline shapefile as a GeoDataFrame        
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    sel_country = world[world['iso_a3'] == country]

    if country in country_crs:
        sel_country = sel_country.to_crs(country_crs[country])
    else:
        raise ValueError(f'No CRS defined for {country}')
    
    # Convert storm track data to a GeoDataFrame
    geometry = [Point(lon, lat) for lat, lon in zip(track_syn['lat'], track_syn['lon'])]
    storm_gdf = gpd.GeoDataFrame(track_syn, geometry=geometry, crs='EPSG:4326')

    # Define a buffer zone around coastline
    buffer_distance = 0.5  # degrees

    # Find the index of the point in the storm track that is closest to the coastline
    landfall_idx = None
    list = []
    for i, point in storm_gdf.geometry.items():
        distances = sel_country.distance(point)
        list.append(distances.min())
        if distances.min() < buffer_distance:
            landfall_idx = i
            break

    # # PLot figure
    # fig = plt.figure(figsize=(6, 8))
    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=df_ws['lon'].mean(), central_latitude=df_ws['lat'].mean()))    # 
    # ax.coastlines(resolution='10m') 
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.plot(track_syn['lon'], track_syn['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
    # ax.plot(track_syn.loc[landfall_idx].lon,track_syn.loc[landfall_idx].lat, 'ro', markersize=10, transform=ccrs.PlateCarree(),  label = 'landfall')
    # ax.plot(track_syn.loc[landfall_idx - 3 * pd.offsets.Hour()].lon,track_syn.loc[landfall_idx - 3 * pd.offsets.Hour()].lat, 'bo', markersize=10, transform=ccrs.PlateCarree(),  label = 'landfall - 1 time step')
    # plt.show()
            
    return track_syn.loc[landfall_idx], sel_country


def storm_shifter(track_syn, track_hist, ds_syn = None):
    # find the point along the hystorical track that is closest to the city of interest or landfall point
    df_ref = closest_point_to_ref(track_hist, prep.dict_storm)
    landfall_point = landfall(track_syn, country = 'USA')

    # Calculate the translation in latitude and longitude between the historical and synthetic tracks at the reference point
    lat_translation = df_ref['lat'].item() - track_syn['lat'].loc[df_ref['time']].item()
    lon_translation = df_ref['lon'].item() - track_syn['lon'].loc[df_ref['time']].item()
    # Shift the synthetic track to match the historical track at the reference point
    track_syn_shifted = track_syn.copy()
    track_syn_shifted['lat'] += lat_translation
    track_syn_shifted['lon'] += lon_translation

    if ds_syn:
        # Shift the dataset by the translation
        ds_syn_shifted = ds_syn.copy()
        ds_syn_shifted['lat'] = ds_syn['lat'] + lat_translation
        ds_syn_shifted['lon'] = ds_syn['lon'] + lon_translation
        ds_syn_shifted['lat'].attrs = ds_syn['lat'].attrs
        ds_syn_shifted['lon'].attrs = ds_syn['lon'].attrs

        return track_syn_shifted, ds_syn_shifted
    
    else:
        return track_syn_shifted
    

def storm_precip_shifter(track_syn, ds_syn = None, save_ds = None):
    '''
    Shift the synthetic track and precipitation dataset to match the peak precipitation at landfall time to be on top of a city of reference.
    '''
    global storm_name
    dict_storm = prep.dict_storm
    storm_df = pd.read_csv('D:/paper_3/data/list_cities_storm.csv', index_col=0)
    city_ref = storm_df.query('cities == @dict_storm[@storm_name]["target_city"]')

    # find the point along the hystorical track that is closest to the city of interest or landfall point
    track_syn_landfall, sel_country = landfall(track_syn, country = 'USA')
    
    # Add a buffer of the minimum spatial resolution to the geometry
    buffer_distance = ds_syn.lon[1].item() - ds_syn.lon[0].item()
    sel_country_buffered = sel_country.buffer(buffer_distance)
    # mask the xarray dataset with the shape of the country
    ds_syn_base = ds_syn.copy()
    ds_syn_base.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    ds_syn_base.rio.write_crs("epsg:4326", inplace=True)
    ds_syn_clipped = ds_syn_base.rio.clip(sel_country_buffered.geometry.values)

    # define a buffer around the landfall point
    buffer_around_centre = 4
    # find the maximum precipitation value in the synthetic track at the time of landfall plus/minus 3 time steps 
    time_res = ds_syn_base.time[1] - ds_syn_base.time[0]
    time_slice = slice(track_syn_landfall.name - 3 * time_res.values, track_syn_landfall.name + 3 * time_res.values)
    rolling_sum = ds_syn_clipped['Ptot'].sel(time=time_slice,
                                   lat=slice(track_syn_landfall.lat + buffer_around_centre, track_syn_landfall.lat - buffer_around_centre),
                                   lon=slice(track_syn_landfall.lon - buffer_around_centre, track_syn_landfall.lon + buffer_around_centre)).rolling(time=3, min_periods=1).sum('time')
    max_value = rolling_sum.max().item()
    max_index = ds_syn_clipped['Ptot'].sel(time=time_slice).where(rolling_sum == max_value, drop=True)

    # rolling_sum.sel(lat = max_index['lat'].item(), lon = max_index['lon'].item(), time = max_index['time'] ).plot()
    # Calculate the translation in latitude and longitude between the city location and the maximum precipitation point
    lat_translation = city_ref['lat'].item() - max_index['lat'].item()
    lon_translation = city_ref['lon'].item() - max_index['lon'].item()

    # calculate the distance between the reference point and the landfall point
    shifting_distance = haversine((city_ref['lat'].item(), city_ref['lon'].item()), (max_index['lat'].item(), max_index['lon'].item())) # in km
    
    # Shift the synthetic track to match the historical track at the reference point
    track_syn_shifted = track_syn.copy()
    track_syn_shifted['lat'] += lat_translation
    track_syn_shifted['lon'] += lon_translation

    if ds_syn:
        ds_syn_shifted = ds_syn.copy()
        # Find the closest lat, lon values in the shifted dataset to the reference point
        nearest_latlon = ds_syn.sel(lat=ds_syn['lat'] + lat_translation, lon=ds_syn['lon'] + lon_translation, method='nearest')
        # store the closest lat, lon values in the dataframe
        # df_relative_track.at[i, 'lat'] = nearest_latlon['lat'].values
        # df_relative_track.at[i, 'lon'] = nearest_latlon['lon'].values

        # Shift the dataset by the translation
        ds_syn_shifted['lat'] = ds_syn['lat'] + lat_translation
        ds_syn_shifted['lon'] = ds_syn['lon'] + lon_translation
        ds_syn_shifted['lat'].attrs = ds_syn['lat'].attrs
        ds_syn_shifted['lon'].attrs = ds_syn['lon'].attrs
        # ds_syn_shifted = ds_syn_shifted.merge(ds_syn, compat="override")
        
        # Save the shifted dataset
        if save_ds:
            storm_year = ds_syn.time.dt.year[0].item()
            ds_syn_shifted.to_netcdf(rf'D:\paper_3\data\spectral_nudging_data\{storm_name}_shifted\BOT_t_HR255_Nd_SU_1015_{save_ds}_{storm_year}_{storm_name}_shifted_storm.nc')


        # PLOT SI FIGURE
        trak_syn_shifted_landfall, sel_country = landfall(track_syn_shifted, country = 'USA')
    
        rolling_sum_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice,
                                    lat=slice(trak_syn_shifted_landfall.lat + buffer_around_centre, trak_syn_shifted_landfall.lat - buffer_around_centre),
                                    lon=slice(trak_syn_shifted_landfall.lon - buffer_around_centre, trak_syn_shifted_landfall.lon + buffer_around_centre)).rolling(time=3, min_periods=1).sum('time')
        max_value_shifted = rolling_sum_shifted.max().item()
        max_index_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice).where(rolling_sum_shifted == max_value_shifted, drop=True)
        
        # plot normal and shifted storms in two panels side by side
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8), subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea(central_longitude=-77, central_latitude=30)})

        # Plot the first independent plot on the first subplot
        axs[0].add_feature(cfeature.LAND, facecolor='lightgray')
        rolling_sum.where(rolling_sum > 5).sel(time=max_index['time']).plot(ax=axs[0], transform=ccrs.PlateCarree(), x='lon', y='lat', vmin=5, cmap='Blues', add_colorbar=False)
        axs[0].plot(max_index['lon'].values, max_index['lat'].values, 'ro', markersize=5, alpha=1, transform=ccrs.PlateCarree(), label='Max precip.')
        axs[0].plot(track_syn['lon'], track_syn['lat'], transform=ccrs.PlateCarree(), linestyle='dashed', color='k', label='original track')
        axs[0].plot(city_ref['lon'].values, city_ref['lat'].values, 'r+',markeredgewidth=2, markersize=12, alpha=0.9, transform=ccrs.PlateCarree(), label='NYC')
        axs[0].set_extent([max_index['lon'].values-10, max_index['lon'].values+10, max_index['lat'].values-10, max_index['lat'].values+10])
        axs[0].legend(frameon=False, loc='upper left')
        axs[0].set_title('Original storm track')
            
        # Specify the locations at which the gridlines should be drawn
        gridlines = axs[0].gridlines(draw_labels=True, linewidth=0)
        gridlines.xlocator = mticker.FixedLocator(range(-180, 180, 10))  # Change the range and step size as needed
        gridlines.ylocator = mticker.FixedLocator(range(-90, 90, 10))  # Change the range and step size as needed
        # Specify whether or not to draw labels on the gridlines
        gridlines.xlabels_bottom = gridlines.ylabels_left = True
        gridlines.xlabels_top = gridlines.ylabels_right = False

        # Plot the second independent plot on the second subplot
        axs[1].add_feature(cfeature.LAND, facecolor='lightgray')
        cs = rolling_sum_shifted.where(rolling_sum_shifted > 5).sel(time=max_index_shifted['time']).plot(ax=axs[1], transform=ccrs.PlateCarree(), x='lon', y='lat', vmin=5, cmap='Blues', add_colorbar=False)
        axs[1].plot(max_index_shifted['lon'].values, max_index_shifted['lat'].values, 'ro', markersize=5, alpha=1, transform=ccrs.PlateCarree(), label='Max precip.')
        axs[1].plot(track_syn['lon'], track_syn['lat'], transform=ccrs.PlateCarree(), linestyle='dashed', color='k', alpha = 1, label='original track')
        axs[1].plot(track_syn_shifted['lon'], track_syn_shifted['lat'], transform=ccrs.PlateCarree(), linestyle='dashed', color='red', label='manipulated track')
        axs[1].plot(city_ref['lon'].values, city_ref['lat'].values, 'r+',markeredgewidth=2, markersize=12, alpha=0.9, transform=ccrs.PlateCarree(), label='NYC')
        axs[1].set_extent([max_index['lon'].values-10, max_index['lon'].values+10, max_index['lat'].values-10, max_index['lat'].values+10])
        axs[1].legend(frameon=False, loc='upper left')
        axs[1].set_title('Manipulated storm track')

        # Specify the locations at which the gridlines should be drawn
        gridlines = axs[1].gridlines(draw_labels=True, linewidth=0)
        gridlines.xlocator = mticker.FixedLocator(range(-180, 180, 10))  # Change the range and step size as needed
        gridlines.ylocator = mticker.FixedLocator(range(-90, 90, 10))  # Change the range and step size as needed
        # Specify whether or not to draw labels on the gridlines
        gridlines.xlabels_bottom = gridlines.ylabels_left = True
        gridlines.xlabels_top = gridlines.ylabels_right = False

        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust the values as needed
        cbar = plt.colorbar(cs, cax=cbar_ax, label='precipitation (mm)')

        # Save the figure
        plt.savefig(rf'D:\paper_3\data\Figures\paper_figures\si_two_subplots.png', dpi=300, bbox_inches='tight')

        # Show the figure
        plt.show()

        return track_syn_shifted, ds_syn_shifted, shifting_distance

    else:
        return track_syn_shifted, shifting_distance

    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-77, central_latitude=30))    # 
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # rolling_sum.where(rolling_sum > 5).sel(time = max_index['time'] ).plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', vmin = 5, cmap = 'Blues' ) # - 3 * pd.offsets.Hour()
    # ax.plot(max_index['lon'].values, max_index['lat'].values, 'ro', markersize=5, alpha = 0.5, transform=ccrs.PlateCarree(), label = 'Max precip.' )
    # ax.plot(track_syn['lon'], track_syn['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
    # ax.plot(city_ref['lon'].values, city_ref['lat'].values, 'ko', markersize=5, alpha = 0.9, transform=ccrs.PlateCarree(), label = 'NYC')
    # ax.set_extent([max_index['lon'].values-10, max_index['lon'].values+10, max_index['lat'].values-10, max_index['lat'].values+10])
    # plt.legend(frameon=False, loc='upper left')
    # plt.show()

    # trak_syn_shifted_landfall, sel_country = landfall(track_syn_shifted, country = 'USA')
    
    # rolling_sum_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice,
    #                                lat=slice(trak_syn_shifted_landfall.lat + buffer_around_centre, trak_syn_shifted_landfall.lat - buffer_around_centre),
    #                                lon=slice(trak_syn_shifted_landfall.lon - buffer_around_centre, trak_syn_shifted_landfall.lon + buffer_around_centre)).rolling(time=3, min_periods=1).sum('time')
    # max_value_shifted = rolling_sum_shifted.max().item()
    # max_index_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice).where(rolling_sum_shifted == max_value_shifted, drop=True)
    # # max_value_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice, 
    # #                                lat = slice(trak_syn_shifted_landfall.lat + buffer_around_centre, trak_syn_shifted_landfall.lat - buffer_around_centre), 
    # #                                lon = slice(trak_syn_shifted_landfall.lon - buffer_around_centre, trak_syn_shifted_landfall.lon + buffer_around_centre)).max()
    # # max_index_shifted = ds_syn_shifted['Ptot'].sel(time=time_slice).where(ds_syn_shifted['Ptot'].sel(time=time_slice) == max_value_shifted, drop=True)

    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=-77, central_latitude=30))    # 
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # rolling_sum_shifted.where(rolling_sum_shifted > 5).sel(time = max_index_shifted['time'] ).plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat',vmin = 5, cmap = 'Blues' ) # - 3 * pd.offsets.Hour()
    # ax.plot(max_index_shifted['lon'].values, max_index_shifted['lat'].values, 'ro', markersize=5, alpha = 0.5, transform=ccrs.PlateCarree(), label = 'Max precip.' )
    # ax.plot(track_syn_shifted['lon'], track_syn_shifted['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'shifted track')
    # ax.plot(city_ref['lon'].values, city_ref['lat'].values, 'ko', markersize=5, alpha = 0.9, transform=ccrs.PlateCarree(), label = 'NYC')
    # ax.set_extent([max_index_shifted['lon'].values-10, max_index_shifted['lon'].values+10, max_index_shifted['lat'].values-10, max_index_shifted['lat'].values+10])
    # plt.show()


    
    # # PLot figure
    # fig = plt.figure(figsize=(6, 8))
    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=df_ws['lon'].mean(), central_latitude=df_ws['lat'].mean()))    # 
    # ax.coastlines(resolution='10m') 
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ds_syn['Ptot'].sel(time = track_syn_landfall.name ).plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat') # - 3 * pd.offsets.Hour()
    # ax.plot(track_syn['lon'], track_syn['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
    # ax.plot(track_syn.loc[track_syn_landfall.name].lon,track_syn.loc[track_syn_landfall.name].lat, 'ro', markersize=10, transform=ccrs.PlateCarree(),  label = 'landfall')
    # ax.plot(track_syn.loc[track_syn_landfall.name - 3 * pd.offsets.Hour()].lon,track_syn.loc[track_syn_landfall.name - 3 * pd.offsets.Hour()].lat, 'bo', markersize=10, transform=ccrs.PlateCarree(),  label = 'landfall - 1 time step')
    # ax.plot(max_index['lon'].values, max_index['lat'].values, 'k*', markersize=10, transform=ccrs.PlateCarree(), label = 'max precip')
    # plt.legend()
    # plt.show()

def storm_shifter_ens(track_syn, track_hist, ds_syn = None):
    if not isinstance(track_syn['member'][0], str):
        track_syn['member'] = track_syn['member'].astype(int).astype(str) 
    storm_shifter_list = []
    ds_storm_shifter_list = []
    for members in ds_syn.member:
        print(f'member: {members}')
        track_syn_shifted, ds_syn_shifted = storm_shifter(track_syn.loc[track_syn['member'] == members], track_hist, ds_syn = ds_syn.sel(member = members))
        
        ds_storm_shifter_list.append(ds_syn_shifted)
        storm_shifter_list.append(track_syn_shifted)
            
    return pd.concat(storm_shifter_list), xr.concat(ds_storm_shifter_list, dim = 'member')

def storm_precip_shifter_ens(track_syn, ds_syn = None, save_ds = None):
    global storm_name
    if not isinstance(track_syn['member'][0], str):
        track_syn['member'] = track_syn['member'].astype(int).astype(str) 
    #     
    storm_shifter_list = []
    ds_storm_shifter_list = []
    shift_distance_list = []
    for members in ds_syn.member:
        print(f'member: {members.item()}')
        # Save the shifted dataset if it is to be saved
        if save_ds:
            save_ds_member = f'{save_ds}_{members.item()}'
        # Shift the synthetic track to match the historical track at the reference point
        track_syn_shifted, ds_syn_shifted, shifting_distance = storm_precip_shifter(track_syn.loc[track_syn['member'] == members], ds_syn = ds_syn.sel(member = members), save_ds = save_ds_member)
        
        if save_ds:
            shift_distance_list.append([save_ds_member, shifting_distance])
        else:
            shift_distance_list.append(shifting_distance)

        ds_storm_shifter_list.append(ds_syn_shifted)
        storm_shifter_list.append(track_syn_shifted)
    ds_precip_shift_ens = xr.concat(ds_storm_shifter_list, dim = 'member')    

    return pd.concat(storm_shifter_list), ds_precip_shift_ens, pd.DataFrame(shift_distance_list, columns=['scenario', 'shifting distance'])


def storm_composite(ds, df_center, radius = 3):
    ''' This function has as objective to create a storm composite.
    Composites are a reference frame centred on the tropical cyclone eye.
    at each time step, select data around the eye center and buffer around X degrees, translating all corrdinates to [0,0] at the eye.'''
    ds = ds.sel(time=slice(df_center.index[0], df_center.index[-1]))
    ds_relative_track = []
    for time_step in ds.time.values:
        # initial time step
        ds_ini = ds.sel(time = time_step, lat = df_center['lat'].loc[time_step], 
                        lon = df_center['lon'].loc[time_step], method = 'nearest')
        # Find radius around the centre
        ds_test = ds.sel(time = time_step, lat = slice(ds_ini['lat'] + radius, ds_ini['lat'] - radius), 
                        lon = slice(ds_ini['lon'] - radius, ds_ini['lon'] + radius))
        # Normalise it so centre becomes (0,0)
        ds_test['lat'] = (ds_test['lat'] - ds_ini['lat']).round(3)
        ds_test['lon'] = (ds_test['lon'] - ds_ini['lon']).round(3)
        # organise data
        ds_test = ds_test.sortby(ds_test.lat)
        ds_test = ds_test.sortby(ds_test.lon)
        ds_relative_track.append(ds_test)

    return xr.concat(ds_relative_track, dim = 'time')

def storm_composite_ens(ds, df_center = None, radius = 3):
    # Same as storm_composite but for data with multiple ensemble members (check if members are in string format)
    if not isinstance(df_center['member'][0], str):
        df_center['member'] = df_center['member'].astype(int).astype(str) 
    list_storm_composite = []
    for members in ds.member:
        list_storm_composite.append(storm_composite(ds.sel(member = members), df_center = df_center[['lat','lon','member']].loc[df_center['member']== members], radius = radius))
    return xr.concat(list_storm_composite, dim='member')

def scenarios_single_timeseries_plot(ds_factual_stat, var, storm_name, mode='mean', graph='normal', start_date=None, end_date=None):
    # Calculate for a chosen variable the aggregated metric for the factual scenario
    if graph == 'normal':
        df_factual = getattr(ds_factual_stat, mode)(['lat', 'lon'])[var].to_dataframe()[var].reset_index()
    elif graph == 'cumulative':
        df_factual = getattr(ds_factual_stat, mode)(['lat', 'lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(df_factual, x='time', y=var, label='factual')
    plt.legend(loc=2, borderaxespad=0., frameon=False)
    plt.title(f'{storm_name} {mode} {var} composite')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return fig


def scenarios_trio_timeseries_plot(ds_counter_stat, ds_factual_stat, ds_merra2_stat, var, storm_name, mode='mean', graph='normal', start_date=None, end_date=None):
    # Calculate for a chosen variable the aggregated metric for the factual scenario
    if graph == 'normal':
        df_counter = getattr(ds_counter_stat, mode)(['lat', 'lon'])[var].to_dataframe()[var].reset_index()
        df_factual = getattr(ds_factual_stat, mode)(['lat', 'lon'])[var].to_dataframe()[var].reset_index()
        ds_merra2 = getattr(ds_merra2_stat, mode)(['lat', 'lon'])[var].to_dataframe()[var].reset_index()
    elif graph == 'cumulative':
        df_counter = getattr(ds_counter_stat, mode)(['lat', 'lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
        df_factual = getattr(ds_factual_stat, mode)(['lat', 'lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
        ds_merra2 = getattr(ds_merra2_stat, mode)(['lat', 'lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(df_counter, x='time', y=var, label='echam6.1')
    sns.lineplot(df_factual, x='time', y=var, label='ERA5')
    sns.lineplot(ds_merra2, x='time', y=var, label='Merra2')
    if var == 'U10m':
        sns.lineplot(x = 'time', y = 'usa_wind', data = df_ws_wnd, linestyle = 'dashed', color = 'black', label = 'IBTrACS')
    elif var == "mslp":
        sns.lineplot(x = 'time', y = 'usa_pres', data = df_ws_wnd, linestyle = 'dashed', color = 'black', label = 'IBTrACS')
    plt.legend(loc=2, borderaxespad=0., frameon=False)
    plt.title(f'{storm_name} {mode} {var} composite')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return fig

def scenarios_timeseries_plot(ds_counter_stat, ds_factual_stat, ds_plus2_stat, var, storm_name, mode = 'mean', ensemble_mode = False, graph = 'normal', df_ws = None, start_date = None, end_date = None):
    # Calculate for a chosen variable the aggregated metric between each SN scenario
    if ensemble_mode == False:
        if graph == 'normal':
            df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index()
            df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index()
            df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index()
            ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values, 2)]
        elif graph == 'cumulative':
            df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
            df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
            df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index()
            ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values, 2)]

        text = f'factual/counter: {str(round(ratios[0],2))}, plus2/counter: {str(round(ratios[1],2))}, plus2/factual: {str(round(ratios[2],2))}'
        print(var + ' ' + text)

        fig, ax = plt.subplots(figsize=(6, 6))   
        if df_ws is not None:
            for index, row in df_ws.iloc[0:-1].iterrows():
                next_row = df_ws.loc[index[0],index[1]+1]
                if row['usa_sshs'] == 'Category 1':
                    plt.axvspan(row['time'], next_row['time'], facecolor='yellow', alpha=0.5)
                if row['usa_sshs'] == 'Category 2':
                    plt.axvspan(row['time'], next_row['time'], facecolor='orange', alpha=0.5)
                if row['usa_sshs'] == 'Category 3':
                    plt.axvspan(row['time'], next_row['time'], facecolor='red', alpha=0.5)
                if row['usa_sshs'] == 'Post-tropical':
                    plt.axvspan(row['time'], next_row['time'], facecolor='grey', alpha=0.5)

        sns.lineplot(df_counter,x = 'time', y = var, label = 'counter')
        sns.lineplot(df_factual,x = 'time', y = var, label = 'factual')
        sns.lineplot(df_plus2,x = 'time', y = var, label = 'plus2')
        plt.legend(loc=2, borderaxespad=0.,frameon=False)
        plt.title(f'{storm_name} {mode} {var} composite')    
        # plt.text(0.05, 0.04, text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    else:#
        if graph == 'normal':
            df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='counter')
            df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='factual')
            df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='plus2')
            df_runs = pd.concat([df_counter,df_factual,df_plus2])
            ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values, 2)]

        elif graph == 'cumulative':
            df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='counter')
            df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='factual')
            df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='plus2')
            df_runs = pd.concat([df_counter,df_factual,df_plus2])
            ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                      round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values, 2)]

        text = f'factual/counter: {str(round(ratios[0],2))}, plus2/counter: {str(round(ratios[1],2))}, plus2/factual: {str(round(ratios[2],2))}'
        print(var + ' ' + text)
        
        fig, ax = plt.subplots(figsize=(6, 6)) 
        if df_ws is not None:
            for index, row in df_ws.iloc[0:-1].iterrows():
                next_row = df_ws.loc[index[0],index[1]+1]
                if row['usa_sshs'] == 'Category 1':
                    plt.axvspan(row['time'], next_row['time'], facecolor='yellow', alpha=0.5)
                if row['usa_sshs'] == 'Category 2':
                    plt.axvspan(row['time'], next_row['time'], facecolor='orange', alpha=0.5)
                if row['usa_sshs'] == 'Category 3':
                    plt.axvspan(row['time'], next_row['time'], facecolor='red', alpha=0.5)
                if row['usa_sshs'] == 'Post-tropical':
                    plt.axvspan(row['time'], next_row['time'], facecolor='grey', alpha=0.5)
                        
        sns.lineplot(df_runs, x = 'time', y = var, style = 'member', hue = 'scenario')
        plt.legend(loc=2, borderaxespad=0.,frameon=False)
        plt.title(f'{storm_name} {mode} {var} composite')
        # plt.text(0.005, 0.04, text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
        # add a note to the graph for the time 2012-10-29T18:00:00.000000000 writing 'NYC landfall'
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xlim(ax.get_xlim())
        datetime_value = datetime.strptime("2012-10-29T18:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
        ax2.set_xticks([datetime_value])
        ax2.set_xticklabels(["NY"])
        # plt.text("2012-10-29T18:00:00.000000000", 0.04, 'NYC landfall', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    return fig

def scenarios_timeseries_four_plot(ds_hist_stat, ds_counter_stat, ds_factual_stat, ds_plus2_stat, var, storm_name, mode = 'mean', hist_name = 'ECHAM 6.1', graph = 'normal', start_date = None, end_date = None):
    # Calculate for a chosen variable the aggregated metric between each SN scenario

    if graph == 'normal':
        df_hist = getattr(ds_hist_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario=hist_name).assign(member=hist_name)
        df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='counter')
        df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='factual')
        df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='plus2')
        df_runs = pd.concat([df_hist, df_counter,df_factual,df_plus2])
        ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values, 2)]

    elif graph == 'cumulative':
        df_hist = getattr(ds_hist_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario=hist_name).assign(member=hist_name)
        df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='counter')
        df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='factual')
        df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].cumsum(dim='time').to_dataframe()[var].reset_index().assign(scenario='plus2')
        df_runs = pd.concat([df_hist, df_counter,df_factual,df_plus2])
        ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].sum().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].sum().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].sum().values, 2)]

    text = f'factual/counter: {str(round(ratios[0],2))}, plus2/counter: {str(round(ratios[1],2))}, plus2/factual: {str(round(ratios[2],2))}'
    print(var + ' ' + text)
    fig, ax = plt.subplots(figsize=(6, 6))        
    sns.lineplot(df_runs, x = 'time', y = var, style = 'member', hue = 'scenario')
    plt.legend(loc=2, borderaxespad=0.,frameon=False)
    plt.title(f'{storm_name} {mode} {var} composite')
    plt.text(0.005, 0.04, text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    return fig

def fig_composites_collection(mslp, wind, ptot, storm_name, mode):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6))
    ax1.imshow(mslp.canvas.renderer.buffer_rgba())
    ax2.imshow(wind.canvas.renderer.buffer_rgba())
    ax3.imshow(ptot.canvas.renderer.buffer_rgba())
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.savefig(f'D:/paper_3/data/Figures/storm_figures/composite_{mode}_{storm_name}.png', dpi=300, bbox_inches='tight')

def map_dif_plot(ds_plus, ds_fac, var):
    # Plots maps of the relative TC variable composite around the TC centre and calculate the ratios between different scenarios
    (ds_plus[var].mean(['time']) - ds_fac[var].mean(['time']) ).plot(robust = True)
    plt.title(f'{var}: plus2 - counter entire life of TC')
    plt.show()
    plt.close()
    print(f'total {var} increase across grid-space between the two scenarios is: {ds_plus[var].mean().values/ds_fac[var].mean().values}')

def track_storm(storm_name, ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, 
                df_ws, large_box, small_box, smoothing_step, radius, shift_storm = True):
    # Storm tracker for mean and ensemble members
    step = smoothing_step

    # Storm tracker for mean
    storm_track_factual_mean = storm_tracker_mslp(ds_factual_mean, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)
    storm_track_counter_mean = storm_tracker_mslp(ds_counter_mean, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)
    storm_track_plus2_mean = storm_tracker_mslp(ds_plus2_mean, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)
    
    # Storm tracker for ensemble members
    storm_track_factual_ens = storm_tracker_mslp_ens(ds_factual_ens, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)
    storm_track_counter_ens = storm_tracker_mslp_ens(ds_counter_ens, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)
    storm_track_plus2_ens = storm_tracker_mslp_ens(ds_plus2_ens, df_ws, smooth_step = step, large_box=large_box, small_box=small_box)

    # Plot minimum track following a proper track algorithm considering different components
    fig_mean_track = plot_minimum_track(storm_track_counter_mean, storm_track_factual_mean, storm_track_plus2_mean, df_ws, storm_name)
    fig_ens_track = plot_minimum_track(storm_track_counter_ens, storm_track_factual_ens, storm_track_plus2_ens, df_ws, storm_name)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 8))
    ax1.imshow(fig_mean_track.canvas.renderer.buffer_rgba())
    ax2.imshow(fig_ens_track.canvas.renderer.buffer_rgba())
    ax1.set_axis_off()
    ax2.set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.savefig(f'D:/paper_3/data/Figures/storm_figures/track_{storm_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate the distance between the two tracks
    if shift_storm:
    # Load data for copying the new shifted storms
        dict_storm_full = {'sandy':{'storm_date_start':'2012-10-01','storm_date_end':'2012-11-04'}}
        ds_factual_ens_full, ds_counter_ens_full, ds_plus2_ens_full = prep.load_multiple_var(['Ptot_mmh', 'mslp_hPa', 'u10m_m_s', 'v10m_m_s'],
                                                                            dict_storm_full[storm_name]['storm_date_start'].split('-')[0],
                                                                            f'D:/paper_3/data/spectral_nudging_data/{storm_name}/',
                                                                            dict_storm_full[storm_name]['storm_date_start'],
                                                                            dict_storm_full[storm_name]['storm_date_end'],
                                                                            name = storm_name,
                                                                            calc_U10m=False)

        storm_track_counter_ens_shift, ds_counter_ens_shift, df_shifting_distance_counter_ens = storm_precip_shifter_ens(storm_track_counter_ens, ds_counter_ens_full, save_ds = 'counter')
        storm_track_factual_ens_shift, ds_factual_ens_shift, df_shifting_distance_factual_ens = storm_precip_shifter_ens(storm_track_factual_ens, ds_factual_ens_full, save_ds = 'factual')
        storm_track_plus2_ens_shift, ds_plus2_ens_shift, df_shifting_distance_plus2_ens = storm_precip_shifter_ens(storm_track_plus2_ens, ds_plus2_ens_full, save_ds = 'plus2')

        # concatenate the dataframes of shifting distances and save to csv
        df_shifting_distance = pd.concat([df_shifting_distance_counter_ens, df_shifting_distance_factual_ens, df_shifting_distance_plus2_ens], axis=0)
        df_shifting_distance.to_csv(f'D:/paper_3/data/shifting_distance_{storm_name}.csv')

        plot_minimum_track(storm_track_counter_ens_shift, storm_track_factual_ens_shift, storm_track_plus2_ens_shift, df_ws, storm_name)

    # storm_track_factual_ens_shift_real = storm_tracker_mslp_ens(ds_counter_ens_shift, df_ws, smooth_step = step)
    # storm_track_counter_ens_shift_real = storm_tracker_mslp_ens(ds_factual_ens_shift, df_ws, smooth_step = step)
    # storm_track_plus2_ens_shift_real = storm_tracker_mslp_ens(ds_plus2_ens_shift, df_ws, smooth_step = step)

    # Define composites for each GW level
    ds_rel_counter = storm_composite(ds_counter_mean, df_center = storm_track_counter_mean, radius = radius)
    ds_rel_factual = storm_composite(ds_factual_mean, df_center = storm_track_factual_mean, radius = radius)
    ds_rel_plus2 = storm_composite(ds_plus2_mean, df_center = storm_track_plus2_mean, radius = radius)

    ds_rel_counter_ens = storm_composite_ens(ds_counter_ens, df_center = storm_track_counter_ens, radius = radius)
    ds_rel_factual_ens = storm_composite_ens(ds_factual_ens, df_center = storm_track_factual_ens, radius = radius)
    ds_rel_plus2_ens = storm_composite_ens(ds_plus2_ens, df_center = storm_track_plus2_ens, radius = radius)

    # Plot variables of each storm composite
    # MAX U10m
    fig_max_wind_mean = scenarios_timeseries_plot(ds_rel_counter, ds_rel_factual, ds_rel_plus2, 'U10m', storm_name, mode = 'max', graph = 'normal', df_ws = df_ws)
    fig_max_wind_ens = scenarios_timeseries_plot(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'U10m', storm_name, mode = 'max',ensemble_mode = True, graph = 'normal', df_ws = df_ws)
    # Max Ptot
    fig_sum_ptot_mean = scenarios_timeseries_plot(ds_rel_counter, ds_rel_factual, ds_rel_plus2, 'Ptot', storm_name, mode = 'mean', graph = 'normal', df_ws = df_ws)
    fig_sum_ptot_ens = scenarios_timeseries_plot(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', storm_name, mode = 'mean',ensemble_mode = True, graph = 'normal', df_ws = df_ws)
    # Min MSLP
    fig_min_mslp_mean = scenarios_timeseries_plot(ds_rel_counter, ds_rel_factual, ds_rel_plus2, 'mslp', storm_name, mode = 'min', graph = 'normal', df_ws = df_ws)
    fig_min_mslp_ens = scenarios_timeseries_plot(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'mslp', storm_name, mode = 'min',ensemble_mode = True, graph = 'normal', df_ws = df_ws)

    # Plot composites of each storm
    fig_composites_collection(fig_min_mslp_ens, fig_max_wind_ens, fig_sum_ptot_ens, storm_name, mode = 'ens')
    fig_composites_collection(fig_min_mslp_mean, fig_max_wind_mean, fig_sum_ptot_mean, storm_name, mode = 'mean')
    
    # Get ratios between peaks
    def get_ratios(ds_counter_stat, ds_factual_stat, ds_plus2_stat, var, mode = 'max', start_date = None, end_date = None):
        if (start_date is not None) and (end_date is not None):
            ds_counter_stat = ds_counter_stat.sel(time = slice(start_date, end_date))
            ds_factual_stat = ds_factual_stat.sel(time = slice(start_date, end_date))
            ds_plus2_stat = ds_plus2_stat.sel(time = slice(start_date, end_date))
        print(ds_counter_stat.time.values[0])
        print(ds_counter_stat.time.values[-1])
        df_counter = getattr(ds_counter_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='counter')
        df_factual = getattr(ds_factual_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='factual')
        df_plus2 = getattr(ds_plus2_stat,mode)(['lat','lon'])[var].to_dataframe()[var].reset_index().assign(scenario='plus2')
        df_runs = pd.concat([df_counter,df_factual,df_plus2])
        ratios = [round(getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_counter_stat,mode)(['lat','lon'])[var].mean().values, 2),
                    round(getattr(ds_plus2_stat,mode)(['lat','lon'])[var].mean().values/getattr(ds_factual_stat,mode)(['lat','lon'])[var].mean().values, 2)]
        text = f'factual/counter: {str(round(ratios[0],2))}, plus2/counter: {str(round(ratios[1],2))}, plus2/factual: {str(round(ratios[2],2))}'
        print(var + ' ' + text)
        return df_runs, ratios
    
    # Get ratios for each storm per gw level
    df_runs_wind_peak, ratios_max_wind_peak = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'U10m', mode = 'max', start_date='2012-10-29T12:00:00.000000000', end_date='2012-10-30T00:00:00.000000000')
    df_runs_mslp_peak , ratios_min_mslp_peak = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'mslp', mode = 'min', start_date='2012-10-29T12:00:00.000000000', end_date='2012-10-30T00:00:00.000000000')
    df_runs_ptot_peak, ratios_sum_ptot_peak = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', mode = 'mean', start_date='2012-10-24T12:00:00.000000000', end_date='2012-10-26T0:00:00.000000000')
    df_runs_ptot_nyc, ratios_sum_ptot_nyc = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', mode = 'mean', start_date='2012-10-29T12:00:00.000000000', end_date='2012-10-30T3:00:00.000000000')

    # plot df_runs_ptot_nyc having the x axis as time, y axis as Ptot, and the hue as scenario
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.lineplot(df_runs_ptot_nyc, x = 'time', y = 'Ptot', lw=2, style = 'member', hue = 'scenario')
    plt.legend(loc=2, borderaxespad=0.,frameon=False)
    plt.show()


    # how to calculate the standard deviation of each scenario for df_runs_ptot_nyc
    df_runs_ptot_peak.groupby(['scenario']).std()['Ptot']
    df_runs_ptot_peak.groupby(['scenario']).mean()['Ptot']

    df_runs_ptot_nyc.groupby(['scenario']).mean().loc['plus2']['Ptot'] - df_runs_ptot_nyc.groupby(['scenario']).mean().loc['counter']['Ptot'] > df_runs_ptot_nyc.groupby(['scenario']).std().loc['plus2']['Ptot']

    # now make the line of code above a function
    def check_significance(df_runs, scenario1, scenario2, var):
        dif_scen = round((df_runs.groupby(['scenario']).mean().loc[scenario1][var] - df_runs.groupby(['scenario']).mean().loc[scenario2][var]),4) 
        ratio_scen = round((df_runs.groupby(['scenario']).mean().loc[scenario1][var] / df_runs.groupby(['scenario']).mean().loc[scenario2][var]),4) 
        std_scens = round((df_runs.groupby(['scenario']).std().loc[scenario1][var] + df_runs.groupby(['scenario']).std().loc[scenario2][var])/2,4)

        if scenario1 == 'plus2' and scenario2 == 'counter':
            temp_dif = 15.5 - 13.6
        elif scenario1 == 'plus2' and scenario2 == 'factual':
            temp_dif = 15.5 - 14.28
        elif scenario1 == 'factual' and scenario2 == 'counter':
            temp_dif = 14.28 - 13.6

        print (f'The ratio between {scenario1} and {scenario2} per global warming level is {round((ratio_scen/temp_dif),4)}')

        if dif_scen > std_scens:
            print (f'The difference between {scenario1} and {scenario2} ({round(dif_scen,2)}) is larger than the standard deviation of the two scenarios ({std_scens})')
        else:
            print (f'The difference between {scenario1} and {scenario2} ({round(dif_scen,2)}) is smaller than the standard deviation of the two scenarios ({std_scens})')

    check_significance(df_runs_ptot_peak, 'plus2', 'counter', 'Ptot')
    check_significance(df_runs_ptot_peak, 'plus2', 'factual', 'Ptot')
    check_significance(df_runs_ptot_peak, 'factual', 'counter', 'Ptot')

    check_significance(df_runs_ptot_nyc, 'plus2', 'counter', 'Ptot')
    check_significance(df_runs_ptot_nyc, 'plus2', 'factual', 'Ptot')
    check_significance(df_runs_ptot_nyc, 'factual', 'counter', 'Ptot')

    df_runs_wind, ratios_max_wind = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'U10m', mode = 'max')
    df_runs_mslp , ratios_min_mslp = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'mslp', mode = 'min')
    df_runs_ptot, ratios_sum_ptot = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', mode = 'mean')

    # save as csv to "D:\paper_3\data\storm_composite"
    df_runs_wind.to_csv(f'D:/paper_3/data/storm_composite/{storm_name}_wind.csv')
    df_runs_mslp.to_csv(f'D:/paper_3/data/storm_composite/{storm_name}_mslp.csv')
    df_runs_ptot.to_csv(f'D:/paper_3/data/storm_composite/{storm_name}_ptot.csv')

    df_ws['usa_wind'] = df_ws['usa_wind'] * 0.514


    # Generate first figure paper
    singlecol = 8.3 * 0.393701
    doublecol = 14 * 0.393701
    fontsize=9
    dict_scenarios = {'counter':'PI', 'factual':'PD', 'plus2':'2C'}
    dict_colors = {'counter':'#4e79a7', 'factual':'#f28e2b', 'plus2':'#67000d'}
    datetime_value = datetime.strptime("2012-10-29T18:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
    datetime_value_ini = datetime.strptime("2012-10-29T12:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
    datetime_value_end = datetime.strptime("2012-10-30T03:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
    import matplotlib.ticker as mticker
    # Figure
    fig = plt.figure(figsize=(9, 6))
    plt.rcParams.update({'font.size': 9})

    ax2 = plt.subplot(332)
    ax3 = plt.subplot(335, sharex=ax2)
    ax4 = plt.subplot(338, sharex=ax2)
    # panel 1
    central_lon = storm_track_counter_ens.lon.min() + (storm_track_counter_ens.lon.max() - storm_track_counter_ens.lon.min())/2
    central_lat = storm_track_counter_ens.lat.min() + (storm_track_counter_ens.lat.max() - storm_track_counter_ens.lat.min())/2
    ax1 = fig.add_subplot(131, projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat) )    # 
    # ax1.coastlines(resolution='50m') 
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashdot', color = 'dimgray', label = 'IBTrACS')
    sns.lineplot(ax=ax1, data = storm_track_counter_ens, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['counter'], sort=False, style = 'member', label = 'counter') 
    sns.lineplot(ax=ax1, data = storm_track_factual_ens, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['factual'], sort=False, style = 'member',label = 'factual') 
    sns.lineplot(ax=ax1, data = storm_track_plus2_ens, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['plus2'], sort=False, style = 'member',label = 'plus2') 
    # add extent of plot
    ax1.set_extent([storm_track_counter_ens.lon.min()-5, storm_track_counter_ens.lon.max()+4, storm_track_counter_ens.lat.min()-5, storm_track_counter_ens.lat.max()+5], crs=ccrs.PlateCarree())
    ax1.set_title(f'a)', loc='left',pad=10)
    ax1.get_legend().remove()
        
    # Specify the locations at which the gridlines should be drawn
    gridlines = ax1.gridlines(draw_labels=True, linewidth=0)
    gridlines.xlocator = mticker.FixedLocator(range(-180, 180, 10))  # Change the range and step size as needed
    gridlines.ylocator = mticker.FixedLocator(range(-90, 90, 10))  # Change the range and step size as needed
    # Specify whether or not to draw labels on the gridlines
    gridlines.xlabels_top = gridlines.ylabels_left = True
    gridlines.xlabels_bottom = gridlines.ylabels_right = False
    
    # panel 2
    var = 'mslp'      
    sns.lineplot(df_runs_mslp, ax=ax2, x = 'time', y = var, style = 'member', hue = 'scenario',palette=dict_colors)
    sns.lineplot(x = 'time', y = 'usa_pres', data = df_ws, linestyle = 'dashdot', color = 'dimgray', label = 'IBTrACS', ax=ax2)
    ax2.axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray')
    ax2.set_title(f'b)', loc='left',pad=0)
    ax2.get_legend().remove()
    ax2.set_ylabel('MSLP [hPa]')

    # panel 3
    var = 'U10m' 
    sns.lineplot(df_runs_wind, ax=ax3, x = 'time', y = var, style = 'member', hue = 'scenario', palette=dict_colors)
    sns.lineplot(x = 'time', y = 'usa_wind', data = df_ws, linestyle = 'dashdot', color = 'dimgray', label = 'IBTrACS', ax=ax3)
    ax3.axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray')
    ax3.set_title(f'c)', loc='left', pad=0)
    ax3.get_legend().remove()
    ax3.set_ylabel('wind speed [m/s]')

    # panel 4
    var = 'Ptot' 
    sns.lineplot(df_runs_ptot, ax=ax4, x = 'time', y = var, style = 'member', hue = 'scenario', palette=dict_colors)
    ax4.axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray')
    ax4.set_title(f'd)', loc='left', pad=0)
    ax4.get_legend().remove()
    ax4.set_ylabel('precipitation [mm/h]')
    ax4.set_xlabel('Date (Oct 2012)')

    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter("%d") # TODO: try only days
    ax2.xaxis.set_major_formatter(date_format)
    ax3.xaxis.set_major_formatter(date_format)
    ax4.xaxis.set_major_formatter(date_format)
    ax4.set_xticks(ax4.get_xticks()[::2])
    
    ax2.spines['right'].set_visible(True)
    ax3.spines['right'].set_visible(True)
    ax4.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(True)
    ax3.spines['top'].set_visible(True)
    ax4.spines['top'].set_visible(True)
    
    fig.subplots_adjust(wspace=0.36, hspace=0.3)

    # add NY in every subplot
    ax5 = ax2.secondary_xaxis("top")
    ax5.set_xlim(ax2.get_xlim())
    ax5.set_xticks([datetime_value_ini, datetime_value_end])
    ax5.set_xticklabels(["    NYC", " "], fontsize = 6)
    ax5.tick_params(axis='x', direction='in')

    ax6 = ax3.secondary_xaxis("top")
    ax6.set_xlim(ax3.get_xlim())
    ax6.set_xticks([datetime_value_ini, datetime_value_end])
    ax6.set_xticklabels(["    NYC", " "], fontsize = 6)
    ax6.tick_params(axis='x', direction='in')

    ax7 = ax4.secondary_xaxis("top")
    ax7.set_xlim(ax4.get_xlim())
    ax7.set_xticks([datetime_value_ini, datetime_value_end])
    ax7.set_xticklabels(["    NYC", " "], fontsize = 6)
    ax7.tick_params(axis='x', direction='in')

    handles, labels = ax1.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    sorted_unique = sorted(unique, key=lambda x: x[1])
    sorted_unique_updated = [(h, dict_scenarios[l]) if l in dict_scenarios else (h, l) for h, l in sorted_unique]
    sorted_unique_updated = sorted_unique_updated[3:7] + sorted_unique_updated[0:3]
    # ax.legend(*zip(*unique),loc='',bbox_to_anchor=bbox_to_anchor, frameon = False)
    plt.legend(*zip(*sorted_unique_updated), loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, frameon = False)

    fig.autofmt_xdate(rotation=0, ha='center')
    plt.tight_layout()

    fig.savefig(rf'd:\paper_3\data\Figures\paper_figures\paper_fig1_track_composite_ens_sandy.png', dpi=300, bbox_inches='tight')
    plt.show()

    # SI1 figure
    
    fig = plt.figure(figsize=(doublecol, 8))
    plt.rcParams.update({'font.size': 9})

    ny_lat, ny_lon = 40.7128, -74.0060
    central_lon = storm_track_counter_ens.lon.min() + (storm_track_counter_ens.lon.max() - storm_track_counter_ens.lon.min())/2
    central_lat = storm_track_counter_ens.lat.min() + (storm_track_counter_ens.lat.max() - storm_track_counter_ens.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    # 
    # ax1.coastlines(resolution='50m') 
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(ny_lon, ny_lat, 'ro', markersize=6, transform=ccrs.PlateCarree(), label='New York')
    # ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashdot', color = 'dimgray', label = 'IBTrACS')
    sns.lineplot(ax=ax, data = storm_track_counter_ens_shift, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['counter'], sort=False, style = 'member', label = 'counter') 
    sns.lineplot(ax=ax, data = storm_track_factual_ens_shift, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['factual'], sort=False, style = 'member',label = 'factual') 
    sns.lineplot(ax=ax, data = storm_track_plus2_ens_shift, x = 'lon', y = 'lat', transform=ccrs.PlateCarree(), units = "member", estimator = None, color = dict_colors['plus2'], sort=False, style = 'member',label = 'plus2') 
    # add extent of plot
    ax.set_extent([storm_track_counter_ens.lon.min()-5, storm_track_counter_ens.lon.max()+4, storm_track_counter_ens.lat.min()-5, storm_track_counter_ens.lat.max()+5], crs=ccrs.PlateCarree())
    
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    sorted_unique = sorted(unique, key=lambda x: x[1])
    sorted_unique_updated = [(h, dict_scenarios[l]) if l in dict_scenarios else (h, l) for h, l in sorted_unique]
    # ax.legend(*zip(*unique),loc='',bbox_to_anchor=bbox_to_anchor, frameon = False)
    plt.legend(*zip(*sorted_unique_updated), loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon = False)
     
    # Specify the locations at which the gridlines should be drawn
    gridlines = ax.gridlines(draw_labels=True, linewidth=0)
    gridlines.xlocator = mticker.FixedLocator(range(-180, 180, 10))  # Change the range and step size as needed
    gridlines.ylocator = mticker.FixedLocator(range(-90, 90, 10))  # Change the range and step size as needed
    # Specify whether or not to draw labels on the gridlines
    gridlines.xlabels_top = gridlines.ylabels_left = True
    gridlines.xlabels_bottom = gridlines.ylabels_right = False

    fig.savefig(rf'd:\paper_3\data\Figures\paper_figures\paper_fig_si1_track_sandy.png', dpi=300, bbox_inches='tight')

    plt.show()

    # # plot a map of ds_factual_ens_shift['Ptot'] at landfall time for member 1:
    # fig = plt.figure(figsize=(doublecol, 8))
    # plt.rcParams.update({'font.size': 9})
    # ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon, central_latitude=central_lat))    #
    # ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # ax.plot(ny_lon, ny_lat, 'ro', markersize=6, transform=ccrs.PlateCarree(), label='New York')
    # ax.set_extent([storm_track_counter_ens.lon.min()-5, storm_track_counter_ens.lon.max()+4, storm_track_counter_ens.lat.min()-5, storm_track_counter_ens.lat.max()+5], crs=ccrs.PlateCarree())
    # # ds_plus2_ens['Ptot'].sel(member="1", time="2012-10-29T18:00:00.000000000").plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', alpha=0.5, vmin = 1)
    # test['Ptot'].sel(time="2012-10-29T21:00:00.000000000").plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', alpha=0.5, vmin = 5)

    # # open the file with xarray "BOT_t_HR255_Nd_SU_1015_plus2_1_2012_sandy_shifted_storm" at D:\paper_3\data\spectral_nudging_data\sandy_shifted:
    # test = xr.open_dataset(r'D:\paper_3\data\spectral_nudging_data\sandy_shifted\BOT_t_HR255_Nd_SU_1015_plus2_1_2012_sandy_shifted_storm.nc')
    # # make test to have values <3 == nan
    # test['Ptot'] = test['Ptot'].where(test['Ptot'] > 3)



################################################################################
# MAIN SCRIPT
################################################################################
# LOAD DATA from previous script (sn_storm_preprocessing.py)
if __name__ == "__main__":
    storm_name = 'sandy' # sandy xaver xynthia
    
    # Define box size for storm tracking
    if (storm_name == 'xynthia') or (storm_name == 'xaver'):
        large_box = 4
        small_box = 2
    else:
        large_box = 7
        small_box = 4

    ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws = prep.preprocess_data(storm_name = storm_name, clip = True, plots = False, dict_storm = prep.dict_storm)

    track_storm(storm_name, ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, 
                df_ws, large_box=large_box, small_box=small_box, smoothing_step = 'savgol', radius = 10, shift_storm = True)


############################################################
    # plot historical runs?
    historical_runs = True
    if historical_runs == True:
        ###### Historical run - sandy
        # load and preprocess historical data
        ds_hist_merge_echam, df_ws_hist = prep.process_historical_data(storm_name = 'sandy', data = 'echam', clip = True, plots = False, dict_storm = prep.dict_storm, common_time = True)
        ds_hist_era5, df_ws_hist = prep.process_historical_data(storm_name = 'sandy', data = 'era5', clip = True, plots = False, dict_storm = prep.dict_storm, common_time = True)
        ds_hist_merra2, df_ws_hist = prep.process_historical_data(storm_name = 'sandy', data = 'merra2', clip = True, plots = False, dict_storm = prep.dict_storm, common_time = True)
        # Track historical storm
        storm_track_hist = storm_tracker_mslp(ds_hist_merge_echam, df_ws = df_ws_hist, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        storm_track_hist_era5 = storm_tracker_mslp(ds_hist_era5, df_ws = df_ws_hist, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        storm_track_hist_merra2 = storm_tracker_mslp(ds_hist_merra2, df_ws = df_ws_hist, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        # Plot minimum track
        plot_minimum_track_single(storm_track_hist, df_ws_hist, storm_name)
        plot_minimum_track_single(storm_track_hist_era5, df_ws_hist, storm_name)
        plot_minimum_track_single(storm_track_hist_merra2, df_ws_hist, storm_name)

        # Plot two tracks together
        fig = plt.figure(figsize=(6, 8))
        central_lon = ds_hist_merge_echam.lon.min() + (ds_hist_merge_echam.lon.max() - ds_hist_merge_echam.lon.min())/2
        central_lat = ds_hist_merge_echam.lat.min() + (ds_hist_merge_echam.lat.max() - ds_hist_merge_echam.lat.min())/2
        ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon.item(), central_latitude=central_lat.item()))    # 
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.plot(df_ws_hist['lon'], df_ws_hist['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')
        ax.plot(storm_track_hist['lon'], storm_track_hist['lat'],transform=ccrs.PlateCarree(), color = 'red', label = 'ECHAM6')
        ax.plot(storm_track_hist_era5['lon'], storm_track_hist_era5['lat'],transform=ccrs.PlateCarree(), color = 'blue', label = 'ERA5')
        ax.plot(storm_track_hist_merra2['lon'], storm_track_hist_merra2['lat'],transform=ccrs.PlateCarree(), color = 'green', label = 'Merra2')
        plt.title(f'{storm_name} track for mean GW levels')
        legend_without_duplicate_labels(ax,loc = 'center left')
        plt.tight_layout()
        fig.savefig(f'D:/paper_3/data/Figures/storm_figures/track_{storm_name}_echam6_era5.png', dpi=300, bbox_inches='tight')

        plt.show()

        # Plot storm composites
        ds_rel_hist_echam = storm_composite(ds_hist_merge_echam, df_center = storm_track_hist, radius = 10)
        ds_rel_hist_era5 = storm_composite(ds_hist_era5, df_center = storm_track_hist, radius = 10)
        ds_rel_hist_merra2 = storm_composite(ds_hist_merra2, df_center = storm_track_hist, radius = 10)

        # plot ds_hist_merge_echam using minimum mslp across lat lon and over time of storm (start and end)
        df_ws_wnd = df_ws.copy()
        # change wind speed from feet to m
        df_ws_wnd['usa_wind'] = df_ws_wnd['usa_wind'] * 0.514

        ### Compare ECHAM and ERA5
        # Plot composites of each storm
        fig_max_wind_echam_era5 = scenarios_trio_timeseries_plot(ds_rel_hist_echam, ds_rel_hist_era5, ds_rel_hist_merra2, 'U10m', storm_name, mode = 'max', graph = 'normal')
        fig_sum_ptot_echam_era5 = scenarios_trio_timeseries_plot(ds_rel_hist_echam, ds_rel_hist_era5, ds_rel_hist_merra2,'Ptot', storm_name, mode = 'mean', graph = 'normal')
        fig_min_mslp_echam_era5 = scenarios_trio_timeseries_plot(ds_rel_hist_echam, ds_rel_hist_era5, ds_rel_hist_merra2,'mslp', storm_name, mode = 'min', graph = 'normal')
        fig_composites_collection(fig_min_mslp_echam_era5, fig_max_wind_echam_era5, fig_sum_ptot_echam_era5, storm_name, mode = 'hist_echam_era5')

        fig_max_wind_echam_era5_abs = scenarios_trio_timeseries_plot(ds_hist_merge_echam, ds_hist_era5, ds_hist_merra2, 'U10m', storm_name, mode = 'max', graph = 'normal')
        fig_sum_ptot_echam_era5_abs = scenarios_trio_timeseries_plot(ds_hist_merge_echam, ds_hist_era5, ds_hist_merra2, 'Ptot', storm_name, mode = 'mean', graph = 'normal')
        fig_min_mslp_echam_era5_abs = scenarios_trio_timeseries_plot(ds_hist_merge_echam, ds_hist_era5, ds_hist_merra2, 'mslp', storm_name, mode = 'min', graph = 'normal')

        fig = plt.figure(figsize=(14, 8))
        sns.lineplot(x = 'time', y = 'U10m', data = ds_hist_merge_echam.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ECHAM6')
        sns.lineplot(x = 'time', y = 'U10m', data = ds_hist_era5.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ERA5')
        sns.lineplot(x = 'time', y = 'U10m', data = ds_hist_merra2.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'Merra2')
        sns.lineplot(x = 'time', y = 'usa_wind', data = df_ws_wnd, linestyle = 'dashed-dotted', color = 'black', label = 'IBTrACS')
        plt.legend()
        plt.title(f'Maximum U10m in North Atlantic over {storm_name} time')
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(14, 8))
        sns.lineplot(x = 'time', y = 'mslp', data = ds_hist_merge_echam.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ECHAM6')
        sns.lineplot(x = 'time', y = 'mslp', data = ds_hist_era5.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ERA5')
        sns.lineplot(x = 'time', y = 'mslp', data = ds_hist_merra2.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'Merra2')
        sns.lineplot(x = 'time', y = 'usa_pres', data = df_ws_wnd, linestyle = 'dashed-dotted', color = 'black', label = 'IBTrACS')
        plt.legend()
        plt.title(f'Min MSLP in North Atlantic over {storm_name} time')
        plt.tight_layout()
        plt.show()

        df_runs_ptot, ratios_sum_ptot = get_ratios(ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', mode = 'mean')

        # load df_runs_wind and df_runs_mslp from sandy
        df_runs_wind = pd.read_csv(f'D:/paper_3/data/storm_composite/{storm_name}_wind.csv', index_col = 0)
        df_runs_mslp = pd.read_csv(f'D:/paper_3/data/storm_composite/{storm_name}_mslp.csv', index_col = 0)
        df_runs_ptot = pd.read_csv(f'D:/paper_3/data/storm_composite/{storm_name}_ptot.csv', index_col = 0)
        
        # plot the space between minimum and maximum values in df_runs_mslp for each timestep
        min_values_mslp = df_runs_mslp.groupby('time')['mslp'].min()
        max_values_mslp = df_runs_mslp.groupby('time')['mslp'].max()

        min_values_wind = df_runs_wind.groupby('time')['U10m'].min()
        max_values_wind = df_runs_wind.groupby('time')['U10m'].max()

        min_values_ptot = df_runs_ptot.groupby('time')['Ptot'].min()
        max_values_ptot = df_runs_ptot.groupby('time')['Ptot'].max()
        
        datetime_value = datetime.strptime("2012-10-29T18:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
        datetime_value_ini = datetime.strptime("2012-10-29T12:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")
        datetime_value_end = datetime.strptime("2012-10-30T03:00:00.000000", "%Y-%m-%dT%H:%M:%S.%f")

        # Figure
        doublecol = 14 * 0.393701
        # define figure text as 8


        fig, axs = plt.subplots(3,1,figsize=(doublecol, doublecol*1.5)) # sharex=True, sharey=True) 
        plt.rcParams.update({'font.size': 9})
        # panel 2
        var = 'mslp'
        sns.lineplot(x = 'time', y = 'mslp', data = ds_rel_hist_echam.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ECHAM6', ax = axs[0])
        sns.lineplot(x = 'time', y = 'mslp', data = ds_rel_hist_era5.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ERA5', ax = axs[0])
        sns.lineplot(x = 'time', y = 'mslp', data = ds_rel_hist_merra2.min(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'Merra2', ax = axs[0])
        sns.lineplot(x = 'time', y = 'usa_pres', data = df_ws_wnd, linestyle = 'dashdot', color = 'black', label = 'IBTrACS', ax = axs[0])
        axs[0].fill_between(max_values_mslp.index, min_values_mslp, max_values_mslp, alpha=0.4, color = 'darkred', zorder = 1, label = 'spectrally nudged storylines')
        axs[0].axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray', zorder = 0)
        axs[0].set_title(f'a)', loc='left', pad=0)
        axs[0].get_legend().remove()
        axs[0].set_ylabel('MSLP [hPa]')
        axs[0].set_xlabel('')
        axs[0].text(datetime_value_ini + (datetime_value_end - datetime_value_ini) / 2, 1008, 'NYC', ha='center', va='center', fontsize=8, color='black')

        sns.lineplot(x = 'time', y = 'U10m', data = ds_rel_hist_echam.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ECHAM6', ax = axs[1])
        sns.lineplot(x = 'time', y = 'U10m', data = ds_rel_hist_era5.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ERA5', ax = axs[1])
        sns.lineplot(x = 'time', y = 'U10m', data = ds_rel_hist_merra2.max(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'Merra2', ax = axs[1])
        sns.lineplot(x = 'time', y = 'usa_wind', data = df_ws_wnd, linestyle = 'dashdot', color = 'black', label = 'IBTrACS', ax = axs[1])
        axs[1].fill_between(min_values_wind.index, min_values_wind, max_values_wind, alpha=0.4, color = 'darkred', zorder = 1)
        axs[1].axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray', zorder = 0)
        axs[1].set_title(f'b)', loc='left', pad=0)
        axs[1].get_legend().remove()
        axs[1].set_ylabel('wind speed [m/s]')
        axs[1].set_xlabel('')
        axs[1].text(datetime_value_ini + (datetime_value_end - datetime_value_ini) / 2, df_ws_wnd['usa_wind'].max() * 0.99, 'NYC', ha='center', va='center', fontsize=8, color='black')

        sns.lineplot(x = 'time', y = 'Ptot', data = ds_rel_hist_echam.mean(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ECHAM6', ax = axs[2])
        sns.lineplot(x = 'time', y = 'Ptot', data = ds_rel_hist_era5.mean(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'ERA5', ax = axs[2])
        sns.lineplot(x = 'time', y = 'Ptot', data = ds_rel_hist_merra2.mean(dim = ['lat', 'lon']).sel(time = slice(storm_track_hist.index.min(), storm_track_hist.index.max())), label = 'Merra2', ax = axs[2])
        axs[2].fill_between(max_values_ptot.index, min_values_ptot, max_values_ptot, alpha=0.4, color = 'darkred', zorder = 1)
        axs[2].axvspan(datetime_value_ini, datetime_value_end, alpha=0.5, color='lightgray', zorder = 0)
        axs[2].set_title(f'c)', loc='left', pad=0)
        axs[2].get_legend().remove()
        axs[2].set_ylabel('precipitation [mm/h]')
        axs[2].set_xlabel('Date (Oct 2012)')
        axs[2].text(datetime_value_ini + (datetime_value_end - datetime_value_ini) / 2, max_values_ptot.max() * 0.99, 'NYC', ha='center', va='center', fontsize=8, color='black')

        import matplotlib.dates as mdates
        date_format = mdates.DateFormatter("%d") # TODO: try only days
        axs[0].xaxis.set_major_formatter(date_format)
        axs[1].xaxis.set_major_formatter(date_format)
        axs[2].xaxis.set_major_formatter(date_format)
        axs[0].set_xticks(axs[0].get_xticks()[::2])
        axs[1].set_xticks(axs[1].get_xticks()[::2])
        axs[2].set_xticks(axs[2].get_xticks()[::2])
        
        axs[0].spines['right'].set_visible(True)
        axs[1].spines['right'].set_visible(True)
        axs[2].spines['right'].set_visible(True)
        axs[0].spines['top'].set_visible(True)
        axs[1].spines['top'].set_visible(True)
        axs[2].spines['top'].set_visible(True)

        fig.autofmt_xdate(rotation=0, ha='center')
        
        # now add a legend for all plots
        handles, labels = axs[0].get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        sorted_unique = sorted(unique, key=lambda x: x[1])
        plt.legend(*zip(*sorted_unique), loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3, frameon = False)
        plt.tight_layout()
        plt.savefig(rf'd:\paper_3\data\Figures\paper_figures\paper_fig_si2_hist_echam_era5.png', dpi=300, bbox_inches='tight')
        plt.show()


        # Plot storm composites#############################
        # Storm tracker for ensemble members
        storm_track_factual_ens = storm_tracker_mslp_ens(ds_factual_ens, df_ws, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        storm_track_counter_ens = storm_tracker_mslp_ens(ds_counter_ens, df_ws, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        storm_track_plus2_ens = storm_tracker_mslp_ens(ds_plus2_ens, df_ws, smooth_step = 'savgol', large_box=large_box, small_box=small_box)
        # nOW PLOT EVERYTHING TOGETHER
        figs_all = plot_minimum_track_validation(storm_track_hist, storm_track_factual_ens, storm_track_counter_ens, storm_track_plus2_ens, df_ws_hist, storm_name)

        radius = 10
        ds_rel_counter_ens = storm_composite_ens(ds_counter_ens, df_center = storm_track_counter_ens, radius = radius)
        ds_rel_factual_ens = storm_composite_ens(ds_factual_ens, df_center = storm_track_factual_ens, radius = radius)
        ds_rel_plus2_ens = storm_composite_ens(ds_plus2_ens, df_center = storm_track_plus2_ens, radius = radius)

        # delta for sandy duration on new york
        # delta = ds_rel_hist_echam.mean() / ds_rel_factual_ens.mean()
        #calculate a new delta using max for u10m, min for mslp and mean for Ptot
        # delta = xr.Dataset({'U10m': ds_rel_hist_echam['U10m'].max() / ds_rel_factual_ens['U10m'].max(['time','lat','lon']).mean(),
        #                     'Ptot': ds_rel_hist_echam['Ptot'].mean() / ds_rel_factual_ens['Ptot'].mean(),
        #                     'mslp': ds_rel_hist_echam['mslp'].min() / ds_rel_factual_ens['mslp'].min(['time','lat','lon']).mean()})

        # ds_rel_counter_ens = ds_rel_counter_ens * delta
        # ds_rel_factual_ens = ds_rel_factual_ens * delta
        # ds_rel_plus2_ens = ds_rel_plus2_ens * delta

        
        # composites for each GW level - ECHAM historical
        fig_max_wind_ens = scenarios_timeseries_four_plot(ds_rel_hist_echam, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'U10m', storm_name, hist_name='ECHAM 6.1', mode = 'max', graph = 'normal')
        fig_sum_ptot_ens = scenarios_timeseries_four_plot(ds_rel_hist_echam, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', storm_name, hist_name='ECHAM 6.1',  mode = 'mean', graph = 'normal')
        fig_min_mslp_ens = scenarios_timeseries_four_plot(ds_rel_hist_echam, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'mslp', storm_name, hist_name='ECHAM 6.1', mode = 'min', graph = 'normal')

        fig_composites_collection(fig_min_mslp_ens, fig_max_wind_ens, fig_sum_ptot_ens, storm_name, mode = 'ens_hist_echam')
        
        # composites for each GW level - ERA5 historical
        fig_max_wind_ens_era5 = scenarios_timeseries_four_plot(ds_rel_hist_era5, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'U10m', storm_name, hist_name='ERA5', mode = 'max', graph = 'normal')
        fig_sum_ptot_ens_era5 = scenarios_timeseries_four_plot(ds_rel_hist_era5, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'Ptot', storm_name, hist_name='ERA5',  mode = 'mean', graph = 'normal')
        fig_min_mslp_ens_era5 = scenarios_timeseries_four_plot(ds_rel_hist_era5, ds_rel_counter_ens, ds_rel_factual_ens, ds_rel_plus2_ens, 'mslp', storm_name, hist_name='ERA5',  mode = 'min', graph = 'normal')

        fig_composites_collection(fig_min_mslp_ens_era5, fig_max_wind_ens_era5, fig_sum_ptot_ens_era5, storm_name, mode = 'ens_hist_era5')

        
        
