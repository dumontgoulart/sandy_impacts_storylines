# -*- coding: utf-8 -*-

import os
os.chdir('D:/paper_3/code')
import xarray as xr 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import storm_functions as sf

# import seaborn as sns
import datetime as dt
import storm_functions as sf

def figures_gtsm(name_file, storm_name, city, storm_duration, reg_box_sel, vmin, vmax):
    data_dir = f'D:\\paper_3\\data\\'
    fig_dir = f'D:\\paper_3\\data\\Figures\\local_gtsm_runs\\{storm_name}\\'

    storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
    city = city
    
    ds_gtsm = xr.open_dataset(name_file) #.sel(lat = slice(reg_box_xynthia['lat_min'], reg_box_xynthia['lat_max']), lon = slice(reg_box_xynthia['lon_min'],reg_box_xynthia['lon_max']))
    if 'x' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'x':'station_x_coordinate', 'y':'station_y_coordinate'})
    if 'lat' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'lon':'station_x_coordinate', 'lat':'station_y_coordinate'})

    stations_sel = np.where((ds_gtsm.station_x_coordinate < reg_box_sel['lon_max']) & 
                                (ds_gtsm.station_x_coordinate > reg_box_sel['lon_min']) &
                                (ds_gtsm.station_y_coordinate < reg_box_sel['lat_max']) &
                                (ds_gtsm.station_y_coordinate > reg_box_sel['lat_min']))[0]

    ds_gtsm = ds_gtsm.isel(stations=stations_sel)

    da_gtsm_waterlevel = ds_gtsm['waterlevel']


    if 'stations' in da_gtsm_waterlevel.coords:
        da_gtsm_waterlevel = da_gtsm_waterlevel.drop_vars('stations') 
    ### Plot a time series
    lat = storm_df[storm_df.cities == city].lat
    lon = storm_df[storm_df.cities == city].lon

    sta = sf.closest_gtsm_station(da_gtsm_waterlevel, lon, lat)[0]
    ts = da_gtsm_waterlevel.sel(stations=sta)
    
    total_surge_city = np.round(ds_gtsm.sel(time = slice(storm_duration[0],storm_duration[1]))["waterlevel"].sum().item(),3)
    print("total surge during storm:",total_surge_city)

    figure_name = name_file.split("\\")[-1].split(".")[0]
    print(figure_name)
    plt.figure(figsize=(8, 8))
    ts.plot() #label = f'{variable} levels'
    plt.plot(ts.idxmax(), ts.sel(time = ts.idxmax()), '*', c='red', markersize=12)
    plt.savefig(f'{fig_dir}timseries_{figure_name}.png', dpi=150)
    plt.close()

    # Plot on the time of maximum surge
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ds_gtsm.sel(time = ts.idxmax()).plot.scatter(
                        x='station_x_coordinate', 
                        y='station_y_coordinate', 
                        hue = 'waterlevel',
                        s=80,
                        edgecolor='none',
                        vmin=vmin,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        ax=ax,
                        cmap = 'RdBu_r',
                        cbar_kwargs={'shrink':0.6},
                        );
    ax.scatter(lon, lat, 300, c='green', edgecolors='black', marker='*',
                transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m') #ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    plt.title(f'{storm_name} storm surge - total surge: {total_surge_city}')
    plt.savefig(f'{fig_dir}map_{figure_name}.png', dpi=200)
    plt.close()

    # Plot MAXIMUM SURGE ACROSS TIME IN EACH LOCATION
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ds_gtsm.sel(time = slice(storm_duration[0],storm_duration[1])).max(dim='time').plot.scatter(
                        x='station_x_coordinate', 
                        y='station_y_coordinate', 
                        hue = 'waterlevel',
                        s=80,
                        edgecolor='none',
                        vmin=0,
                        vmax=vmax,
                        transform=ccrs.PlateCarree(),
                        ax=ax,
                        cbar_kwargs={'shrink':0.6},
                        );
    ax.scatter(lon, lat, 300, c='green', edgecolors='black', marker='*',
                transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.coastlines(resolution='10m') #ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    plt.title(f"{storm_name} max storm surge - total surge: {np.round(ds_gtsm.sel(time = slice(storm_duration[0],storm_duration[1])).max(dim='time')['waterlevel'].sum().item(),3)}")
    plt.savefig(f'{fig_dir}map_max_surge_{figure_name}.png', dpi=200)
    plt.close()

def gtsm_timeseries(name_file, storm_name, city, storm_duration, reg_box_sel, vmin, vmax):
    data_dir = f'D:\\paper_3\\data\\'
    fig_dir = f'D:\\paper_3\\data\\Figures\\local_gtsm_runs\\{storm_name}\\'

    storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
    city = city
    
    ds_gtsm = xr.open_dataset(name_file) #.sel(lat = slice(reg_box_xynthia['lat_min'], reg_box_xynthia['lat_max']), lon = slice(reg_box_xynthia['lon_min'],reg_box_xynthia['lon_max']))
    if 'x' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'x':'station_x_coordinate', 'y':'station_y_coordinate'})
    if 'lat' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'lon':'station_x_coordinate', 'lat':'station_y_coordinate'})

    if storm_duration:
        ds_gtsm = ds_gtsm.sel(time = slice(storm_duration[0],storm_duration[1]))

    stations_sel = np.where((ds_gtsm.station_x_coordinate < reg_box_sel['lon_max']) & 
                                (ds_gtsm.station_x_coordinate > reg_box_sel['lon_min']) &
                                (ds_gtsm.station_y_coordinate < reg_box_sel['lat_max']) &
                                (ds_gtsm.station_y_coordinate > reg_box_sel['lat_min']))[0]

    ds_gtsm = ds_gtsm.isel(stations=stations_sel)

    da_gtsm_waterlevel = ds_gtsm['waterlevel']


    if 'stations' in da_gtsm_waterlevel.coords:
        da_gtsm_waterlevel = da_gtsm_waterlevel.drop_vars('stations') 
    ### Plot a time series
    lat = storm_df[storm_df.cities == city].lat
    lon = storm_df[storm_df.cities == city].lon

    sta = sf.closest_gtsm_station(da_gtsm_waterlevel, lon, lat)[0]
    ts = da_gtsm_waterlevel.sel(stations=sta)
    print(ts.idxmax().values) 
    return ts


def gtsm_spatial(name_file, storm_name, city, storm_duration, reg_box_sel, vmin, vmax):
    data_dir = f'D:\\paper_3\\data\\'
    fig_dir = f'D:\\paper_3\\data\\Figures\\local_gtsm_runs\\{storm_name}\\'

    storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
    city = city
    
    ds_gtsm = xr.open_dataset(name_file) #.sel(lat = slice(reg_box_xynthia['lat_min'], reg_box_xynthia['lat_max']), lon = slice(reg_box_xynthia['lon_min'],reg_box_xynthia['lon_max']))
    if 'x' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'x':'station_x_coordinate', 'y':'station_y_coordinate'})
    if 'lat' in ds_gtsm.coords:
        ds_gtsm = ds_gtsm.rename({'lon':'station_x_coordinate', 'lat':'station_y_coordinate'})

    stations_sel = np.where((ds_gtsm.station_x_coordinate < reg_box_sel['lon_max']) & 
                                (ds_gtsm.station_x_coordinate > reg_box_sel['lon_min']) &
                                (ds_gtsm.station_y_coordinate < reg_box_sel['lat_max']) &
                                (ds_gtsm.station_y_coordinate > reg_box_sel['lat_min']))[0]

    sta = sf.closest_gtsm_station(da_gtsm_waterlevel, lon, lat)[0]
    ts = da_gtsm_waterlevel.sel(stations=sta)

    da_gtsm_waterlevel = ds_gtsm['waterlevel']

    if 'stations' in da_gtsm_waterlevel.coords:
        da_gtsm_waterlevel = da_gtsm_waterlevel.drop_vars('stations') 
    
    return da_gtsm_waterlevel.sel(time = ts.idxmax())
    
##########################################################################################

storm_name = 'sandy'
city= 'New York'
storm_duration = ['2012-10-28','2012-11-01']
reg_box_sandy = {'lon_min':-89.472656, 'lon_max':-30.078125, 'lat_min':7.532249, 'lat_max':56} #-89.472656,7.532249,-30.078125,56
vmin, vmax = -3,3


directory = f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid'

for name_file in os.listdir(directory):
    file = os.path.join(directory, name_file)
    figures_gtsm(file, storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)

ts_counter1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_counter2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_counter3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_tidesonly = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_tidesonly_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)


plt.figure(figsize=(8, 8))
ts_counter1.plot() #label = f'{variable} levels'
ts_counter2.plot() #label = f'{variable} levels'
ts_counter3.plot() #label = f'{variable} levels'
ts_factual1.plot() #label = f'{variable} levels'
ts_factual2.plot() #label = f'{variable} levels'
ts_factual3.plot() #label = f'{variable} levels'
ts_plus2_1.plot() #label = f'{variable} levels'
ts_plus2_2.plot() #label = f'{variable} levels'
ts_plus2_3.plot() #label = f'{variable} levels'
# plot tidesonly with a dashed linestyle
ts_tidesonly.plot(label = 'tides only', linestyle = 'dashed', color = 'black') #label = f'{variable} levels'
# save plot
plt.savefig(f'D:\\paper_3\\data\\Figures\\local_gtsm_runs\\{storm_name}\\timseries_{storm_name}_all.png', dpi=350)
plt.legend()
plt.show()

ts_counter1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_counter2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_counter3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_factual3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_plus2_3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)
ts_tidesonly = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_tidesonly_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_sandy, vmin, vmax)

# subtract all waterlevel scenarios from the tides only waterlevel
surge_counter1 = ts_counter1 - ts_tidesonly
surge_counter2 = ts_counter2 - ts_tidesonly
surge_counter3 = ts_counter3 - ts_tidesonly
surge_factual1 = ts_factual1 - ts_tidesonly
surge_factual2 = ts_factual2 - ts_tidesonly
surge_factual3 = ts_factual3 - ts_tidesonly
surge_plus2_1 = ts_plus2_1 - ts_tidesonly
surge_plus2_2 = ts_plus2_2 - ts_tidesonly
surge_plus2_3 = ts_plus2_3 - ts_tidesonly

# now plot the graphs of the surge
plt.figure(figsize=(8, 8))
plt.fill_between(surge_counter1.time, 0, surge_factual2.max(), where=ts_tidesonly > ts_tidesonly.mean(), facecolor='grey', alpha=0.4, label = 'High tide periods')
surge_counter1.plot() #label = f'{variable} levels'
surge_counter2.plot() #label = f'{variable} levels'
surge_counter3.plot() #label = f'{variable} levels'
surge_factual1.plot() #label = f'{variable} levels'
surge_factual2.plot() #label = f'{variable} levels'
surge_factual3.plot() #label = f'{variable} levels'
surge_plus2_1.plot() #label = f'{variable} levels'
surge_plus2_2.plot() #label = f'{variable} levels'
surge_plus2_3.plot() #label = f'{variable} levels'
# ts_tidesonly.plot(label = 'Tide', linestyle = 'dashed', color = 'black') #label = f'{variable} levels'
plt.legend(frameon=False, loc='upper right', ncol = 2) 
plt.title('Surge levels for alternative realisations of Sandy (coloured lines)')
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_si_surge_tide.png', dpi=300, bbox_inches='tight')
plt.show()

# XYNTHIA
storm_name = 'xynthia'
city= 'La Rochelle'
storm_duration = ['2010-02-27','2010-03-01']
reg_box_xynthia = {'lon_min':-5, 'lon_max':-0.5, 'lat_min':44, 'lat_max':48}
vmin, vmax = -4,4

directory = f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid'

for name_file in os.listdir(directory):
    file = os.path.join(directory, name_file)
    figures_gtsm(file, storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)


ts_counter1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_counter2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_counter3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_counter_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_factual1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_factual2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_factual3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_factual_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_plus2_1 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_1_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_plus2_2 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_2_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_plus2_3 = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_plus2_3_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)
ts_tidesonly = gtsm_timeseries(f'D:\\paper_3\\data\\gtsm_local_runs\\{storm_name}_fine_grid\\gtsm_fine_{storm_name}_tidesonly_0000_his_waterlevel.nc', storm_name, city, storm_duration, reg_box_xynthia, vmin, vmax)

plt.figure(figsize=(8, 8))
ts_counter1.plot() #label = f'{variable} levels'
ts_counter2.plot() #label = f'{variable} levels'
ts_counter3.plot() #label = f'{variable} levels'
ts_factual1.plot() #label = f'{variable} levels'
ts_factual2.plot() #label = f'{variable} levels'
ts_factual3.plot() #label = f'{variable} levels'
ts_plus2_1.plot() #label = f'{variable} levels'
ts_plus2_2.plot() #label = f'{variable} levels'
ts_plus2_3.plot() #label = f'{variable} levels'
ts_tidesonly.plot(label = 'tides only') #label = f'{variable} levels'
# plt.plot(ts_counter1.idxmax(), ts_counter1.sel(time = ts_counter1.idxmax()), '*', c='red', markersize=12)
plt.legend()
plt.show()

surge = ts_counter2 - ts_tidesonly

ts_counter2.plot(label = 'surge & tides') #label = f'{variable} levels'
ts_tidesonly.plot(label = 'tides only') #label = f'{variable} levels'
surge.plot(label = 'surges (s&t - tides)') #label = f'{variable} levels'
# plt.plot(ts_counter1.idxmax(), ts_counter1.sel(time = ts_counter1.idxmax()), '*', c='red', markersize=12)
plt.legend()
plt.show()