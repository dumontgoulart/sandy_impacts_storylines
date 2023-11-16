""" 
Script based and inspired by Dewi Le Bars
 """

import os
os.chdir('D:/paper_3/code')
import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as cfeature
import matplotlib as mpl
# Import local functions from storm_functions.py
import storm_functions as sf
# Basic plot setups
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

#################################
# Define functions for plotting
def plot_all(storm, city, day_s, day_e, box, export_fig, dataset = 'new', variable = 'waterlevel', time_res = 'hourly', rp = None, generate_nc = False):
    '''Plot storm surge time series, large plot of peak sea level and an overview 
    of the storm surge with 9 maps, 4 before and 4 after the maximum sea level.'''
    
    gtsm_ts_ds_sel = sf.select_gtsm_ds(day_s, day_e, box, dataset = dataset, variable = variable, time_res = time_res)
    if 'stations' in gtsm_ts_ds_sel.coords:
            gtsm_ts_ds_sel_2 = gtsm_ts_ds_sel.drop_vars('stations') 
    ### Plot a time series
    lat = storm_df[storm_df.cities == city].lat
    lon = storm_df[storm_df.cities == city].lon

    sta = sf.closest_gtsm_station(gtsm_ts_ds_sel_2, lon, lat)[0]

    ts = gtsm_ts_ds_sel_2.sel(stations=sta)[variable]

    print('Time at at the maximum storm surge:')
    print(gtsm_ts_ds_sel_2.sel(time = ts.idxmax()).time.values)
       
    plt.figure(figsize=(8, 8))
    ts.plot() #label = f'{variable} levels'
    plt.plot(ts.idxmax(), ts.sel(time = ts.idxmax()), '*', c='red', markersize=12)

    def timeseries_closest_station(box, rp, variable, ref = None):
        ds_rp = sf.select_ds_rp(box, rp = rp, variable = variable, ref = None)
        if 'stations' in ds_rp.coords:
                ds_rp = ds_rp.drop_vars('stations') 
        
        sta_rp = sf.closest_gtsm_station(ds_rp, lon, lat)[0]
        ts_rp = ds_rp.sel(stations=sta_rp)[list(ds_rp.keys())[0]]
        return ts_rp
            
    if rp is not None:
        if rp is 'multiple':
            ts_rp10 = timeseries_closest_station(box, '10', variable, ref = None)
            ts_rp50 = timeseries_closest_station(box, '50', variable, ref = None)
            ts_rp100 = timeseries_closest_station(box, '100', variable, ref = None)
            
            plt.axhline(ts_rp10, linestyle = 'dashed', color = 'yellow', label = f'RP-10')
            plt.axhline(ts_rp50, linestyle = 'dashed', color = 'orange', label = f'RP-50')
            plt.axhline(ts_rp100, linestyle = 'dashed', color = 'red', label = f'RP-100')
        
        elif rp in ['10','50', '100']:
            ts_rp = timeseries_closest_station(box = box, rp = rp, variable = variable, ref = None)

            plt.axhline(ts_rp, linestyle = 'dashed', color = 'black', label = f'RP-{rp}')
        else:
            raise ValueError('RP can only be "multiple" or 10,50,100.')

    plt.title(f'{storm} {variable} time series at {city}')
    plt.legend()
    plt.tight_layout()
    
    if export_fig:
        plt.savefig(f'{fig_dir}{storm}_{city}_timeseries_{variable}.pdf', dpi=150)
        
    # Plot on the time of maximum surge
    plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    gtsm_ts_ds_sel_2.sel(time = ts.idxmax()).plot.scatter(
                        x='station_x_coordinate', 
                        y='station_y_coordinate', 
                        hue = variable,
                        # markersize=0,
                        #size=7,
                        # robust=True,
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

    plt.title(f'{storm} storm surge')
    
    if export_fig:
        plt.savefig(f'{fig_dir}{storm}_{city}_map_max_{variable}.pdf', dpi=150)

    plt.close('all')

# generate a .nc file with the coastal stations within the boundary 
# conditions and their water levels during the time interval selected:
    if generate_nc:
        if 'station_x_coordinate' or 'station_y_coordinate' in gtsm_ts_ds_sel.coords:
                gtsm_ts_ds_sel = gtsm_ts_ds_sel.rename({'station_y_coordinate': 'lat', 'station_x_coordinate': 'lon'})
                # gtsm_ts_ds_sel = gtsm_ts_ds_sel[["stations", "time",'lon', 'lat',variable]]

        gtsm_ts_ds_sel.to_netcdf(f'{data_dir}gtsm_outputs/gtsm_{storm}_{variable}_{day_s}_{day_e}.nc')

######################################################
# Main code
######################################################
# Load data
data_dir = 'D:/paper_3/data/'
fig_dir = data_dir+'Figures/'
gtsm_rp_dir = data_dir+'GTSM_Copernicus_ReturnPeriods/'
export_fig = True
# Load cities for the storms:
storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
######################################################


######################################################
# Generate plots of the storm at different stages and with a city as reference.
# Xaver
storm = 'Xaver'
reg_box_xaver = {'lon_min':-4, 'lon_max':9.3, 'lat_min':50, 'lat_max':58}
day_s, day_e = '2013-12-01', '2013-12-07'
rp = 'multiple'
variable = 'waterlevel' #'surge', 'waterlevel'
dataset = 'new'
time_res = 'hourly' # '10min' for (very minimal) improvements but slower processing
generate_nc = True

plot_all(storm=storm, city='Cuxhaven', day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig, generate_nc = generate_nc)

plot_all(storm=storm, city='Aalborg', day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

plot_all(storm=storm, city='Hamburg',day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

plot_all(storm=storm, city='Edinburgh', day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

plot_all(storm=storm, city='Newcastle', day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

plot_all(storm=storm, city='Dover', day_s=day_s, day_e=day_e, box=reg_box_xaver, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

# Xynthia
storm = 'Xynthia'
reg_box_xynthia = {'lon_min':-10, 'lon_max':5, 'lat_min':40, 'lat_max':54}
day_s, day_e = '2010-02-23', '2010-03-01' # day_s < day_e
rp = None
variable = 'waterlevel' #'surge', 'waterlevel'
dataset = 'new'
time_res = 'hourly' # '10min' for (very minimal) improvements but slower processing
generate_nc = False

plot_all(storm='Xynthia' , city='La Rochelle', day_s=day_s, day_e=day_e, box=reg_box_xynthia, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig, generate_nc = generate_nc)

# Rotterdam
plot_all(storm='Xynthia' , city='Rotterdam', day_s=day_s, day_e=day_e, box=reg_box_xynthia, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

# Vigo
plot_all(storm='Xynthia' , city='Vigo', day_s=day_s, day_e=day_e, box=reg_box_xynthia, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

# La Coruna
plot_all(storm='Xynthia' , city='La Coruna', day_s=day_s, day_e=day_e, box=reg_box_xynthia, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig)

# test Italy
storm = 'Italy'
reg_box_italy = {'lon_min':12.05,'lat_min':45.22,'lon_max':12.999,'lat_max':45.65}
day_s, day_e = '2010-02-01', '2010-02-10' # day_s < day_e
rp = None
variable = 'waterlevel' #'surge', 'waterlevel'
dataset = 'new'
time_res = 'hourly' # '10min' for (very minimal) improvements but slower processing

plot_all(storm= storm , city='Venice', day_s=day_s, day_e=day_e, box=reg_box_italy, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig, generate_nc = True)
        
# Sandy
storm = 'sandy'
reg_box_sandy = {'lon_min':-82.001953, 'lon_max':-68.686523, 'lat_min':35.657296, 'lat_max':44.166445} #-82.001953,35.657296,-68.686523,44.166445
day_s, day_e = '2012-10-25', '2012-11-02' # day_s < day_e
rp = 'multiple'
variable = 'waterlevel' #'surge', 'waterlevel'
dataset = 'new'
time_res = 'hourly' # '10min' for (very minimal) improvements but slower processing
generate_nc = True

plot_all(storm=storm , city='New York', day_s=day_s, day_e=day_e, box=reg_box_sandy, dataset = dataset, 
        variable = variable, time_res = time_res, rp =  rp, export_fig=export_fig, generate_nc = generate_nc)
