# -*- coding: utf-8 -*-
'''
Script to determine and locate multiple storms in a given region.
1) Identification: Use the historical storm tracker (IBTracks) to define the storms of interest (either the most intense, or above certain thresholds);
2) Tracking: For each storm, find in the SN data the counterfactuals paths following the function <storm_tracker>
3) Analysis: Caclualte the statistical differences across all storms over the different GW levels.
Input: IBTracs data, SN data, physical domain where to find the storms.
Output: Statistical differences that can be attributable to GW.
'''
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
import datetime as dt
# # local libraries
# import sn_storm_preprocessing as prep
# import sn_storm_tracker_composite as trac


# storm_list = ['sandy', 'xynthia', 'xaver', 'igor', 'earl', 'ophelia','katia', 'irene'] # add later gonzalo, rafael, 

# for storm_name in storm_list:
#     print(storm_name)
#     ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws = prep.preprocess_data(storm_name = storm_name, plots = False)

#     trac.track_storm(storm_name, ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws, smoothing_step = 4, radius = 10)


import storm_functions as sf

data_dir = 'D:/paper_3/data/'

storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
city = 'La Rochelle'
reg_box_xynthia = {'lon_min':-10, 'lon_max':5, 'lat_min':40, 'lat_max':54}

ds_gtsm = xr.open_dataset('D:\paper_3\data\gtsm_local_runs\gtsm_fine_xynthia_0000_his.nc') #.sel(lat = slice(reg_box_xynthia['lat_min'], reg_box_xynthia['lat_max']), lon = slice(reg_box_xynthia['lon_min'],reg_box_xynthia['lon_max']))
stations_sel = np.where((ds_gtsm.station_x_coordinate < reg_box_xynthia['lon_max']) & 
                            (ds_gtsm.station_x_coordinate > reg_box_xynthia['lon_min']) &
                            (ds_gtsm.station_y_coordinate < reg_box_xynthia['lat_max']) &
                            (ds_gtsm.station_y_coordinate > reg_box_xynthia['lat_min']))[0]

ds_gtsm = ds_gtsm.isel(stations=stations_sel)

for var in ds_gtsm.keys():
    print(var)

da_gtsm_waterlevel = ds_gtsm['waterlevel']


da_gtsm_waterlevel.mean('time')

if 'stations' in da_gtsm_waterlevel.coords:
    da_gtsm_waterlevel = da_gtsm_waterlevel.drop_vars('stations') 
### Plot a time series
lat = storm_df[storm_df.cities == city].lat
lon = storm_df[storm_df.cities == city].lon

sta = sf.closest_gtsm_station(da_gtsm_waterlevel, lon, lat)[0]

ts = da_gtsm_waterlevel.sel(stations=sta)

plt.figure(figsize=(8, 8))
ts.plot() #label = f'{variable} levels'
plt.plot(ts.idxmax(), ts.sel(time = ts.idxmax()), '*', c='red', markersize=12)

# Plot on the time of maximum surge
plt.figure(figsize=(8, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

ds_gtsm.sel(time = ts.idxmax()).plot.scatter(
                    x='station_x_coordinate', 
                    y='station_y_coordinate', 
                    hue = 'waterlevel',
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

plt.title(f'xynthia storm surge')

# #GTSM file format needed:
# latitude: 90-> -90
# longitude -180 to 180
# time = normal
# add attributes

def convert2FMH(input_dir, output_dir, name_file, date_string):
# create variables 'dictionary' for filename pattern, variable name in ncin, variable name in ncout  
    var_dict = {
    "u10" : {
        "standard_name" : "eastward_wind",
        "long_name" : "10 metre U wind component",
        "units" : "m s**-1",
        "scale_factor" : float(0.01),
        "offset" : float(0)},
    "v10" : {
        "standard_name" : "northward_wind",
        "long_name" : "10 metre V wind component",
        "units" : "m s**-1",
        "scale_factor" : float(0.01),
        "offset" : float(0)},
    "msl" : {
        "standard_name" : "air_pressure",
        "long_name" : "Mean sea level pressure",
        "units" : "Pa",
        "scale_factor" : float(1),
        "offset" : float(100000)}}
    coor_dict = {
    "latitude" : {
        "standard_name" : "latitude",
        "long_name" : "latitude",
        "units" : "degrees north"},
    "longitude" : {
        "standard_name" : "longitude",
        "long_name" : "longitude",
        "units" : "degrees_east"}}
    
    time_unit = "hours since 1900-01-01 00:00:00" # hard coded
    # define ref/start/stop times 
    tdate = dt.datetime.strptime(date_string, '%Y_%m').date()
    spinup_period = [1,1,15,0] # imposed 1 day zero, 1 day transition, 15 days spinup --> compute days of spinup required
    date_start_zero = dt.datetime(tdate.year,tdate.month,1)-dt.timedelta(days=int(np.sum(spinup_period[0:3]))) #eg 15 dec
    date_start_transition = date_start_zero+dt.timedelta(days=spinup_period[0]) #eg 16 dec
    date_start_spinup = date_start_zero+dt.timedelta(days=spinup_period[0]+spinup_period[1]) #eg 17 dec
    # import dataset
    ds = xr.open_mfdataset(input_dir+"\\"+name_file, chunks='auto', parallel=True).sel(time=slice(date_start_zero, None)); ds.close()
    date_end = dt.datetime(tdate.year,ds.time.dt.month[-1].item(), ds.time.dt.day[-1].item())
    #check if lat and lon
    if 'lat' in ds.coords:
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
    if 'u10m' in ds.var():
        ds = ds.rename({'u10m': 'u10'})
    if 'v10m' in ds.var():
        ds = ds.rename({'v10m': 'v10'})  
    if 'mslp' in ds.var():
        ds = ds.rename({'mslp': 'msl'})  
    ds = ds.sortby(ds.longitude)
    ds = ds.sortby(ds.latitude, ascending=False)
    
    for varname in ds.keys():
        # drop times   
        ds = ds.where(ds.time >= np.datetime64(date_start_spinup), 0) # values to zero for initalization SLR
        bool = (ds.time > np.datetime64(date_start_transition)) & (ds.time < np.datetime64(date_start_spinup)) # select transition period
        ds = ds.where(~bool, drop=True) # drop times for transition period
        # set attributes + encoding
        ds.attrs['standard_name'] = var_dict[varname]['standard_name']
        ds.attrs['long_name'] = var_dict[varname]['long_name']
        ds.attrs['units'] = var_dict[varname]['units']
        ds.attrs['coordinates'] = 'longitude latitude'
        
        for coor in coor_dict.keys():
            ds[coor].attrs['standard_name'] = coor_dict[coor]['standard_name']
            ds[coor].attrs['units'] = coor_dict[coor]['units']
            ds[coor].attrs['long_name'] = coor_dict[coor]['long_name']
        
        # write output
        comp = dict(scale_factor=var_dict[varname]['scale_factor'], add_offset=var_dict[varname]['offset'])  
        encoding = {varname: comp, 'time':{'units': time_unit}} 
        filename_out = f'{name_file.split( "." )[0]}_{varname}_gtsm.nc'
        output_path = output_dir + '\\' + filename_out
        ds.to_netcdf(output_path, encoding=encoding, format='NETCDF4', engine='netcdf4'); ds.close()

if __name__ == "__main__":

    # for folder in ['mslpa_hpa','u10m_m_s','v10m_m_s']:

    input_folder = r'd:\paper_3\data\spectral_nudging_data\xynthia\mslp_hPa'
    output_folder = input_folder + r'\output_fm'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.nc'):
            # Open input file
            convert2FMH(input_folder, output_folder, filename,'2010_02')


# def convert2FM(input_dir,date_string):
#     # create variables 'dictionary' for filename pattern, variable name in ncin, variable name in ncout  
#     var_dict = {
#     "u10" : {
#         "standard_name" : "eastward_wind",
#         "long_name" : "10 metre U wind component",
#         "units" : "m s**-1",
#         "scale_factor" : float(0.01),
#         "offset" : float(0)},
#     "v10" : {
#         "standard_name" : "northward_wind",
#         "long_name" : "10 metre V wind component",
#         "units" : "m s**-1",
#         "scale_factor" : float(0.01),
#         "offset" : float(0)},
#     "msl" : {
#         "standard_name" : "air_pressure",
#         "long_name" : "Mean sea level pressure",
#         "units" : "Pa",
#         "scale_factor" : float(1),
#         "offset" : float(100000)}}
#     coor_dict = {
#     "latitude" : {
#         "standard_name" : "latitude",
#         "long_name" : "latitude",
#         "units" : "degrees north"},
#     "longitude" : {
#         "standard_name" : "longitude",
#         "long_name" : "longitude",
#         "units" : "degrees_east"}}
#     time_unit = "hours since 1900-01-01 00:00:00" # hard coded
#     # define ref/start/stop times 
#     tdate = dt.datetime.strptime(date_string, '%Y_%m').date()
#     spinup_period = [1,1,15,0] # imposed 1 day zero, 1 day transition, 15 days spinup --> compute days of spinup required
#     date_start_zero = dt.datetime(tdate.year,tdate.month,1)-dt.timedelta(days=int(np.sum(spinup_period[0:3]))) #eg 15 dec
#     date_start_transition = date_start_zero+dt.timedelta(days=spinup_period[0]) #eg 16 dec
#     date_start_spinup = date_start_zero+dt.timedelta(days=spinup_period[0]+spinup_period[1]) #eg 17 dec
#     # import dataset
#     input_path = os.path.join(input_dir,f'BOT_t_HR255_Nd_SU_1015_counter_1_{tdate.year}*') 
#     ds = xr.open_mfdataset(input_path,chunks='auto',parallel=True).sel(time=slice(date_start_zero, None)); ds.close()
#     date_end = dt.datetime(tdate.year,ds.time.dt.month[-1].item(), ds.time.dt.day[-1].item())
#     #check if lat and lon
#     if 'lat' in ds.coords:
#         ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
#     if 'u10m' in ds.var():
#         ds = ds.rename({'u10m': 'u10'})
#     if 'v10m' in ds.var():
#         ds = ds.rename({'v10m': 'v10'})  
#     if 'mslp' in ds.var():
#         ds = ds.rename({'mslp': 'msl'})  
#     # copy latitude and create new longitude
#     lats = ds['latitude'][:]
#     lons = ds['longitude'][:]
#     part1 = (ds.longitude>178) #move 180:360 part to -180:0 so field now runs from longitute -180 to 180
#     part2 = (ds.longitude<182) #take a bit of overlap to avoid interpolation issues at edge
#     lons_new=np.hstack((lons[part1]-360,lons[part2]))

#     # load data 
#     for varname in ds.keys():
#         print(varname)
#         datasets = []
#         for itime, time in enumerate(ds.time):
#             var = ds[varname].isel(time=itime)
#             var_new = np.hstack((var[:,part1],var[:,part2]))
#             coords = {'latitude': lats, 'longitude': lons_new, 'time': time.values}
#             da = xr.DataArray(var_new, coords=coords, dims=['latitude', 'longitude'])
#             da.name = varname
#             dat = xr.concat([da],'time')
#             datasets.append(dat)
#         ds_var_merged = xr.concat(datasets, dim='time') 
#         # drop times   
#         ds_var_merged = ds_var_merged.where(ds_var_merged.time >= np.datetime64(date_start_spinup), 0) # values to zero for initalization SLR
#         bool = (ds_var_merged.time > np.datetime64(date_start_transition)) & (ds_var_merged.time < np.datetime64(date_start_spinup)) # select transition period
#         ds_var_merged = ds_var_merged.where(~bool, drop=True) # drop times for transition period
#         # set attributes + encoding
#         ds_var_merged.attrs['standard_name'] = var_dict[varname]['standard_name']
#         ds_var_merged.attrs['long_name'] = var_dict[varname]['long_name']
#         ds_var_merged.attrs['units'] = var_dict[varname]['units']
#         ds_var_merged.attrs['coordinates'] = 'longitude latitude'
#         for coor in coor_dict.keys():
#             ds_var_merged[coor].attrs['standard_name'] = coor_dict[coor]['standard_name']
#             ds_var_merged[coor].attrs['units'] = coor_dict[coor]['units']
#             ds_var_merged[coor].attrs['long_name'] = coor_dict[coor]['long_name']
#         # write output
#         comp = dict(scale_factor=var_dict[varname]['scale_factor'], add_offset=var_dict[varname]['offset'])  
#         encoding = {varname: comp, 'time':{'units': time_unit}} 
#         filename = f'ERA5_CDS_atm_{varname}_{dt.datetime.strftime(date_start_zero, "%Y-%m-%d")}_{dt.datetime.strftime(date_end, "%Y-%m-%d")}.nc'
#         output_path = os.path.join(input_dir.replace('meteo_ERA5','meteo_ERA5_fm'), filename)
#         ds_var_merged.to_netcdf(output_path, encoding=encoding, format='NETCDF4', engine='netcdf4'); ds_var_merged.close()

# if __name__ == "__main__":
#     convert2FM(r'D:\paper_3\data\spectral_nudging_data\xynthia\mslp_hPa', '2010_02')


# # # stack variables into a single DataArray
# #     da = ds.to_array().transpose('variable', 'time', 'latitude', 'longitude')

# #     # update longitude coordinate
# #     da = da.assign_coords(longitude=lons_new)

# #     # create an array of boolean values that indicates which times to drop
# #     times = da.time.values
# #     times_bool = (times > np.datetime64(date_start_spinup)) | ((times > np.datetime64(date_start_transition)) & (times < np.datetime64(date_start_spinup)))

# #     # set values to zero for initialization period
# #     da = da.where(~times_bool, 0)

# #     # remove transition period
# #     da = da.where(~times_bool, drop=True)