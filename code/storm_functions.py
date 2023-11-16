# Functions to be used by ExploreBigStorms.ipynb, ExploreBigStorms_GTSM.ipynb and RegridGTSM.ipynb
# Created by Dewi Le Bars, altered by Henrique M. D. Goulart
import numpy as np
import pandas as pd
import xarray as xr

# Function that selects the closest GTSM station to the desired lat and lon
def closest_gtsm_station(gtsm_ds, lon_in, lat_in, x = 'station_x_coordinate', y = 'station_y_coordinate'):
    # use GTSM standard lat/lon format
    if 'stations' in gtsm_ds.coords:
        gtsm_ds = gtsm_ds.drop_vars('stations')

    ds_nearest_location = []
    if type(lon_in) == float:
        dist = ((gtsm_ds[x] - lon_in)**2 + (gtsm_ds[y] - lat_in)**2)
    else:
        for lon, lat in zip(lon_in, lat_in):
            dist = ((gtsm_ds[x] - lon)**2 + (gtsm_ds[y] - lat)**2)

    ds_nearest_location.append(dist.argmin().values.item())
        
    return ds_nearest_location

def closest_notnan_cell(ds, lon_in, lat_in):
    # Remove NANs
    df = ds.to_dataframe().dropna()
    # Calculate dist and find closest nonNAN grid cell
    dist = (df.index.get_level_values('lat') - lat_in)**2 + (df.index.get_level_values('lon') - lon_in)**2
    nearest_coords = dist.argmin()
    # Find lat and lon
    lat_sel = df.iloc[[nearest_coords]].index.get_level_values('lat').item()
    lon_sel = df.iloc[[nearest_coords]].index.get_level_values('lon').item()

    return ds.sel(lat=[lat_sel], lon=[lon_sel])

# Storm function to select individual storms based on their names
def storm_selection(storm):
    # Time and space selection for individual storms
    if storm == 'Xaver':
        reg_box = {'lon_min':4, 'lon_max':9.3, 'lat_min':52, 'lat_max':58}
        year, month, day_s, day_e = 2013, 12, 1, 11
        
    elif storm == 'Xynthia':
        reg_box = {'lon_min':-7.5, 'lon_max':0, 'lat_min':42, 'lat_max':48}
        year, month, day_s, day_e = 2010, '02', 22, 28
    else:
        raise("Storm not accepted")
    return year, month, day_s, day_e, reg_box

def select_gtsm_ds(day_s, day_e, box, dataset = 'new', variable = 'waterlevel', time_res = 'hourly'):
    '''
    Read a GTSM monthly file.
    Correct the longitude values.
    Make a space and time selection.
    Remove nan values from the dataset.
    Useful to plot or to use a input for regridding to regular grid.'''
    
    data_dir = 'D:/paper_3/data/'
    gtsm_ts_dir = data_dir+'surge_tides/'

    range_dates = pd.period_range(day_s, day_e, freq="M")
    list_files = []
    for date in range_dates:
        if dataset == 'old':
            file_str = xr.open_dataset(f'{gtsm_ts_dir}era5-water_level-{date.year}-m{date.month}-v0.0.nc')
        elif dataset == 'new':
            file_str = f'{gtsm_ts_dir}reanalysis_{variable}_{time_res}_{date.year}_{date.month:02}_v1.nc'
        list_files.append(file_str)
    gtsm_ts_ds = xr.open_mfdataset(list_files)           
    
    # Correct the longitude values
    gtsm_ts_ds['station_x_coordinate'] = xr.where(gtsm_ts_ds['station_x_coordinate'] > 180, 
                                                  gtsm_ts_ds['station_x_coordinate'] - 360,
                                                  gtsm_ts_ds['station_x_coordinate'])

    stations_sel = np.where((gtsm_ts_ds.station_x_coordinate < box['lon_max']) & 
                            (gtsm_ts_ds.station_x_coordinate > box['lon_min']) &
                            (gtsm_ts_ds.station_y_coordinate < box['lat_max']) &
                            (gtsm_ts_ds.station_y_coordinate > box['lat_min']))[0]

    gtsm_ts_ds_sel = gtsm_ts_ds.isel(stations=stations_sel)    
    gtsm_ts_ds_sel = gtsm_ts_ds_sel.sel(time = slice(day_s, day_e))

    gtsm_ts_ds_sel.station_x_coordinate.attrs = {'units': 'degrees_east', 
                                                 'long_name': 'longitude', 
                                                 'short_name': 'longitude'}
    # Remove nan values
    stations_sel = np.where(~np.isnan(gtsm_ts_ds_sel.isel(time=0)[list(gtsm_ts_ds_sel.keys())[0]]))[0]
    gtsm_ts_ds_sel = gtsm_ts_ds_sel.isel(stations=stations_sel)
    
    return gtsm_ts_ds_sel

def select_ds_rp(box, rp = '10', variable = 'waterlevel', ref = None):
    '''
    Read a GTSM-RP (no time) file.
    Correct the longitude values.
    Make a space selection.
    '''
    gtsm_ts_ds = xr.open_dataset(f'../data/surge_tides/rp/reanalysis_{variable}_actual-value_1985-2014_rp{rp}_best-fit_v1.nc')
    
    # Correct the longitude values
    gtsm_ts_ds['station_x_coordinate'] = xr.where(gtsm_ts_ds['station_x_coordinate'] > 180, 
                                                  gtsm_ts_ds['station_x_coordinate'] - 360,
                                                  gtsm_ts_ds['station_x_coordinate'])

    stations_sel = np.where((gtsm_ts_ds.station_x_coordinate < box['lon_max']) & 
                            (gtsm_ts_ds.station_x_coordinate > box['lon_min']) &
                            (gtsm_ts_ds.station_y_coordinate < box['lat_max']) &
                            (gtsm_ts_ds.station_y_coordinate > box['lat_min']))[0]

    gtsm_ts_ds_sel = gtsm_ts_ds.isel(stations=stations_sel)    
    gtsm_ts_ds_sel.station_x_coordinate.attrs = {'units': 'degrees_east', 
                                                 'long_name': 'longitude', 
                                                 'short_name': 'longitude'}
    if ref is not None:
        gtsm_ts_ds_sel = gtsm_ts_ds_sel.where(ref > -100)
    
    return gtsm_ts_ds_sel

# load ERA5 precipitation data
def open_era5_data(full_address = None, year = None, month = None, extra = '', coords = None, start = None, end = None):
    '''
    Load precipitation data based on ERA5 reanalysis for precipitation
    '''
    data_dir = 'D:/paper_3/data/'
    era5_dir = data_dir+'era5/' 
    if full_address is not None:
        ds_era5 = xr.open_dataset(f'{era5_dir}{full_address}.nc')
    elif (year != None) and (month != None):    
        ds_era5 = xr.open_dataset(f'{era5_dir}era5_hourly{extra}_{year}_{month}.nc')
    else:
        raise ValueError('Specify either a "year" and "month" or a full address of the file inside the ERA5 folder.')
    # Convert latitude to lat
    if 'latitude' or 'longitude' in ds_era5.coords:
        ds_era5 = ds_era5.rename({'latitude': 'lat', 'longitude': 'lon'})
    if coords is not None:
        ds_era5 = ds_era5.sel(lat = slice(coords['lat_max'],coords['lat_min']), lon = slice(coords['lon_min'],coords['lon_max']))
    if (start is not None) and (end is not None):
        ds_era5 = ds_era5.sel(time = slice(start,end))
    return ds_era5