# -*- coding: utf-8 -*-
"""
To activate virtual environment: ctrl+shift+p >> "Select Interpreter" >> "hydromt"

# important: for comparisons with RP, the data ERA5 data should be at 0.25 resolution. 
# If for impact estimation, then it is better at higher resolution, 0.1.

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

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# functions
# Plot timeseries of precipitation averaged over region:
def plot_timeseries(ds, export_figure = False, city = '', daily = True, rp_ref = None, rp = 'multiple'):
    plt.figure(figsize=(8,8)) #plot clusters
    
    if city is not '':
        city_coords = storm_df[storm_df['cities']==city]
        ds = sf.closest_notnan_cell(ds, lon_in = city_coords['lon'].values, lat_in = city_coords['lat'].values)

    if daily == True:
        ds_daily = ds.resample(time='1d').sum()
        ds_daily.mean(['lat','lon']).plot(label = 'Daily')        
        precip_max = ds_daily.mean(['lat','lon']).max().values
        plt.ylabel('Precip. (m/day)')

        if (rp_ref is not None) and (np.isnan(rp_ref[f'r10yrrp'].sel(lat=ds['lat'].values, lon=ds['lon'].values, method="nearest").mean().values) is not True):
            rp_ref_city = sf.closest_notnan_cell(rp_ref, lon_in = city_coords['lon'].values, lat_in = city_coords['lat'].values)
            if rp != "multiple":
                rp_values = rp_ref_city[f'r{rp}yrrp'].mean(['lat','lon']).values
                print(f'The maximum precipitation value is {precip_max*1000/rp_values} times larger than the {rp}-RP')
                plt.axhline(y = rp_values/1000, linestyle='dashed', color = 'black', label = f'{rp}-RP')
            elif rp == "multiple":
                rp_10 = rp_ref_city[f'r10yrrp'].mean(['lat','lon']).values
                rp_50 = rp_ref_city[f'r50yrrp'].mean(['lat','lon']).values
                rp_100 = rp_ref_city[f'r100yrrp'].mean(['lat','lon']).values
                plt.axhline(y = rp_10/1000, linestyle='dashed', color = 'yellow', label = '10-RP')
                plt.axhline(y = rp_50/1000, linestyle='dashed', color = 'orange', label = '50-RP')
                plt.axhline(y = rp_100/1000, linestyle='dashed', color = 'red', label = '100-RP')
                print(f'The maximum precipitation value is {precip_max*1000/rp_50} times larger than the 50-RP')
    else:
        ds.mean(['lat','lon']).plot(label = 'hourly')
        plt.ylabel('Precip. (m/hour)')

    plt.title(f'Precipitation time series {city}')
    plt.legend()
    plt.tight_layout()
    if export_figure:
        plt.savefig(f'{data_dir}Figures/precipitation_figures/precipitation_{ds.time.dt.year[0].values.item()}_{city}_TimeSeries.pdf', dpi=150)
    plt.show()
    
# Plot maps
def plot_map(da, export_figure = False, coords = None, colormap = None, city = None):

    if 'time' in da.coords:
        if city is not None:
            city_coords = storm_df[storm_df['cities']==city]
            da_city = sf.closest_notnan_cell(da, lon_in = city_coords['lon'].values, lat_in = city_coords['lat'].values)

            time_max = da_city.sel(time = da_city.idxmax(dim = 'time')).time.values[0]
            da = da.sel(time = time_max)
        else:
            da = da.mean('time')
    plt.figure(figsize=(12,10)) #plot clusters
    ax=plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=city_coords['lon'].values.item(), central_latitude=city_coords['lat'].values.item()))
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, alpha=0.5)
    
    if colormap is None:
        da.plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', robust=True)
    elif colormap is not None:
        da.plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = colormap, robust=True)
    
    if city is not None:
        ax.scatter(da_city['lon'].values, da_city['lat'].values, 300, c='yellow', 
        edgecolors='black', marker='*', transform=ccrs.PlateCarree())

    if coords is not None:
        ax.set_extent([coords['lon_min'], coords['lon_max'], coords['lat_min'], coords['lat_max']])
    if export_figure:
        plt.savefig(f'{data_dir}Figures/precipitation_figures/precipitation_{da.time.dt.year[0].values.item()}_{city}_Map.pdf', dpi=150)
    plt.show()

def edit_data_catalog(catalog_path, file_path, section_name):
    import yaml
    with open(catalog_path) as f:
        list_doc = yaml.safe_load(f)

    list_doc[section_name]['path'] = file_path

    with open(catalog_path, "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)

################################################################################
# Main code
################################################################################
# Load core setup
data_dir = 'D:/paper_3/data/'
era5_dir = data_dir+'era5/' 
# Load cities for the storms:
storm_df = pd.read_csv(data_dir+'list_cities_storm.csv', index_col=0)
# Load ERA5 return periods to have context on the precipitation levels obtained above
ds_precip_rp_era5 = xr.open_mfdataset(('../data/extreme_precip_rp/precipitation-at-fixed-return-period_europe_era5_*.nc')).rename({'latitude': 'lat', 'longitude': 'lon'})
ds_precip_rp_eobs = xr.open_mfdataset(('../data/extreme_precip_rp/precipitation-at-fixed-return-period_europe_e-obs_*.nc')).rename({'latitude': 'lat', 'longitude': 'lon'})
ds_precip_rp_ecad = xr.open_mfdataset(('../data/extreme_precip_rp/precipitation-at-fixed-return-period_europe_ecad_*.nc')).rename({'latitude': 'lat', 'longitude': 'lon'})


################################################################################
# Load ERA5 precipitation data for Xaver
# Storm info
xaver_coords = {'lon_min':-4, 'lon_max':12, 'lat_min':50, 'lat_max':58}
storm_date_start, storm_date_end ='2013-12-02', '2013-12-07'
# Load data and organise into storm details
ds_precip_xaver = sf.open_era5_data(full_address="era5_hourly_ukstorm_land_2013_2014_12_2", 
                  coords = xaver_coords, start = storm_date_start, end = storm_date_end, extra = '_land')
# SAVE
ds_precip_xaver.to_netcdf(f'{data_dir}era5_storm_outputs/era5_01x01_xaver_{storm_date_start}_{storm_date_end}.nc')

# Plot timeseries city
plot_timeseries(ds_precip_xaver['tp'], city = 'Cuxhaven', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xaver['tp'], city = 'Hamburg', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xaver['tp'], city = 'Aalborg', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xaver['tp'], city = 'Edinburgh', rp_ref = ds_precip_rp_ecad, export_figure=True)
# plot_timeseries(ds_precip_xaver['tp'], city = 'Dover', rp_ref = ds_precip_rp_ecad, export_figure=True)

# Plot maps with cities indicated by a star
plot_map(ds_precip_xaver['tp'], colormap = 'Blues', city = 'Cuxhaven', export_figure=True)
plot_map(ds_precip_xaver['tp'], colormap = 'Blues', city = 'Hamburg', export_figure=True)
plot_map(ds_precip_xaver['tp'], colormap = 'Blues', city = 'Aalborg', export_figure=True)
plot_map(ds_precip_xaver['tp'], colormap = 'Blues', city = 'Edinburgh', export_figure=True)
# plot_map(ds_precip_xaver['tp'], colormap = 'Blues', city = 'Dover', export_figure=True) # Too low precipitation values



##################################################
# Load ERA5 precipitation data for Xynthia
storm_date_start,storm_date_end ='2010-02-22', '2010-03-02'
ds_precip_xynthia = sf.open_era5_data(full_address="era5_hourly_ukstorm_land_2010_2010_2_3", 
                    start = storm_date_start, end = storm_date_end, extra = '_land') 

ds_precip_xynthia.to_netcdf(f'{data_dir}era5_storm_outputs/era5_01x01_xynthia_{storm_date_start}_{storm_date_end}.nc')

# Plot timeseries city
plot_timeseries(ds_precip_xynthia['tp'], city = 'La Rochelle', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xynthia['tp'], city = 'Vigo', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xynthia['tp'], city = 'La Coruna', rp_ref = ds_precip_rp_ecad, export_figure=True)
plot_timeseries(ds_precip_xynthia['tp'], city = 'Rotterdam', rp_ref = ds_precip_rp_ecad, export_figure=True)

# Plot maps with cities indicated by a star
plot_map(ds_precip_xynthia['tp'], colormap = 'Blues', city = 'La Rochelle', export_figure=True)
plot_map(ds_precip_xynthia['tp'], colormap = 'Blues', city = 'Vigo', export_figure=True)
plot_map(ds_precip_xynthia['tp'], colormap = 'Blues', city = 'La Coruna', export_figure=True)
plot_map(ds_precip_xynthia['tp'], colormap = 'Blues', city = 'Rotterdam', export_figure=True)


##################################################
# Load ERA5 precipitation data for Sandy - NY
storm_date_start,storm_date_end ='2012-10-22', '2012-10-30'
ds_precip_sandy = sf.open_era5_data(full_address="era5_hourly_sandy_single_2012_10", 
                    start = storm_date_start, end = storm_date_end, extra = '_land') 

ds_precip_sandy.to_netcdf(f'{data_dir}era5_storm_outputs/era5_01x01_sandy_{storm_date_start}_{storm_date_end}.nc')

edit_data_catalog("d:\paper_3\data\sfincs_ini\data_gtsm_test.yml", f'{data_dir}era5_storm_outputs/era5_01x01_sandy_{storm_date_start}_{storm_date_end}.nc', f"era5_hourly_sandy")
##################################################
# Plot timeseries city
plot_timeseries(ds_precip_sandy['tp'], city = 'New York', rp_ref = ds_precip_rp_ecad, export_figure=True)

plot_map(ds_precip_sandy['tp'], colormap = 'Blues', city = 'New York', export_figure=True)





#####
# compare ERA5 precipitations values with spectral nuding runs for Sandy #bbox -77.519531,38.307181,-72.246094,41.738528
# Load ERA5 precipitation data for Sandy - NY
storm_date_start,storm_date_end ='2012-10-22', '2012-10-30'
ds_precip_sandy = sf.open_era5_data(full_address="era5_hourly_sandy_single_2012_10") #.sel(lat = slice(41.738528,38.307181), lon = slice(-77.519531, -72.246094))
ds_precip_sandy['tp'] = ds_precip_sandy['tp']*1000 # convert to mm/h
# load spectral nuding data
data_dir = "d:/paper_3/data/spectral_nudging_data/sandy/Ptot_mm3h/"
ds_precip_sandy_sn_counter1 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_counter_1_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_counter2 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_counter_2_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_counter3 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_counter_3_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_factual1 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_factual_1_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_factual2 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_factual_2_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_factual3 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_factual_3_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_plus2_1 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_plus2_1_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_plus2_2 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_plus2_2_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))
ds_precip_sandy_sn_plus2_3 = xr.open_dataset(f'{data_dir}BOT_t_HR255_Nd_SU_1015_plus2_3_2012_sandy_storm.Ptot_mmh.nc').sel(time = slice(ds_precip_sandy.time[0], ds_precip_sandy.time[-1]), lat = slice(ds_precip_sandy.lat[0], ds_precip_sandy.lat[-1]), lon = slice(ds_precip_sandy.lon[0], ds_precip_sandy.lon[-1]))



# plot timeseries for mean precipitation values per time
plt.figure(figsize=(12,10))
ax = plt.axes()
ds_precip_sandy['tp'].mean(['lat','lon']).plot(ax=ax,linestyle = 'dashed', label='ERA5')
ds_precip_sandy_sn_counter1.Ptot.mean(['lat','lon']).plot(ax=ax, label='counter1')
ds_precip_sandy_sn_counter2.Ptot.mean(['lat','lon']).plot(ax=ax, label='counter2')
ds_precip_sandy_sn_counter3.Ptot.mean(['lat','lon']).plot(ax=ax, label='counter3')
ds_precip_sandy_sn_factual1.Ptot.mean(['lat','lon']).plot(ax=ax, label='factual1')
ds_precip_sandy_sn_factual2.Ptot.mean(['lat','lon']).plot(ax=ax, label='factual2')
ds_precip_sandy_sn_factual3.Ptot.mean(['lat','lon']).plot(ax=ax, label='factual3')
ds_precip_sandy_sn_plus2_1.Ptot.mean(['lat','lon']).plot(ax=ax, label='plus2_1')
ds_precip_sandy_sn_plus2_2.Ptot.mean(['lat','lon']).plot(ax=ax, label='plus2_2')
ds_precip_sandy_sn_plus2_3.Ptot.mean(['lat','lon']).plot(ax=ax, label='plus2_3')
plt.legend()
plt.show()

ds_precip_sandy_sn_counter1.Ptot.mean()
ds_precip_sandy_sn_counter2.Ptot.mean()
ds_precip_sandy_sn_counter3.Ptot.mean()
ds_precip_sandy['tp'].mean()


plt.figure(figsize=(12,10)) #plot clusters
ax=plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=ds_precip_sandy['lon'].mean().values.item(), central_latitude=ds_precip_sandy['lat'].mean().values.item()))

ds_precip_sandy_sn_counter1['Ptot'].isel(time=0).plot(ax=ax,transform=ccrs.PlateCarree(), x='lon', y='lat', robust=True)
ax.coastlines(resolution='10m') 
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
plt.show()

























##################################################
##################################################
# EXPLORE
# Where does the maximum precipitation value during the Xaver storm occur?
precip_max = ds_precip_xaver['tp'].sel(time = slice(storm_date_start,storm_date_end)).max().values
ds_precip_max = ds_precip_xaver['tp'].sel(time = slice(storm_date_start,storm_date_end)).where(ds_precip_xaver['tp'] >= precip_max, drop=True)

# Plot map of precipitation at maximum aggregate value
plot_map(ds_precip_xaver['tp'].sel(time=ds_precip_xaver['tp'].sel(time = slice(storm_date_start,storm_date_end)).mean(['lat','lon']).idxmax()), colormap = 'Blues', coords = None)
# Plot map of precipitation at maximum local value
plot_map(ds_precip_xaver['tp'].sel(time=ds_precip_max.time), colormap = 'Blues', coords = None)


##################################################
# Load ERA5 precipitation data for Xynthia
ds_precip_xynthia = sf.open_era5_data(2010, '02')

# Plot timeseries
plot_timeseries(ds_precip_xynthia['tp'])

# Where does the maximum precipitation value during the xynthia storm occur?
storm_date_start ='2010-02-27'
storm_date_end ='2010-02-28'
precip_max = ds_precip_xynthia['tp'].sel(time = slice(storm_date_start,storm_date_end)).max().values
ds_precip_max = ds_precip_xynthia['tp'].sel(time = slice(storm_date_start,storm_date_end)).where(ds_precip_xynthia['tp'] >= precip_max, drop=True)

# Plot map of precipitation at maximum aggregate value
coords_test = {'lon_min':-10.1953125, 'lon_max':13, 'lat_min':40, 'lat_max':59.1759282}
plot_map(ds_precip_xynthia['tp'].sel(time=ds_precip_xynthia['tp'].sel(time =slice(storm_date_start,storm_date_end)).mean(['lat','lon']).idxmax()), colormap = 'Blues',coords = coords_test)

# Sequence of plots over the day of the storm
p = ds_precip_xynthia['tp'].resample(time='1d').sum().plot(x="lon", y="lat", cmap = 'Blues', col="time", col_wrap=4, transform=ccrs.PlateCarree(),subplot_kws={"projection": ccrs.PlateCarree()})
for ax in p.axes.flat:
    ax.coastlines()
plt.show()

# Select city: Cuxhaven
city_sel = 'La Rochelle'
city_info = storm_df.where(storm_df['cities']==city_sel).dropna()
ds_precip_xynthia_city = ds_precip_xynthia.sel(lat=city_info['lat'].values, lon=city_info['lon'].values, method="nearest")
ds_precip_xynthia_larochelle = ds_precip_xynthia.sel(lat = [53.55], lon = [9.99],  method = 'nearest')

# Plot map of precipitation at maximum local value
plot_map(ds_precip_xynthia['tp'].sel(time=ds_precip_max.time), colormap = 'Blues', city_coords = city_info)

# Plot timeseries city
plot_timeseries(ds_precip_xynthia_city['tp'], city = city_sel, rp_ref=ds_precip_rp_ecad)
# Plot at La rochelle
plot_timeseries(ds_precip_xynthia['tp'].sel(lat=[46.20], lon=[1.15], method="nearest"), city = 'La Rochelle',rp_ref=ds_precip_rp_ecad)

# Plot the timeseries for the location with the highest precipitation peak
plot_timeseries(ds_precip_xynthia['tp'].sel(lat=ds_precip_max['lat'].values, lon=ds_precip_max['lon'].values, method="nearest"),
                rp_ref=ds_precip_rp_ecad)
# Plot at La Coruna
plot_timeseries(ds_precip_xynthia['tp'].sel(lat=[43.2], lon=[-8.35], method="nearest"), city = 'La Coruna',
                rp_ref=ds_precip_rp_ecad)
# Plot at Vigo
plot_timeseries(ds_precip_xynthia['tp'].sel(lat=[42.12], lon=[-8.81], method="nearest"), city = 'Vigo', rp_ref=ds_precip_rp_ecad)

################################################################################
# UK Winter storms
# Set parameters
uk_storm_coords = {'lon_min':-10.283203, 'lon_max':11.381836, 'lat_min':49.181703, 'lat_max': 58.153403}
# storm dates
storm_date_start ='2013-12-05'
storm_date_end ='2014-02-25'

# Load ERA5 precipitation data for UK winter stomrs 2013-2014
ds_precip_ukstorm = sf.open_era5_data(full_address="era5_hourly_ukstorm_land_2013_2014_12_2", start = storm_date_start, end = storm_date_end, coords = uk_storm_coords)

# Plot timeseries
plot_timeseries(ds_precip_ukstorm['tp'])

# Where does the maximum precipitation value during the Xaver storm occur?
precip_max = ds_precip_ukstorm['tp'].max().values
ds_precip_max = ds_precip_ukstorm['tp'].where(ds_precip_ukstorm['tp'] >= precip_max, drop=True)

# Plot map of precipitation at maximum aggregate value
plot_map(ds_precip_ukstorm['tp'].sel(time=ds_precip_ukstorm['tp'].mean(['lat','lon']).idxmax()), colormap = 'Blues', coords = None)
# Plot map of precipitation at maximum local value
plot_map(ds_precip_ukstorm['tp'].sel(time=ds_precip_max.time), colormap = 'Blues', coords = None)

# Plot at Bristol
test_city_name, test_lat, test_lon = 'Hamburg', 53.580761, 10.064936

plot_map(ds_precip_ukstorm['tp'].sel(time=ds_precip_max.time), colormap = 'Blues', city_coords ={'lat':test_lat,'lon':test_lon}, coords =uk_storm_coords )
plot_map(ds_precip_rp_ecad['r100yrrp'], colormap = 'Blues', city_coords ={'lat':test_lat,'lon':test_lon}, coords =uk_storm_coords)
plot_timeseries(ds_precip_ukstorm['tp'].sel(lat=[test_lat], lon=[test_lon], method="nearest"), city = test_city_name, rp_ref=ds_precip_rp_ecad)


##################################################
# Load ERA5 precipitation data for Dutch Storm 2012 Groningen
ds_precip_groningen_event = sf.open_era5_data(2012, 1)

# Plot timeseries
plot_timeseries(ds_precip_groningen_event['tp'], daily = True)

# Plot timeseries at groningen
plot_timeseries(ds_precip_groningen_event['tp'].sel(lat=[53.22], lon=[6.4], method="nearest"), city = 'Groningen',rp_ref=ds_precip_rp_ecad)
plot_map(ds_precip_groningen_event['tp'].resample(time='1d').sum().sel(time='2012-01-05'), colormap = 'Blues', city_coords = ds_precip_groningen_event['tp'].sel(lat=[53.22], lon=[6.56], method="nearest"))
plot_map(ds_precip_rp_ecad['r50yrrp'], colormap = 'Blues', city_coords = ds_precip_groningen_event['tp'].sel(lat=[53.22], lon=[6.4], method="nearest"))

##################################################
# plot mpa of RP
plot_map(ds_precip_rp_ecad['r50yrrp'])
plot_map(ds_precip_rp_ecad['r100yrrp'])
