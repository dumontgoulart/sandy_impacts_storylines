# -*- coding: utf-8 -*-
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
import geopandas as gpd

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['figure.dpi'] = 144
mpl.rcParams.update({'font.size': 14})

# List of Functions
def load_process_ds(path, start_date, end_date, save_nc=False):
    print(path)
    ds = xr.open_dataset(path)
    if type(ds.time.values[0]) == np.float64:
        ds = process_datetime(ds)
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    if start_date is not None:
        ds = ds.sel(time = slice(start_date, end_date))
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    if save_nc:
        # Get the original filename without extension
        filename = os.path.splitext(ds.encoding['source'])[0]
        # Save the dataset to netCDF format
        ds.to_netcdf(filename + 'time_corrected.nc')
    return ds

def process_ds(ds, start_date, end_date):
    if type(ds.time.values[0]) == np.float64:
        ds = process_datetime(ds)
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds.sel(time = slice(start_date, end_date))
    ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
    ds = ds.sortby(ds.lon)
    return ds


def process_datetime(ds):
    # extract the time values as floats
    times_float = ds.time.values.astype(float)
    # add a small number to avoid rounding errors
    times_float += 1e-12
    # extract the integer part of the time values (i.e., year, month, day)
    times_int = times_float.astype(int).astype(str)
    # extract the fractional part of the time values (i.e., the decimal places)
    times_frac = np.modf(times_float)[0]
    # convert the fractional part to hours
    hours = np.round(times_frac * 24).astype(int)
    # create datetime strings with leading zeros for the hours
    hours_str = np.char.zfill(hours.astype(str), 2)
    # combine the date and time strings
    date_time_str = np.char.add(times_int, hours_str)
    date_time = pd.to_datetime(date_time_str, format='%Y%m%d%H')
    ds = ds.assign_coords(time=date_time)
    return ds

def U10m_totalwind(ds):
    if all(elem in ds.keys() for elem in ['u10m', 'v10m']): 
        ds['u10m'].attrs['units'] = 'm/s'
        ds['v10m'].attrs['units'] = 'm/s'

        ds['U10m'] = np.sqrt(ds['u10m']**2+ds['v10m']**2)
        ds['U10m'].attrs['standard_name'] = 'total_wind_speed'
        ds['U10m'].attrs['units'] = 'm/s'
    else:
        raise ValueError('To calculate the absolute wind speed, it is required the u and v components')

def load_multiple_var(list_vars, year, sn_dir, storm_date_start,storm_date_end, name, calc_U10m = True):
    list_ds_vars_factual = []
    list_ds_vars_counter = []
    list_ds_vars_plus2 = []
    for var_unit_name in list_vars:
        var = var_unit_name.split('_')[0] 

        # Load ensemble members
        ds_factual = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_factual_1_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_factual2 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_factual_2_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_factual3 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_factual_3_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_factual_ens = xr.concat([ds_factual, ds_factual2, ds_factual3], pd.Index(['1','2','3'], name='member'))
        list_ds_vars_factual.append(ds_factual_ens)

        ds_counter = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_counter_1_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_counter2 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_counter_2_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_counter3 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_counter_3_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_counter_ens = xr.concat([ds_counter, ds_counter2, ds_counter3], pd.Index(['1','2','3'], name='member'))
        list_ds_vars_counter.append(ds_counter_ens)

        ds_plus2 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_plus2_1_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_plus22 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_plus2_2_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_plus23 = load_process_ds(sn_dir+var_unit_name+f'/BOT_t_HR255_Nd_SU_1015_plus2_3_{year}_{name}_storm.{var_unit_name}.nc',storm_date_start, storm_date_end)
        ds_plus2_ens = xr.concat([ds_plus2, ds_plus22, ds_plus23], pd.Index(['1','2','3'], name='member'))
        list_ds_vars_plus2.append(ds_plus2_ens)

    ds_fac, ds_count, ds_plus = xr.merge(list_ds_vars_factual), xr.merge(list_ds_vars_counter), xr.merge(list_ds_vars_plus2)

    if calc_U10m:
        U10m_totalwind(ds_fac)
        U10m_totalwind(ds_count)
        U10m_totalwind(ds_plus)

    return ds_fac, ds_count, ds_plus

def locate_storm(ds, storm_id):
    # use IBTracs to locate storms based on their SID code.
    storm_ids = [''.join(name.astype(str)) for name in ds.variables['sid'].values]
    # create a dictionary with the saffit-simpson hurricane scale
    wind_speed_scale = {
    -5: 'Unknown',
    -4: 'Post-tropical',
    -3: 'Miscellaneous disturbances',
    -2: 'Subtropical',
    -1: 'Tropical depression',
    0: 'Tropical storm',
    1: 'Category 1',
    2: 'Category 2',
    3: 'Category 3',
    4: 'Category 4',
    5: 'Category 5'}

    sel_tracks = []
    # filter name
    if storm_id:
        if not isinstance(storm_id, list):
            storm_id = [storm_id]
        for storm in storm_id:
            sel_tracks.append(storm_ids.index(storm))
        sel_tracks = np.array(sel_tracks)
        ds_storm = ds.sel(storm = sel_tracks)
        print(ds_storm.name.values)
        df_wind_speed = ds_storm[['usa_wind','usa_pres','storm_speed','usa_status','usa_sshs']].to_dataframe().dropna()
        df_wind_speed['usa_status'] = df_wind_speed['usa_status'].apply(lambda x: x.decode())
        df_wind_speed['usa_sshs'] = df_wind_speed['usa_sshs'].map(wind_speed_scale)
        df_wind_speed['time'] = pd.to_datetime(df_wind_speed['time']).round('1s')
    return ds_storm, df_wind_speed

# Check for robustness defined by Linda in a two-way approach (upper > lower & lower < upper)
def robustness_test_ensemble(ds_upper, ds_lower):
    '''
    Function to measure at which grid points different datasets overlap with each other.
    If there is no overlap between the two datasets, the difference is considered signifcant,
    which is represented by a grid cell value of 1; if not, it is represented as 0.
    We add a tolerance factor to make sure the values do not get too close. The tolerance
    is defined as a percentage value (0-1).
    '''
    tolerance = 0.001 #(0.1%)
    ds_upper_min = ds_upper.min(dim='member').round(decimals = 4)
    ds_upper_max = ds_upper.max(dim='member').round(decimals = 4)
    ds_lower_min = ds_lower.min(dim="member").round(decimals = 4)
    ds_lower_max = ds_lower.max(dim="member").round(decimals = 4)

    # Now calculate which grid cells the ensemble members are not touching
    ds_dif_pos = ds_upper_min * (1-tolerance) - ds_lower_max
    ds_dif_neg = ds_upper_max * (1+tolerance) - ds_lower_min

    # Build a mask where 1 represents no intersection nor overlapping between members
    ds_sig_mask = xr.where( (ds_dif_pos>ds_upper_min.mean()*tolerance) | (ds_dif_neg< (0-ds_upper_min.mean()*tolerance)), 1, 0) 
    ds_sig_mask = ds_sig_mask.rename(name_dict={var:'robustness'})
    return ds_sig_mask

def legend_without_duplicate_labels(ax, loc = 'center left', bbox_to_anchor=(1, 0.5)):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),loc=loc,bbox_to_anchor=bbox_to_anchor, frameon = False)

# Plot map for single plot and variable
def spatial_ax_map(ds, df_ws, title = None, cmap = 'Blues'):
    central_lon = ds.lon.min() + (ds.lon.max() - ds.lon.min())/2
    central_lat = ds.lat.min() + (ds.lat.max() - ds.lat.min())/2
    ax = plt.axes(projection=ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon.item(), central_latitude=central_lat.item()+10))    # 
    ax.coastlines(resolution='10m') 
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'observed track')
    if cmap:
        ds.plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = cmap, robust=True)
    else:
        ds.plot(ax = ax,transform=ccrs.PlateCarree(), x='lon', y='lat', robust=True)
    plt.title(title)
    plt.show()
    plt.close()

# Set of plots to compare either members of ensemble of GW levels
def spatial_plot_dif_mult(ds1, ds2, ds3, storm_name, name_members = ["1 - 2", "1 - 3", "2 - 3"], col_name = 'members', title = None, robust_test = False, export_figure = True):
    ds_dif1 = ds1 - ds2
    ds_dif2 = ds1 - ds3
    ds_dif3 = ds2 - ds3
    ds_dif_concat = xr.concat([ds_dif1, ds_dif2, ds_dif3], pd.Index(name_members, name=col_name))
    central_lon = ds1.lon.min() + (ds1.lon.max() - ds1.lon.min())/2
    central_lat = ds1.lat.min() + (ds1.lat.max() - ds1.lat.min())/2
    map_proj = ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon.item(), central_latitude=central_lat.item()+10)
    print(ds1.sum().values)
    print(ds2.sum().values)
    print(ds3.sum().values)
    print(ds1.sum().values/ds2.sum().values)
    print(ds1.sum().values/ds3.sum().values)
    print(ds2.sum().values/ds3.sum().values)
    
    p = ds_dif_concat.plot(transform=ccrs.PlateCarree(),x="lon", y="lat", col=col_name, subplot_kws={"projection": map_proj}, robust = True,size=4)   
    for ax in p.axs.flat:
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([ds_dif_concat['lon'].min(), ds_dif_concat['lon'].max(), ds_dif_concat['lat'].min(), ds_dif_concat['lat'].max()])
        ax.set_aspect('equal', 'box')
    plt.suptitle(title)
    if export_figure:
        plt.savefig(f'{data_dir}Figures/sn_precip_plots/{ds1.attrs["standard_name"]}_map_{storm_name}_{storm_date_start}_{storm_date_end}_{title}.png', dpi=200)
    plt.show()
    plt.close()

# More optimal set of plots comparing GW levels
def spatial_plot_dif_mult_GW(ds_factual, ds_counter, ds_plus2, name_members, storm_name, col_name = 'members', title = None, robust_test = False, track = True, export_figure = True):
    ds_dif1 = ds_factual.mean(dim = 'member') - ds_counter.mean(dim = 'member')
    ds_dif2 = ds_plus2.mean(dim = 'member') - ds_factual.mean(dim = 'member')
    ds_dif3 = ds_plus2.mean(dim = 'member') - ds_counter.mean(dim = 'member')
    ds_dif_concat = xr.concat([ds_dif1, ds_dif2, ds_dif3], pd.Index(name_members, name=col_name))
    # Difference between GW levels
    print(ds_factual[var].sum().values/ds_counter[var].sum().values)
    print(ds_plus2[var].sum().values/ds_factual[var].sum().values)
    print(ds_plus2[var].sum().values/ds_counter[var].sum().values)
    
    if robust_test:
        df1_robust = robustness_test_ensemble(ds_factual, ds_counter)
        df2_robust = robustness_test_ensemble(ds_plus2, ds_factual)
        df3_robust = robustness_test_ensemble(ds_plus2, ds_counter)

        df_r_all = xr.concat([df1_robust, df2_robust, df3_robust], pd.Index(name_members, name='version'))
        if 'time' in df_r_all['robustness'].dims:
            df_r_all['robustness'] = df_r_all['robustness'].mean('time')

    central_lon = ds_factual.lon.min() + (ds_factual.lon.max() - ds_factual.lon.min())/2
    central_lat = ds_factual.lat.min() + (ds_factual.lat.max() - ds_factual.lat.min())/2
    map_proj = ccrs.LambertAzimuthalEqualArea(central_longitude=central_lon.item(), central_latitude=central_lat.item()+10)
    p = ds_dif_concat[var].plot(transform=ccrs.PlateCarree(),x="lon", y="lat", col=col_name, subplot_kws={"projection": map_proj}, robust = True, figsize=(12,6))   
    for ax in p.axs.flat:
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if robust_test:
            ax.contourf(ds_dif_concat.lon,ds_dif_concat.lat, df_r_all['robustness'].sel(version=ax.title.get_text().split('= ')[1]), 
                        transform=ccrs.PlateCarree(), colors='none',levels=[0.5,1.5],hatches=[6*'/',0*'/'],)
        if track:
            ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'observed track')
        ax.set_extent([ds_dif_concat['lon'].min(), ds_dif_concat['lon'].max(), ds_dif_concat['lat'].min(), ds_dif_concat['lat'].max()])
        ax.set_aspect('equal', 'box')
    plt.suptitle(title+f' {ds_factual[var].attrs["standard_name"]} map')
    if export_figure:
        plt.savefig(f'{data_dir}Figures/sn_precip_plots/{ds_factual[var].attrs["standard_name"]}_map_{storm_name}_{storm_date_start}_{storm_date_end}_{title}.png', dpi=200)
    # plt.tight_layout()
    plt.show()
    plt.close()

# Plot timeseries for the ensembles
def plot_gw_levels_timeseries(ds_factual, ds_counter, ds_plus2, storm_name, aggregate = None, mode = 'ensemble', export_figure = True):
    # Aggregate can be '1d'
    ds_precip_agg_ens = xr.concat([ds_factual.mean(['lat','lon']), ds_counter.mean(['lat','lon']), ds_plus2.mean(['lat','lon'])], pd.Index(['factual','counter','plus2'], name='gw_level'))
    if aggregate is not None:
        ds_precip_agg_ens = ds_precip_agg_ens.resample(time=aggregate).mean()
    else:
        aggregate = '3hr'
    custom_dict = {'factual': 0, 'counter': 1, 'plus2': 2} 

    counter_agg_ens = ds_precip_agg_ens.to_dataframe().reset_index()
    counter_agg_ens.sort_values(by=['time','gw_level'], key=lambda x: x.map(custom_dict), inplace = True)

    counter_avg = counter_agg_ens.groupby(["time","gw_level"]).mean().reset_index()    
    counter_avg.sort_values(by=['time','gw_level'], key=lambda x: x.map(custom_dict), inplace = True)

    plt.figure(figsize=(6,6)) #plot clusters
    if mode == 'ensemble':
        sns.lineplot(data = counter_agg_ens, x = 'time', y = var, units = "member", hue = 'gw_level',style = 'gw_level', markers=False, estimator=None)
    elif mode == 'average':
        sns.lineplot(data = counter_avg, x = 'time', y = var, hue = 'gw_level',style = 'gw_level', markers=False, estimator=None)
    plt.title(f'{ds_factual[var].attrs["standard_name"]} timeseries for {storm_name} for {aggregate} {mode}')
    if export_figure:
        plt.savefig(f'{data_dir}Figures/sn_precip_plots/{ds_factual[var].attrs["standard_name"]}_timeseries_{storm_name}_{storm_date_start}_{storm_date_end}_{aggregate}{mode}.png', dpi=200)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

dict_storm = {'sandy':{'storm_date_start':'2012-10-22','storm_date_end':'2012-10-30', 'plot_day':'2012-10-25', 'sid':'2012296N14283', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'}, 
              'igor':{'storm_date_start':'2010-09-11','storm_date_end':'2010-09-22', 'plot_day':'2010-09-19', 'sid':'2010251N14337', 'bbox':[-89.472656,7.532249,-30.078125,55], 'target_city':'New York'},
              'earl':{'storm_date_start':'2010-08-29','storm_date_end':'2010-09-04', 'plot_day':'2010-09-04', 'sid':'2010236N12341', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'irene':{'storm_date_start':'2011-08-22','storm_date_end':'2011-08-28', 'plot_day':'2011-08-27', 'sid':'2011233N15301', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'katia':{'storm_date_start':'2011-09-02','storm_date_end':'2011-09-10', 'plot_day':'2011-09-06', 'sid':'2011240N10341', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'ophelia':{'storm_date_start':'2011-09-28','storm_date_end':'2011-10-03', 'plot_day':'2011-10-02', 'sid':'2011263N12323', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'rafael':{'storm_date_start':'2012-10-12','storm_date_end':'2012-10-24', 'plot_day':'2012-10-16', 'sid':'2012287N15297', 'bbox':[-85.473633,7.532249,-30.078125,56], 'target_city':'Santiago de Cuba'},
              'gonzalo':{'storm_date_start':'2014-10-13','storm_date_end':'2014-10-18', 'plot_day':'2014-10-16', 'sid':'2014285N16305', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'Santiago de Cuba'},
              'hayan':{'storm_date_start':'2013-11-04','storm_date_end':'2013-11-09','plot_day':'2013-11-07', 'sid':'2013306N07162', 'bbox':[117.641602,6.271618,131.308594,19.932041],'target_city':'Tacloban'}, 
              'xynthia':{'storm_date_start':'2010-02-27','storm_date_end':'2010-03-01','plot_day':'2010-02-27', 'sid':'xynthia_track', 'bbox':[-45.771484,22.431340,29.707031,64.227957],'target_city':'La Rochelle'},
              'xaver':{'storm_date_start':'2013-12-04','storm_date_end':'2013-12-07','plot_day':'2013-12-05', 'sid':'xaver_track', 'bbox':[-46.982422,22.979488,36.826172,69.886265],'target_city':'Hamburg'}, 
              'megi':{'storm_date_start':'2010-10-12','storm_date_end':'2010-10-24','plot_day':'2010-10-20', 'sid':'2010285N13145', 'bbox':[117.597656,6.664608,133.154297,18.729502],'target_city':'Tacloban'}, 
              'yasi':{'storm_date_start':'2011-01-31','storm_date_end':'2011-02-05','plot_day':'2011-02-02', 'sid':'2011028S13180', 'bbox':[144.316406,-21.779905,161.718750,-10.790141],'target_city':'Townsville'}, 
              'idai':{'storm_date_start':'2019-03-04','storm_date_end':'2019-03-21','plot_day':'2019-03-14','sid':'2019063S17066','bbox':[31.673584,-22.309426,45.483398,-14.392118],'target_city':'Beira'}} 

def preprocess_data(storm_name, clip = False, plots = False, dict_storm = None):
    if dict_storm == None:
        dict_storm = dict_storm
    ### List of variables
    list_vars = ['Ptot_mmh', 'mslp_hPa', 'u10m_m_s', 'v10m_m_s'] #, 
    ### Dates
    year = dict_storm[storm_name]['storm_date_start'].split('-')[0]
    storm_date_start,storm_date_end  = dict_storm[storm_name]['storm_date_start'],dict_storm[storm_name]['storm_date_end'] 
    # Load core setup
    data_dir = 'D:/paper_3/data/'
    # local folder location
    if storm_name in ['igor', 'earl']:
        sn_dir = data_dir+f'spectral_nudging_data/season_2010/'  
    elif storm_name in ['ophelia','katia']:
        sn_dir = data_dir+f'spectral_nudging_data/season_2011/'  
    elif storm_name in ['gonzalo']:
        sn_dir = data_dir+f'spectral_nudging_data/season_2014/'  
    else:
        sn_dir = data_dir+f'spectral_nudging_data/{storm_name}/'  
    
    # Load reference track (IBTracs or others)
    if dict_storm[storm_name]['sid'] == 'xynthia_track':
        df_ws = pd.read_csv(r"d:\paper_3\data\ibtracs\Xynthia_track.csv")
    elif dict_storm[storm_name]['sid'] == 'xaver_track':
        df_ws = pd.read_csv(r"d:\paper_3\data\ibtracs\Xaver_track.csv")
    elif dict_storm[storm_name.split("_")[0]]['sid'] != None:
        ds = xr.open_mfdataset(r'D:\paper_3\data\ibtracs\IBTrACS.since1980.v04r00.nc')
        ds_storm, df_ws = locate_storm(ds, dict_storm[storm_name.split("_")[0]]['sid'])
    df_ws['time'] = df_ws['time'].astype('datetime64[ns]') #
    df_ws = df_ws.loc[(df_ws['time'] >= storm_date_start) & (df_ws['time'] <= storm_date_end+" 23:59:00")]

    # Load data
    if storm_name in ['igor', 'earl']:
        ds_factual_ens, ds_counter_ens, ds_plus2_ens = load_multiple_var(list_vars, year, sn_dir,storm_date_start,storm_date_end, name = 'season_2010')
    elif storm_name in ['ophelia','katia']:
        ds_factual_ens, ds_counter_ens, ds_plus2_ens = load_multiple_var(list_vars, year, sn_dir,storm_date_start,storm_date_end,  name = 'season_2011')
    elif storm_name in ['gonzalo']:
        ds_factual_ens, ds_counter_ens, ds_plus2_ens = load_multiple_var(list_vars, year, sn_dir,storm_date_start,storm_date_end, name = 'season_2014')
    else:
        ds_factual_ens, ds_counter_ens, ds_plus2_ens = load_multiple_var(list_vars, year, sn_dir,storm_date_start,storm_date_end,  name = storm_name)
    # Load time to plot maps
    time_of_plot = dict_storm[storm_name]['plot_day']
    
    # Clip the bbox
    if clip == True:
        ds_factual_ens = ds_factual_ens.sel(lat = slice(dict_storm[storm_name]['bbox'][3], dict_storm[storm_name]['bbox'][1]), lon = slice(dict_storm[storm_name]['bbox'][0], dict_storm[storm_name]['bbox'][2]))
        ds_counter_ens = ds_counter_ens.sel(lat = slice(dict_storm[storm_name]['bbox'][3], dict_storm[storm_name]['bbox'][1]), lon = slice(dict_storm[storm_name]['bbox'][0], dict_storm[storm_name]['bbox'][2]))
        ds_plus2_ens = ds_plus2_ens.sel(lat = slice(dict_storm[storm_name]['bbox'][3], dict_storm[storm_name]['bbox'][1]), lon = slice(dict_storm[storm_name]['bbox'][0], dict_storm[storm_name]['bbox'][2]))

    # Mean of ensembles
    ds_factual_mean = ds_factual_ens.mean(dim = 'member', keep_attrs = True)
    ds_counter_mean = ds_counter_ens.mean(dim = 'member', keep_attrs = True)
    ds_plus2_mean = ds_plus2_ens.mean(dim = 'member', keep_attrs = True)

    if plots == True:
        ds_counter_mean['Ptot'].sum(['lat','lon']).cumsum().plot(label='counter')
        ds_factual_mean['Ptot'].sum(['lat','lon']).cumsum().plot(label='factual')
        ds_plus2_mean['Ptot'].sum(['lat','lon']).cumsum().plot(label='plus2')
        plt.legend()
        plt.show()
        plt.close()
        
        spatial_ax_map(ds_factual_mean['Ptot'].sel(time = time_of_plot).isel(time=0), df_ws, title = "Factual mean", cmap = 'Blues')

    return ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws

def process_historical_data(storm_name, data = 'echam', clip = False, plots = False, dict_storm = None, common_time = False):
    if dict_storm == None:
        dict_storm = dict_storm
    ### List of variables
    list_vars = ['Ptot_mmh', 'mslp_hPa', 'u10m_m_s', 'v10m_m_s'] #, 
    ### Dates
    year = dict_storm[storm_name]['storm_date_start'].split('-')[0]
    storm_date_start,storm_date_end  = dict_storm[storm_name]['storm_date_start'],dict_storm[storm_name]['storm_date_end'] 
    # Load core setup
    data_dir = 'D:/paper_3/data/'
    # local folder location for storm track
    if dict_storm[storm_name.split("_")[0]]['sid'] != None:
        ds = xr.open_mfdataset(r'D:\paper_3\data\ibtracs\IBTrACS.since1980.v04r00.nc')
        ds_storm, df_ws = locate_storm(ds, dict_storm[storm_name.split("_")[0]]['sid'])
        df_ws['time'] = df_ws['time'].astype('datetime64[ns]') #
        df_ws = df_ws.loc[(df_ws['time'] >= storm_date_start) & (df_ws['time'] <= storm_date_end+" 23:59:00")]

    # Load data for storm
    if data == 'echam':
        list_ds_var_hist = []
            # Correct time date for historical data
        time_correction = False
        if time_correction:
            for var_unit_name in list_vars:
                    var = var_unit_name.split('_')[0] 
                    ds_var_hist = load_process_ds(data_dir+f'spectral_nudging_data/historical/CLI_MPI-ESM-XR_t255l95_echam_{storm_name}_{var_unit_name}.nc',"2012-10-01", "2012-12-01", save_nc=True)
        for var_unit_name in list_vars:
            var = var_unit_name.split('_')[0] 
            ds_var_hist = load_process_ds(data_dir+f'spectral_nudging_data/historical/CLI_MPI-ESM-XR_t255l95_echam_{storm_name}_{var_unit_name}.nc',storm_date_start, storm_date_end)
            list_ds_var_hist.append(ds_var_hist)
        # Merge all variables
        ds_hist_merge = xr.merge(list_ds_var_hist)
    elif data == 'era5':
        ds_var_era5 = load_process_ds(data_dir+f'era5/era5_hourly_vars_{storm_name}_single_2012_10.nc',storm_date_start, storm_date_end)
        # rename msl to mslp, tp to ptot, u10 to u10m, and v10 to v10m
        ds_var_era5 = ds_var_era5.rename_vars({'msl':'mslp'})
        ds_var_era5 = ds_var_era5.rename_vars({'tp':'Ptot'})
        ds_var_era5 = ds_var_era5.rename_vars({'u10':'u10m'})
        ds_var_era5 = ds_var_era5.rename_vars({'v10':'v10m'})
        # convert mslp to hPa and Ptot from mm/s to mm/h
        ds_var_era5['mslp'] = ds_var_era5['mslp']/100
        ds_var_era5['Ptot'] = ds_var_era5['Ptot']*1000
        ds_hist_merge = ds_var_era5
    elif data == 'merra2':
        # load all netcdf files with xarray inside the folder:
        ds_merra2_slp = xr.open_mfdataset(r'D:\paper_3\data\merra2\MERRA2_400.inst3_3d_asm_Np*.nc4.nc4', combine='by_coords')['SLP']
        ds_merra2_ptot = xr.open_mfdataset(r'D:\paper_3\data\merra2\MERRA2_400.tavg1_2d_flx_Nx.*.nc4.nc4', combine='by_coords')
        ds_merra2_ptot['time'] = ds_merra2_ptot['time'].dt.floor('H')
        ds_merra2_ptot = ds_merra2_ptot.assign_coords(time=ds_merra2_ptot['time'])
        # Find the common time range across all variables
        common_times = np.intersect1d(ds_merra2_slp['time'].values, ds_merra2_ptot['time'].values)  # Find common time values

        # Clip the dataset to include only the common time range
        ds_merra2_slp = ds_merra2_slp.sel(time=common_times)
        ds_merra2_ptot = ds_merra2_ptot.sel(time=common_times)
        ds_merra2 = xr.merge([ds_merra2_slp, ds_merra2_ptot])
        ds_merra2 = ds_merra2.compute()
        
        # rename msl to mslp, tp to ptot, u10 to u10m, and v10 to v10m
        ds_merra2 = ds_merra2.rename_vars({'SLP':'mslp'})
        ds_merra2 = ds_merra2.rename_vars({'ULML':'u10m'})
        ds_merra2 = ds_merra2.rename_vars({'VLML':'v10m'})
        ds_merra2 = ds_merra2.rename_vars({'PRECTOT':'Ptot'})
        ds_merra2['mslp'] = ds_merra2['mslp']/100
        ds_merra2['Ptot'] = ds_merra2['Ptot']*3600

        
        ds_merra2 = ds_merra2.sortby('lat', ascending=False)
        ds_hist_merge = process_ds(ds_merra2, storm_date_start, storm_date_end)
        


    # create total wind
    U10m_totalwind(ds_hist_merge)

    # Load time to plot maps
    time_of_plot = dict_storm[storm_name]['plot_day']

    if common_time == True:
        # Find common values in time
        common_times = np.intersect1d(ds_hist_merge['time'].values, df_ws['time'].values)  # Find common time values
        ds_hist_merge = ds_hist_merge.sel(time=common_times)  # Select common time values in dataset
    
    # Clip the bbox
    if clip == True:
        ds_hist_merge = ds_hist_merge.sel(lat = slice(dict_storm[storm_name]['bbox'][3], dict_storm[storm_name]['bbox'][1]), lon = slice(dict_storm[storm_name]['bbox'][0], dict_storm[storm_name]['bbox'][2]))

    if plots == True:
        ds_hist_merge['Ptot'].sum(['lat','lon']).cumsum().plot(label='hist')
        plt.legend()
        plt.show()
        plt.close()
        
        spatial_ax_map(ds_hist_merge['Ptot'].sel(time = time_of_plot).isel(time=0), df_ws, title = "Factual mean", cmap = 'Blues')
    
    return ds_hist_merge, df_ws

################################################################################ 
# Main code
################################################################################ 

if __name__ == "__main__":
    ### Choose storm and variable:
    storm_name = 'sandy'

   # ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws = preprocess_data(storm_name = storm_name, clip = True, plots = True, dict_storm = dict_storm)
    
    ds_hist_merge, df_ws_hist = process_historical_data(storm_name = 'sandy',data = 'merra2', clip = True, plots = True, dict_storm = dict_storm)

    ################################################################################
    test = xr.open_dataset()
    spatial_ax_map(ds_factual_mean[var].sel(time = time_of_plot).isel(time=0), title = "Factual mean", cmap = 'RdBu_r')
    # spatial_ax_map(ds_counter_mean[var].sel(time = time_of_plot).isel(time=0), title = "Counter mean")
    # spatial_ax_map(ds_plus2_mean[var].sel(time = time_of_plot).isel(time=0), title = "Plus2 mean")

    # # Differences between members
    # spatial_plot_dif_mult(ds_factual_ens[var].sel(member = '1', time = time_of_plot).isel(time=0), 
    #                       ds_factual_ens[var].sel(member = '2',time = time_of_plot).isel(time=0),
    #                       ds_factual_ens[var].sel(member = '3',time = time_of_plot).isel(time=0), title = 'factual')

    # spatial_plot_dif_mult(ds_counter_ens[var].sel(member = '1',time = time_of_plot).isel(time=0), 
    #                       ds_counter_ens[var].sel(member = '2',time = time_of_plot).isel(time=0),
    #                       ds_counter_ens[var].sel(member = '3',time = time_of_plot).isel(time=0), title = 'counter')

    # spatial_plot_dif_mult(ds_plus2_ens[var].sel(member = '1',time = time_of_plot).isel(time=0), 
    #                       ds_plus2_ens[var].sel(member = '2',time = time_of_plot).isel(time=0),
    #                       ds_plus2_ens[var].sel(member = '3',time = time_of_plot).isel(time=0), title = 'plus2')

    # # difference between gw levels
    # spatial_plot_dif_mult_GW(ds_factual_ens.sel(time = time_of_plot).isel(time=0),
    #                       ds_counter_ens.sel(time = time_of_plot).isel(time=0),
    #                       ds_plus2_ens.sel(time = time_of_plot).isel(time=0), 
    #                       name_members = ["Factual - Counter", "Plus2 - Factual", "Plus2 - Counter"], col_name = 'members', title = 'GW_levels',robust_test = True)

    # # difference between gw levels
    # spatial_plot_dif_mult_GW(ds_factual_ens.sel(time = time_of_plot).resample(time='1d').mean(),
    #                       ds_counter_ens.sel(time = time_of_plot).resample(time='1d').mean(),
    #                       ds_plus2_ens.sel(time = time_of_plot).resample(time='1d').mean(), 
    #                       name_members = ["Factual - Counter", "Plus2 - Factual", "Plus2 - Counter"], col_name = 'members', title = 'GW_levels_1d',robust_test = True)

    # # difference between gw levels
    # spatial_plot_dif_mult_GW(ds_factual_ens.resample(time='7d').mean(),
    #                       ds_counter_ens.resample(time='7d').mean(),
    #                       ds_plus2_ens.resample(time='7d').mean(), 
    #                       name_members = ["Factual - Counter", "Plus2 - Factual", "Plus2 - Counter"], col_name = 'members', title = 'GW_levels_7d',robust_test = True)

    # # difference between gw levels
    # spatial_plot_dif_mult_GW(ds_factual_ens.mean('time', keep_attrs=True),
    #                       ds_counter_ens.mean('time', keep_attrs=True),
    #                       ds_plus2_ens.mean('time', keep_attrs=True), 
    #                       name_members = ["Factual - Counter", "Plus2 - Factual", "Plus2 - Counter"], col_name = 'members', title = 'GW_levels_time_mean',robust_test = True)

  