import os
os.chdir('D:/paper_3/code')
# local libraries
import sn_storm_preprocessing as prep
import sn_storm_tracker_composite as trac

os.chdir('D:/paper_3/data/spectral_nudging_data/SSTs/')
import xarray as xr 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import geopandas as gpd



def plot_sst(ds_sst_counter, ds_sst_factual, ds_ss_plus2, df_ws, name):
    # mean of all sst
    sst_counter = ds_sst_counter['sst'].mean(dim='time')
    sst_factual = ds_sst_factual['sst'].mean(dim='time')
    sst_plus2 = ds_ss_plus2['sst'].mean(dim='time')

    # isolines
    levels = [-2, -1, 0, 1, 2]  # temperature values for the isolines

    # obtain difference between the scenarios above
    sst_diff = sst_factual - sst_counter
    sst_diff_plus2 = sst_plus2 - sst_counter
    sst_diff_plus2_factual = sst_plus2 - sst_factual

    # plot the difference between factual and counterfactual
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-100, 20, 0, 80], crs=ccrs.PlateCarree())
    ax.coastlines()
    # ax.add_feature(cfeature.LAND, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)
    # ax.gridlines()
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    sst_diff.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2, vmax=2)
    # Add temperature isolines
    ax.contour(sst_diff.lon, sst_diff.lat, sst_diff, levels=levels, colors='grey')

    ax.set_title(f'SST diff. between factual and counter for {ds_sst_counter["time.year"][0].values}')
    plt.savefig(f'D:/paper_3/data/figures/ssts/sst_diff_factual_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()


    # now do the same but for factual and plus2
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-100, 20, 0, 80], crs=ccrs.PlateCarree())
    ax.coastlines()
    # ax.add_feature(cfeature.LAND, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)
    # ax.gridlines()
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    sst_diff_plus2.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2, vmax=2)
    # Add temperature isolines
    ax.contour(sst_diff.lon, sst_diff.lat, sst_diff_plus2, levels=levels, colors='grey')
    ax.set_title(f'SST diff. between plus2 and counter for {ds_sst_counter["time.year"][0].values}')
    plt.savefig(f'D:/paper_3/data/figures/ssts/sst_diff_plus2_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()


    # now do the same but for factual and plus2
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([-100, 20, 0, 80], crs=ccrs.PlateCarree())
    ax.coastlines()
    # ax.add_feature(cfeature.LAND, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.OCEAN, zorder=100, edgecolor='black')
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)
    # ax.gridlines()
    ax.plot(df_ws['lon'], df_ws['lat'],transform=ccrs.PlateCarree(), linestyle = 'dashed', color = 'k', label = 'obs. track')

    sst_diff_plus2_factual.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-2, vmax=2)
    # Add temperature isolines
    ax.contour(sst_diff.lon, sst_diff.lat, sst_diff_plus2_factual, levels=levels, colors='grey')
    ax.set_title(f'SST diff. between plus2 and factual for {ds_sst_counter["time.year"][0].values}')
    plt.savefig(f'D:/paper_3/data/figures/ssts/sst_diff_plus2_factual{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

######################################################################################################################
# load .nc files using xarray for Xynthia

dict_storm = {'sandy':{'storm_date_start':'2012-10-22','storm_date_end':'2012-10-30', 'plot_day':'2012-10-25', 'sid':'2012296N14283', 'bbox':[-89.472656,7.532249,-30.078125,56], 'target_city':'New York'},
              'xynthia':{'storm_date_start':'2010-02-26','storm_date_end':'2010-02-28','plot_day':'2010-02-27', 'sid':'xynthia_track', 'bbox':[-45.771484,22.431340,29.707031,64.227957],'target_city':'La Rochelle'},}

storm_name = 'sandy' # sandy xaver xynthia

# Define box size for storm tracking
if (storm_name == 'xynthia') or (storm_name == 'xaver'):
    large_box = 4
    small_box = 2
else:
    large_box = 7
    small_box = 4

ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws = prep.preprocess_data(storm_name = storm_name, clip = True, plots = False, dict_storm = dict_storm)

# load .nc files using xarray for Sandy
ds_sst_counter_sandy = xr.open_dataset('T255_counter_ncep1sst_2012.nc').sel(time=slice('2012-10-01', '2012-11-02'))
ds_sst_factual_sandy = xr.open_dataset('T255_ncep1sst_2012.nc').sel(time=slice('2012-10-01', '2012-11-02'))
ds_ss_plus2_sandy = xr.open_dataset('T255_plus2_ncep1sst_2012_W.nc').sel(time=slice('2012-10-01', '2012-11-02'))

ds_sst_counter_sandy = ds_sst_counter_sandy.assign_coords(lon=(((ds_sst_counter_sandy.lon + 180) % 360) - 180)).sortby('lon')
ds_sst_factual_sandy = ds_sst_factual_sandy.assign_coords(lon=(((ds_sst_factual_sandy.lon + 180) % 360) - 180)).sortby('lon')
ds_ss_plus2_sandy = ds_ss_plus2_sandy.assign_coords(lon=(((ds_ss_plus2_sandy.lon + 180) % 360) - 180)).sortby('lon')
# Resample the dataset to 3-hourly frequency
ds_sst_counter_sandy_resample = ds_sst_counter_sandy.reindex(time=ds_factual_ens.time, method='nearest')
ds_sst_factual_sandy_resample = ds_sst_factual_sandy.reindex(time=ds_factual_ens.time, method='nearest')
ds_sst_plus2_sandy_resample = ds_ss_plus2_sandy.reindex(time=ds_factual_ens.time, method='nearest')

# Plot the SSTs
plot_sst(ds_sst_counter_sandy, ds_sst_factual_sandy, ds_ss_plus2_sandy, df_ws, name = 'sandy')

# Define composites for each GW level
ds_rel_counter = trac.storm_composite(ds_sst_counter_sandy_resample, df_center = df_ws.set_index('time'), radius = small_box)
ds_rel_factual = trac.storm_composite(ds_sst_factual_sandy_resample, df_center = df_ws.set_index('time'), radius = small_box)
ds_rel_plus2 = trac.storm_composite(ds_sst_plus2_sandy_resample, df_center = df_ws.set_index('time'), radius = small_box)

fig_mean_sst_mean = trac.scenarios_timeseries_plot(ds_rel_counter, ds_rel_factual, ds_rel_plus2, 'sst', storm_name, mode = 'mean', graph = 'normal')

delta_fc = ds_rel_factual - ds_rel_counter
delta_pf = ds_rel_plus2 - ds_rel_factual
delta_pc = ds_rel_plus2 - ds_rel_counter

delta_fc.sst.mean(['lat','lon']).plot(label = 'factual - counter')
delta_pf.sst.mean(['lat','lon']).plot(label = 'plus2 - factual')
delta_pc.sst.mean(['lat','lon']).plot(label = 'plus2 - counter')
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.ylim(-2,2)
plt.title('Sandy')
plt.tight_layout()
plt.show()

######################################################################################################################

storm_name = 'xynthia' # sandy xaver xynthia

# Define box size for storm tracking
if (storm_name == 'xynthia') or (storm_name == 'xaver'):
    large_box = 4
    small_box = 2
else:
    large_box = 7
    small_box = 4

ds_factual_ens, ds_counter_ens, ds_plus2_ens, ds_factual_mean, ds_counter_mean, ds_plus2_mean, df_ws = prep.preprocess_data(storm_name = storm_name, clip = True, plots = False, dict_storm = dict_storm)

# load .nc files using xarray for Xynthia
ds_sst_counter_xynthia = xr.open_dataset('T255_counter_ncep1sst_2010.nc').sel(time=slice('2010-02-01', '2010-03-01'))
ds_sst_factual_xynthia = xr.open_dataset('T255_ncep1sst_2010.nc').sel(time=slice('2010-02-01', '2010-03-01'))
ds_sst_plus2_xynthia = xr.open_dataset('T255_plus2_ncep1sst_2010_W.nc').sel(time=slice('2010-02-01', '2010-03-01'))

ds_sst_counter_xynthia = ds_sst_counter_xynthia.assign_coords(lon=(((ds_sst_counter_xynthia.lon + 180) % 360) - 180)).sortby('lon')
ds_sst_factual_xynthia = ds_sst_factual_xynthia.assign_coords(lon=(((ds_sst_factual_xynthia.lon + 180) % 360) - 180)).sortby('lon')
ds_sst_plus2_xynthia = ds_sst_plus2_xynthia.assign_coords(lon=(((ds_sst_plus2_xynthia.lon + 180) % 360) - 180)).sortby('lon')
# Resample the dataset to 3-hourly frequency
ds_sst_counter_xynthia_resample = ds_sst_counter_xynthia.reindex(time=ds_factual_ens.time, method='nearest')
ds_sst_factual_xynthia_resample = ds_sst_factual_xynthia.reindex(time=ds_factual_ens.time, method='nearest')
ds_sst_plus2_xynthia_resample = ds_sst_plus2_xynthia.reindex(time=ds_factual_ens.time, method='nearest')

# Plot the SSTs
plot_sst(ds_sst_counter_xynthia, ds_sst_factual_xynthia, ds_sst_plus2_xynthia, df_ws, name = 'xynthia')

# Define composites for each GW level
ds_rel_counter = trac.storm_composite(ds_sst_counter_xynthia_resample, df_center = df_ws.set_index('time'), radius = small_box)
ds_rel_factual = trac.storm_composite(ds_sst_factual_xynthia_resample, df_center = df_ws.set_index('time'), radius = small_box)
ds_rel_plus2 = trac.storm_composite(ds_sst_plus2_xynthia_resample, df_center = df_ws.set_index('time'), radius = small_box)

fig_mean_sst_mean = trac.scenarios_timeseries_plot(ds_rel_counter, ds_rel_factual, ds_rel_plus2, 'sst', storm_name, mode = 'mean', graph = 'normal')
plt.show()

delta_fc = ds_rel_factual - ds_rel_counter
delta_pf = ds_rel_plus2 - ds_rel_factual
delta_pc = ds_rel_plus2 - ds_rel_counter

delta_fc.sst.mean(['lat','lon']).plot(label = 'factual - counter')
delta_pf.sst.mean(['lat','lon']).plot(label = 'plus2 - factual')
delta_pc.sst.mean(['lat','lon']).plot(label = 'plus2 - counter')
plt.axhline(y=0, color='k', linestyle='--')
plt.legend()
plt.ylim(-2,2)
plt.title('Xynthia')
plt.tight_layout()
plt.show()