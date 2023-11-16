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

'''
This script is intended to load gtsm runs and return gtsm runs + sea level rise.

For this GTSM runs with waterlevels (composed of tides + surges) are required. Then sealevel rises values based on KNMI studies are added.

The script is intended to be used in the following way:
1. Load the gtsm runs with waterlevels
2. Add the sealevel rise to the gtsm runs
3. Save the gtsm runs with waterlevels + sealevel rise
'''

# define storm name
storm = 'sandy'
sealevel_scenarios = [35, 71, 101] #in centimeters but later on it's converted to meters

# GW scenarios
scenarios = ["counter_1",
            "counter_2",
            "counter_3",
            "factual_1", 
            "factual_2",
            "factual_3",
            "plus2_1",
            "plus2_2",
            "plus2_3" ]

# Add sealevel rise to the gtsm runs

for sealevel_scenario in sealevel_scenarios:
    for scenario in scenarios:
        # load gtsm runs with waterlevels (.nc files)
        ds_gtsm = xr.open_dataset(f'D:/paper_3/data/gtsm_local_runs/{storm}_fine_grid/gtsm_fine_{storm}_{scenario}_0000_his_waterlevel.nc')

        ds_gtsm_slr = ds_gtsm['waterlevel'] + sealevel_scenario / 100
        ds_gtsm_slr = ds_gtsm_slr.to_dataset(name = 'waterlevel')

        # create a new folder for the gtsm runs with waterlevels + sealevel rise
        output_folder = f'D:/paper_3/data/gtsm_local_runs/{storm}_slr{sealevel_scenario}_fine_grid'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the gtsm runs with waterlevels + sealevel rise
        ds_gtsm_slr.to_netcdf(f'{output_folder}/gtsm_fine_{storm}_slr{sealevel_scenario}_{scenario}_0000_his_waterlevel.nc')
    

# do the same as above but for xynthia and only the reanlaysis file:
storm = 'xynthia'
sealevel_scenarios = [19, 43, 60] #in centimeters but later on it's converted to meters

for sealevel_scenario in sealevel_scenarios:
    # load gtsm runs with waterlevels (.nc files)
    ds_gtsm = xr.open_dataset(f'D:/paper_3/data/gtsm_outputs/gtsm_Xynthia_waterlevel_2010-02-23_2010-03-02.nc')

    ds_gtsm_slr = ds_gtsm['waterlevel'] + sealevel_scenario / 100
    ds_gtsm_slr = ds_gtsm_slr.to_dataset(name = 'waterlevel')

    # create a new folder for the gtsm runs with waterlevels + sealevel rise
    output_folder = f'D:/paper_3/data/gtsm_outputs/{storm}_slr{sealevel_scenario}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the gtsm runs with waterlevels + sealevel rise
    ds_gtsm_slr.to_netcdf(f'{output_folder}/gtsm_fine_{storm}_slr{sealevel_scenario}_era5_0000_his_waterlevel.nc')

