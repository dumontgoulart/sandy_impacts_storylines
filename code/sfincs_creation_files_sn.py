# -*- coding: utf-8 -*-
"""
Prepare files for HydroMT
Created on Tue Oct  9 16:39:47 2022
@author: morenodu
"""

import os
import hydromt
import matplotlib.pyplot as plt
import configparser
import yaml
os.chdir('D:/paper_3/code')
import clip_deltares_data_to_region as clip
import hydromt_sfincs_pipeline as sfincs_pipe

root_folder = 'D:/paper_3/data/sfincs_ini/spectral_nudging/'

os.chdir(root_folder)

########################################################################################################################
# general setup for all storms
# Regions for each storm
bbox_xaver = [7.868958,53.118757,10.835266,54.321329]
bbox_xynthia = [-1.900635,45.648608,-0.689392,46.521076] # Standard: -1.900635,45.648608,-0.689392,46.521076; Enlarge it: -3.592529,45.648608,-0.966797,47.743017
bbox_sandy = [-74.564209,40.153687,-72.765198,41.353103] # 28 - 15 // 13/10 -> 31/10 -74.564209,40.153687,-72.765198,41.353103
bbox_sandy_small = [-74.212646,40.532589,-73.487549,41.095912]

# Dict of storms
dict_storms = {
    'xynthia':{'bbox':bbox_xynthia,'tref':'20100226 000000','tstart':'20100226 000000','tend':'20100301 000000', 'topo_map':'fabdem', 'area_mask':None},
    'xynthia_slr19':{'bbox':bbox_xynthia,'tref':'20100226 000000','tstart':'20100226 000000','tend':'20100301 000000', 'topo_map':'fabdem', 'area_mask':None},
    'xynthia_slr43':{'bbox':bbox_xynthia,'tref':'20100226 000000','tstart':'20100226 000000','tend':'20100301 000000', 'topo_map':'fabdem', 'area_mask':None},
    'xynthia_slr60':{'bbox':bbox_xynthia,'tref':'20100226 000000','tstart':'20100226 000000','tend':'20100301 000000', 'topo_map':'fabdem', 'area_mask':None},
    'xaver':{'bbox':bbox_xaver,'tref':'20131205 000000','tstart':'20131205 000000','tend':'20131207 000000', 'topo_map':'fabdem', 'area_mask':None},
    'sandy':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'},
    'sandy_historical':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'},
    'sandy_shifted':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'},
    'sandy_slr35':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'},
    'sandy_slr71':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'},
    'sandy_slr101':{'bbox':bbox_sandy,'tref':'20121028 120000','tstart':'20121028 120000','tend':'20121031 000000', 'topo_map':'cudem', 'area_mask':'d:/paper_3/data/us_dem/sandy_ny_coast_b12km_mask.gpkg'}
    }

########################################################################################################################
# Name of region/storm - Change this to generate new data catalogues and ini files.
storm = 'sandy_shifted'  #sandy_slr10 sandy_historical
topo_map = dict_storms[storm]['topo_map']
topobathy_chosen = 'fabdem'
type_of_data = 'echam_sn'
mode_rain = '_rain'
mode_surge = '_surge'
mask_zsini = True
area_mask = dict_storms[storm]['area_mask']
res = 50
# update date times
tref = dict_storms[storm]['tref']
tstart = dict_storms[storm]['tstart']
tstop = dict_storms[storm]['tend']
year = dict_storms[storm]['tref'][0:4]

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

if '_historical' in storm:
    scenarios = ['']

# Data to be used in the model
data_libs = ['d:\paper_3\data\sfincs_ini\data_gtsm_test.yml', root_folder+f'data_deltares_{storm}/data_catalog.yml']
########################################################################################################################

list_indices_storm = {'sandy':['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', topo_map, 'fabdem', 'dtu10mdt', 'gcn250', 'vito', "rivers_lin2019_v1"],
                      'xynthia':['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', topo_map, 'dtu10mdt', 'gcn250', 'vito',"rivers_lin2019_v1"],}


# 1) Clip deltares catalog to a specific region based on a storm location.
# Takes some time
clip.clip_data_to_region(dict_storms[storm]['bbox'], export_dir = f'data_deltares_{storm}', data = ['deltares_data', data_libs[0]], list_indices = list_indices_storm[storm.split('_')[0]])

# #2) Now remove the 0s from fabdem to make it NAs and work with Hydromt:
clip.correct_fabdem_nas(path_folder = f'data_deltares_{storm}', path_catalogue = f'data_deltares_{storm}/data_catalog.yml')

# 3) Create the data catalogue for the model
# base model for static data
sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = 'base', storm = storm, data_libs = data_libs, area_mask = area_mask,
                                      bbox = dict_storms[storm]['bbox'], writing_mode = 'write', topo_map = topo_map, res = res, tref = tref, tstart = tstart, tstop = tstop)

for scenario in scenarios:
    print(scenario)
    # 3) Add the precipitation and storm surge file to the data catalog (optional)
    if '_slr' in storm:     
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'precip_echam61_spectral_nudging_template', f'precip_{type_of_data}_{scenario}_{storm}', f'd:/paper_3/data/spectral_nudging_data/regular_grid/{storm.split("_slr")[0]}/BOT_t_HR255_Nd_SU_1015_{scenario}_{year}_{storm.split("_slr")[0]}_storm.Ptot_mmh_regular_clip.nc') 
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'gtsm_spectral_nudging_waterlevel_template', f'gtsm_{type_of_data}_{scenario}_{storm}_waterlevel', f'd:/paper_3/data/gtsm_local_runs/{storm}_fine_grid/gtsm_fine_{storm}_{scenario}_0000_his_waterlevel.nc')

    elif '_historical' in storm:     
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'precip_echam61_spectral_nudging_template', f'precip_{type_of_data}_{scenario}_{storm}', f'd:/paper_3/data/spectral_nudging_data/regular_grid/{storm}/CLI_MPI-ESM-XR_t255l95_echam_{storm.split("_historical")[0]}_Ptot_mmh_regular_clip.nc') 
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'gtsm_spectral_nudging_waterlevel_template', f'gtsm_{type_of_data}_{scenario}_{storm}_waterlevel', f'd:/paper_3/data/gtsm_local_runs/{storm}_fine_grid/gtsm_fine_{storm}_0000_his_waterlevel.nc')

    else:
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'precip_echam61_spectral_nudging_template', f'precip_{type_of_data}_{scenario}_{storm}', f'd:/paper_3/data/spectral_nudging_data/regular_grid/{storm}/BOT_t_HR255_Nd_SU_1015_{scenario}_{year}_{storm}_storm.Ptot_mmh_regular_clip.nc')
        clip.data_catalog_edit(f'd:/paper_3/data/sfincs_ini/data_gtsm_test.yml', 'gtsm_spectral_nudging_waterlevel_template', f'gtsm_{type_of_data}_{scenario}_{storm}_waterlevel', f'd:/paper_3/data/gtsm_local_runs/{storm}_fine_grid/gtsm_fine_{storm}_{scenario}_0000_his_waterlevel.nc')
    
    # 4) Generate the sfincs files to be run by the model using hydromt
    sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = scenario, storm = storm, data_libs = data_libs, area_mask = area_mask,
                                      bbox = dict_storms[storm]['bbox'], writing_mode = 'update', topo_map = topo_map, res = res, mode = 'rain', tref = tref, tstart = tstart, tstop = tstop)
    sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = scenario, storm = storm, data_libs = data_libs, area_mask = area_mask,
                                      bbox = dict_storms[storm]['bbox'], writing_mode = 'update', topo_map = topo_map, res = res, mode = 'surge', tref = tref, tstart = tstart, tstop = tstop)
    sfincs_pipe.generate_sfincs_model(root_folder = root_folder, scenario = scenario, storm = storm, data_libs = data_libs, area_mask = area_mask,
                                      bbox = dict_storms[storm]['bbox'], writing_mode = 'update', topo_map = topo_map, res = res, mode = 'rain_surge', tref = tref, tstart = tstart, tstop = tstop)
    

# # 5) RUN SFINCS -- NOT RUNNING IN THIS SCRIPTAT THE LOACL PC ANYMORE - USE SNELLIUS
# fn_exe = r"d:\paper_3\data\sfincs_model_v2.0.1\sfincs.exe"

# for scenario in scenarios:

    # sfincs_pipe.run_sfincs(base_root = f'{root_folder}{storm}_{scenario}' + mode_rain, fn_exe = fn_exe)
    # sfincs_pipe.run_sfincs(base_root = f'{root_folder}{storm}_{scenario}'+ mode_surge, fn_exe = fn_exe)
    # sfincs_pipe.run_sfincs(base_root = f'{root_folder}{storm}_{scenario}'+ mode_rain + mode_surge, fn_exe = fn_exe)
    # sfincs_pipe.run_sfincs(base_root = rf'D:\paper_3\data\sfincs_ini\spectral_nudging\xynthia_counter_2_surge', fn_exe = fn_exe)


# 6) GENERATE MAPS OF MAX INUNDATION
for scenario in scenarios:

    # mod_b = sfincs_pipe.generate_maps(sfincs_root = f'{root_folder}{storm}_{scenario}', catalog_path = f'data_deltares_{storm}/data_catalog.yml') #sfincs_root, catalog_path, storm, scenario, mode
    mod_r = sfincs_pipe.generate_maps(sfincs_root = f'{root_folder}{storm}_{scenario}'+mode_rain, catalog_path = f'data_deltares_{storm}/data_catalog.yml', storm = storm, scenario = scenario, mode = mode_rain)
    mod_s = sfincs_pipe.generate_maps(sfincs_root = f'{root_folder}{storm}_{scenario}'+mode_surge, catalog_path = f'data_deltares_{storm}/data_catalog.yml', storm = storm, scenario = scenario, mode = mode_surge)
    mod_rs = sfincs_pipe.generate_maps(sfincs_root = f'{root_folder}{storm}_{scenario}'+mode_rain + mode_surge, catalog_path = f'data_deltares_{storm}/data_catalog.yml', storm = storm, scenario = scenario, mode = mode_rain+mode_surge)

