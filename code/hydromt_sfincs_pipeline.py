import hydromt
from hydromt_sfincs import SfincsModel, utils
from hydromt.config import configread
from hydromt.log import setuplog
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt
import subprocess
import os
from os.path import join, isfile, isdir, dirname
os.chdir('D:/paper_3/data/sfincs_ini')

'''
This script is intended to run SFINCS for a given storm and a given scenario.
Script is divided in three parts:
1) Preparing files with HydroMT;
2) Running SFINCS;
3) Reading and plotting inundations maps of SFINCS using HydroMT.
'''
#################
# 1) Generating INI files with HydroMT;
#################
def sfincs_files_generation(base_root, ini_file, bbox, data_libs, mode = '', update_ini = None, plot_figure = False, mask_zsini_land = False):
    if not os.path.exists(base_root):
        # create the folder if it doesn't exist
        os.makedirs(base_root)
    # build model base layers
    logger = setuplog('update', join(base_root, "hydromt.log"), log_level=10)
    region = {'bbox': bbox}
    opt = configread(ini_file)
    # kwargs = opt.pop('global',{})
    mod = SfincsModel(root=base_root, data_libs=data_libs, logger=logger, mode = 'w+')

    if update_ini != None:
        opt=configread(update_ini)
        mod = SfincsModel(root=base_root, data_libs=data_libs, logger=logger, mode='r')  # initialize model with default logger in read mode
        mod.update(model_out=base_root+mode, opt=opt)
    
    mod.build(region=region, opt=opt)

    if mask_zsini_land:
        # We set initial water depth in the river channel (called zsini or a restart file in SFINCS)
        mod = SfincsModel(root=base_root+mode, data_libs=data_libs, logger=logger, mode='r+')  # initialize model with default logger in read mode

        # Mask depth based on fabdem land mask
        land_areas = mod.data_catalog.get_geodataframe('osm_landareas')
        dep_masked = mod.staticmaps['dep'].raster.clip_geom(land_areas, mask=False)

    # TODO: DO we have to update these value to include river wet zones?
        zsini = xr.where(dep_masked['mask'] == 1, -100, 0.15)

        zsini.raster.set_nodata(0)
        mod.set_states(zsini, 'zsini')
        mod.config.pop('zsini',None)
        # remove zsini from opt:
        opt['setup_config'].pop('zsini',None)
        mod.update(model_out= base_root+mode, opt=opt)
        # mod.build(region=region, res=res, opt=opt)


    if plot_figure == True:
        # PLOT
        mod.plot_basemap()
        plt.show()

#################
# 1) Generating SFINCS files with HydroMT;
#################

def generate_sfincs_model(root_folder, scenario, storm, data_libs, bbox, topo_map, writing_mode, tref, tstart, tstop, res = 100, mode = None, area_mask = None):
    # Create sfincs model
    if mode != None:
        root=f'{root_folder}{storm}_{scenario}_{mode}'
    else:
        root=f'{root_folder}{storm}_{scenario}'

    if writing_mode == 'write':
        sf = SfincsModel(data_libs = data_libs, root = root, mode="w+")
        # Set up the grid
        sf.setup_grid_from_region(region={'bbox':bbox}, res = res)
        # Load in wanted elevation datasets:
        # the 2nd elevation dataset (gebco) is used where the 1st dataset returned nodata values
        if topo_map in ['fabdem','merit']:
            datasets_dep = [{"elevtn": topo_map, "zmin": 0.001}, {"elevtn": "gebco"}]
            dep = sf.setup_dep(datasets_dep=datasets_dep, buffer_cells = 2)
        elif topo_map == 'cudem':
            datasets_dep = [{"elevtn": topo_map}, {"elevtn": 'fabdem', "zmin": 0.0001}, {"elevtn": "gebco"}]
            dep = sf.setup_dep(datasets_dep=datasets_dep)
        else:
            raise ValueError('topo_map must be either fabdem, merit or cudem for now')

        # Choosing how to choose you active cells can be based on multiple criteria, here we only specify a minimum elevation of -5 meters
        if (area_mask != None) and (storm.split('_')[0] == 'sandy'):
            sf.setup_mask_active(mask = area_mask, include_mask=r"d:\paper_3\data\us_dem\osm_landareas_mask.gpkg", zmin=-10, reset_mask=True)
        else:
            sf.setup_mask_active(zmin=-10, reset_mask=True)

        # Here we add water level cells along the coastal boundary, for cells up to an elevation of -5 meters
        sf.setup_mask_bounds(btype="waterlevel", zmax=-3, reset_bounds=True)

        # Add spatially varying roughness data:
        # read river shapefile and add manning value to the attributes, enforce low roughness data
        gdf = sf.data_catalog.get_rasterdataset("rivers_lin2019_v1", geom=sf.region).to_crs(
            sf.crs
        )
        gdf["geometry"] = gdf.buffer(50)
        gdf["manning"] = 0.03

        # rasterize the manning value of gdf to the  model grid
        da_manning = sf.grid.raster.rasterize(gdf, "manning", nodata=np.nan)
        datasets_rgh = [{"manning": da_manning}, {"lulc": "vito"}]

        sf.setup_manning_roughness(
            datasets_rgh=datasets_rgh,
            manning_land=0.04,
            manning_sea=0.02,
            rgh_lev_land=0,  # the minimum elevation of the land
        )

        # ADD SUBGRID - Make subgrid derived tables TODO: Still not implemented because code does not work, and I wonder if the mix of my data with gebco makes subgrid irrelevant.
        # sf.setup_subgrid(
        #     datasets_dep=datasets_dep,
        #     datasets_rgh=datasets_rgh,
        #     nr_subgrid_pixels=5,
        #     write_dep_tif=True,
        #     write_man_tif=False,
        # )

        # Add spatially varying infiltration data:
        sf.setup_cn_infiltration("gcn250", antecedent_moisture="avg")

        # Add time-series:
        from datetime import datetime
        start_datetime = datetime.strptime(tstart, '%Y%m%d %H%M%S')
        stop_datetime = datetime.strptime(tstop, '%Y%m%d %H%M%S')
        sf.setup_config(
            **{
                "tref": tref,
                "tstart": tstart,
                "tstop": tstop,
                "dtout":10800,
                "dtmaxout": (stop_datetime - start_datetime).total_seconds(),
                "advection": 0,
            }
        )

    elif writing_mode == 'update':
        ## Now update model with forcings
        sf = SfincsModel(data_libs = data_libs, root = f'{root_folder}{storm}_base', mode="r")
        sf.read()
        sf.set_root(root = root, mode='w+')
        # update paths to static files in base root
        
        modes = ['rain', 'surge', 'rain_surge']
        if mode in modes:
            if mode in ['rain', 'rain_surge']:
                # rainfall
                sf.setup_precip_forcing_from_grid(precip=f"precip_echam_sn_{scenario}_{storm}", aggregate=False) 
            if mode in ['surge', 'rain_surge']:
                # SURGE - WATERLEVEL
                sf.setup_waterlevel_forcing(geodataset=f'gtsm_echam_sn_{scenario}_{storm}_waterlevel', offset = 'dtu10mdt', buffer = 1e4)
                # gdf_locations = sf.forcing['bzs'].vector.to_gdf()
                # sf.setup_mask_active(mask = area_mask, zmin=-4, include_mask = gdf_locations, reset_mask=True)         

    # save model
    sf.write()  # write all because all folders are run in parallel on snellius




#################
# 2) Running SFINCS / RUN MODEL
#################
def check_finished(root):
    finished = False
    if isfile(join(root, 'sfincs.log')):
        with open(join(root, 'sfincs.log'), 'r') as f:
            finished = np.any(['Simulation is finished' in l for l in f.readlines()])
    return finished

def run_sfincs(base_root, fn_exe):
    runs = [dirname(fn) for fn in glob.glob(join(base_root, 'sfincs.inp')) if not check_finished(dirname(fn))]
    n = len(runs)
    print(n)
    if n == 0:
        print('No simulations run because simulation is finished')
    for i, root in enumerate(runs):
        print(f'{i+1:d}/{n:d}: {base_root}')
        with open(join(base_root, "sfincs.log"), 'w') as f:
            p = subprocess.Popen([fn_exe], stdout=f, cwd=root)
            p.wait()
            print('Check sfincs.log inside folder for progress.')

############
# 3) CHECK RESULTS
#############
def generate_maps(sfincs_root, catalog_path, storm, scenario, mode):
      # (relative) path to sfincs root
    mod_nr = SfincsModel(sfincs_root, mode="r")
    # we can simply read the model results (sfincs_map.nc and sfincs_his.nc) using the read_results method
    mod_nr.read_results()
    mod_nr.write_raster(f"results.hmax", compress="LZW")

    mod_nr.data_catalog.from_yml(catalog_path)
    gswo = mod_nr.data_catalog.get_rasterdataset("gswo", geom=mod_nr.region, buffer=10)
    # permanent water where water occurence > 5%
    gswo_mask = gswo.raster.reproject_like(mod_nr.grid, method="max") <= 5

    hmin = 0.05  # minimum flood depth [m] to plot
    da_hmax = mod_nr.results["hmax"]  # hmax is computed from zsmax - zb
    if len(da_hmax.dims) != 2:
        # try to reduce to 2D by taking maximum over time dimension
        if "time" in da_hmax.dims:
            da_hmax = da_hmax.max("time")
        elif "timemax" in da_hmax.dims:
            da_hmax = da_hmax.max("timemax")
    # get overland flood depth with GSWO and set minimum flood depth
    da_hmax_fld = da_hmax.where(gswo_mask).where(da_hmax >= hmin)
    # update attributes for colorbar label later
    da_hmax_fld.attrs.update(long_name="flood depth", unit="m")
    da_hmax_fld = da_hmax_fld.where(da_hmax != -9999, np.nan)
    
    # create folder if not exists:
    path_to_folder = rf'D:\paper_3\data\sfincs_inundation_results\{storm}'
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
        os.makedirs(path_to_folder+r"\raster")
        # save to netcdf and raster file
    da_hmax_fld.to_netcdf(rf'{path_to_folder}\hmax_{storm}_{scenario}{mode}.nc') 
    da_hmax_fld.rio.to_raster(rf'{path_to_folder}\raster\hmax_{storm}_{scenario}{mode}.tiff', tiled=True, compress='LZW')

    return mod_nr

def run_pipeline_compound_events(base_root, ini_file, bbox, data_libs, fn_exe):
    # 1) Prepare
    # BASE
    sfincs_files_generation(base_root = base_root, ini_file = ini_file, bbox = bbox, data_libs = data_libs)    
    # 2) Run - Don't run in series as the executable is not paused inside the script - it doesn't run in series
    run_sfincs(base_root = base_root, fn_exe = fn_exe)
    # 3) Results
    mod = generate_maps(sfincs_root = base_root)
    return mod

