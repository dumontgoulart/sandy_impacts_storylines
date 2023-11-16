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

def clip_data_to_region(bbox, export_dir, data = 'deltares_data', list_indices = None):
    '''
    Choose a location to clip, a exportind directory and the indices.
    '''
    #Location
    if type(bbox[0]) == float:
        bbox = bbox
    elif type(bbox[0]) == dict:
        bbox = [bbox['lon_min'],bbox['lat_min'],bbox['lon_max'],bbox['lat_max']]

    # Open catalog:
    data_cat = hydromt.DataCatalog(data_libs=data)
  
    # Select indices:
    if list_indices == None:
        list_indices = ['merit_hydro','gebco', 'osm_coastlines', 'osm_landareas', 'gswo', 'fabdem', 'dtu10mdt_egm96', 'dtu10mdt', 'gcn250', 'vito', 'vito_2019_v3.0.1']

    # clip data:
    os.makedirs(export_dir, exist_ok=True) 
    data_cat.export_data(fr'{export_dir}', 
                        bbox=bbox, #xmin, ymin, xmax, ymax
                        time_tuple=None,
                        source_names=list_indices)

def correct_fabdem_nas(path_folder, path_catalogue):
    # Generate function that removes the 0s at the file and converts them to NAs
    cat = hydromt.DataCatalog(path_catalogue)
    da = cat.get_rasterdataset('fabdem', variables=['elevtn'])
    geom_mask = cat.get_geodataframe('osm_coastlines')
    da = da.where(da.raster.geometry_mask(geom_mask, invert=True), da.raster.nodata)
    da.raster.to_raster(f'{path_folder}/fabdem_mask.tif', compress='deflate') # save
    da.close() # NOT WORKING
    # Now update the catalog to the new raster file (add '_mask' to the path)
    cat['fabdem'].path = cat['fabdem'].path.split(".")[0] +'_mask.'+cat['fabdem'].path.split(".")[1]
    cat.to_yml(path_catalogue)

def ini_file_creation(path_output, tref, tstart, tstop, zsini = '0.5', model_type = None, baseline = None,
                      topobathy = None, precip_fn = None, geodataset_fn = None):
    '''
    Still a bit limited to changes in some specific files on topography and forcings. 
    The use of a loaded file for now seems redundant.
    '''

    # Initial checks
    if ('precip' in model_type) and (precip_fn == None):
        raise ValueError('add precipitaiton specification')
    if ('surge' in model_type) and (geodataset_fn == None):
        raise ValueError('add surge specification')

    # General info
    tref = tref.split('-')[0]+tref.split('-')[1]+tref.split('-')[2]+' '+tref.split('-')[3]
    tstart = tstart.split('-')[0]+tstart.split('-')[1]+tstart.split('-')[2]+' '+tstart.split('-')[3]
    tstop = tstop.split('-')[0]+tstop.split('-')[1]+tstop.split('-')[2]+' '+tstop.split('-')[3]
    
    config = configparser.ConfigParser()
    if baseline:
        config.read(baseline)

    # update existing value
    if 'setup_config' not in config.sections():
        config.add_section('setup_config')
    config.set('setup_config', 'tref', tref)
    config.set('setup_config', 'tstart', tstart)
    config.set('setup_config', 'tstop', tstop)
    config.set('setup_config', 'alpha', '0.5')
    config.set('setup_config', 'zsini', zsini)
    

    def build_base_model():
        if 'setup_dep' not in config.sections():
            config.add_section('setup_dep')
        config.set('setup_dep', 'topobathy_fn', topobathy)
        config.set('setup_dep', 'crs', 'utm')
        config.set('setup_dep', 'topobathy_fn', 'gebco')

        if 'setup_merge_topobathy' not in config.sections():
            config.add_section('setup_merge_topobathy')
        config.set('setup_merge_topobathy', 'topobathy_fn', 'gebco')
        config.set('setup_merge_topobathy', 'mask_fn', 'osm_coastlines')
        config.set('setup_merge_topobathy', 'offset_fn', 'dtu10mdt')
        config.set('setup_merge_topobathy', 'merge_method', 'first')
        config.set('setup_merge_topobathy', 'merge_buffer', '2')
        
        if 'setup_mask_active' not in config.sections():
            config.add_section('setup_mask_active')
        config.set('setup_mask_active', 'zmin', '-5')

        if 'setup_cn_infiltration' not in config.sections():
            config.add_section('setup_cn_infiltration')
        config.set('setup_cn_infiltration', 'cn_fn', 'gcn250')
        config.set('setup_cn_infiltration', 'antecedent_runoff_conditions', 'avg')
        
        if 'setup_manning_roughness' not in config.sections():
            config.add_section('setup_manning_roughness')
        config.set('setup_manning_roughness', 'lulc_fn', 'vito')
        config.set('setup_manning_roughness', 'map_fn', 'None')
        
        if 'setup_mask_bounds' not in config.sections():
            config.add_section('setup_mask_bounds')
        config.set('setup_mask_bounds', 'btype', 'waterlevel')
        config.set('setup_mask_bounds', 'mask_fn ', 'osm_coastlines')
        
    
    def build_precip_model():
        if 'setup_p_forcing_from_grid' not in config.sections():
            config.add_section('setup_p_forcing_from_grid')
        config.set('setup_p_forcing_from_grid', 'precip_fn', precip_fn)
        config.set('setup_p_forcing_from_grid', 'dst_res', 'None')
        config.set('setup_p_forcing_from_grid', 'aggregate', 'False')

    def build_surge_model():
        if 'setup_h_forcing' not in config.sections():
            config.add_section('setup_h_forcing')
        config.set('setup_h_forcing', 'geodataset_fn', geodataset_fn)
        config.set('setup_h_forcing', 'timeseries_fn', 'None')
        config.set('setup_h_forcing', 'buffer', '1e4')
        config.set('setup_h_forcing', 'offset_fn', 'dtu10mdt')

    
    dict = {'base': build_base_model, 'precip': build_precip_model, 'surge':build_surge_model}

    # Build model according to the sections wanted
    if type(model_type) == str:
        if model_type not in ['base','precip','surge']:
            raise ValueError('only base, precip and surge allowed')
        dict.get(model_type)()

    elif type(model_type) == list:
        for model in model_type:
            if model not in ['base','precip','surge']:
                raise ValueError('only base, precip and surge allowed')
            dict.get(model)()

    # save to a file
    with open(path_output, 'w') as configfile:
        config.write(configfile)

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True
    
def data_catalog_edit(data_catalog, key_to_edit, key_name, key_path):
    
    if not data_catalog.endswith(".yml"):
        raise ValueError('file needs to be a .yml format')
    
    # Open the YAML file for reading
    with open(data_catalog) as file:
        data = yaml.full_load(file)

    # Duplicate the entry
    duplicated_entry = data[key_to_edit].copy()

    # Modify the value of the duplicated entry
    duplicated_entry['path'] = key_path

    # Add the duplicated entry to the YAML data
    new_key = key_name
    data[new_key] = duplicated_entry

    # Write the modified data back to the YAML file
    with open(data_catalog, 'w') as file:
        yaml.dump(data, file, Dumper=NoAliasDumper)
        

##########################################################################
if __name__ == '__main__':
    os.chdir('D:/paper_3/data/sfincs_ini/')
    # Regions for each storm
    bbox_xaver = [7.868958,53.118757,10.835266,54.321329]
    bbox_xynthia = [-1.900635,45.648608,-0.689392,46.521076]
    bbox_sandy = [-74.404907,40.355963,-72.811890,41.250968] 

    # Dict of storms
    dict_storms = {
        'xynthia':{'bbox':bbox_xynthia,'tref':'2010-02-23','tstart':'2010-02-23','tend':'2010-03-02'},
        'xaver':{'bbox':bbox_xaver,'tref':'2013-12-05','tstart':'2013-12-05','tend':'2013-12-07'},
        'sandy':{'bbox':bbox_sandy,'tref':'2012-10-29','tstart':'2012-10-29','tend':'2012-10-31'}
        }

    # Name of region/storm - Change this to generate new data catalogues and ini files.
    storm = 'sandy' 
    
    #1) Clip deltares catalog to a specific region based on a storm location.
    clip_data_to_region(dict_storms[storm]['bbox'], export_dir = f'data_deltares_{storm}', list_indices = None)
    
    # TODO: Still have to check if this code is good. Or if it's cutting wrong stuff
    #2) Now remove the 0s from fabdem to make it NAs and work with Hydromt:
    correct_fabdem_nas(path_folder = f'data_deltares_{storm}', path_catalogue = f'data_deltares_{storm}/data_catalog.yml')

    # 3) Add the precipitation and storm surge file to the data catalog (optional)


    # 3) Create ini files for each storm
    topobathy_chosen = 'fabdem' #'fabdem'

    ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/sfincs_{storm}_base.ini', 
                    tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0',
                    model_type = 'base', topobathy = topobathy_chosen)

    ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/update_sfincs_{storm}_rain.ini', 
                    tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0', 
                    model_type = 'precip', topobathy = topobathy_chosen, precip_fn = f'era5_hourly_{storm}')

    ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/update_sfincs_{storm}_surge.ini', 
                    tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0', 
                    model_type = 'surge', topobathy = topobathy_chosen, geodataset_fn = f'gtsm_{storm}_waterlevel') 
    

    # # SPECTRAL NUDGING TEST
    
    # ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/update_sfincs_{storm}_sn_factual_rain.ini', 
    #                 tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0', 
    #                 model_type = 'precip', topobathy = topobathy_chosen, precip_fn = f'spectral_nudging_factual_rain_{storm}')
    
    # ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/update_sfincs_{storm}_sn_counter_rain.ini', 
    #                 tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0', 
    #                 model_type = 'precip', topobathy = topobathy_chosen, precip_fn = f'spectral_nudging_counter_rain_{storm}')

    # ini_file_creation(path_output = f'D:/paper_3/data/sfincs_ini/update_sfincs_{storm}_sn_plus2_rain.ini', 
    #                 tref = dict_storms[storm]['tref'], tstart = dict_storms[storm]['tstart'], tstop = dict_storms[storm]['tend'], zsini='0.0', 
    #                 model_type = 'precip', topobathy = topobathy_chosen, precip_fn = f'spectral_nudging_plus2_rain_{storm}')

