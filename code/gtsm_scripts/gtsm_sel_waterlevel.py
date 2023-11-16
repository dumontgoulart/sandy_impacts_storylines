import os
import xarray as xr
import numpy as np

folder_path = 'D:/paper_3/data/gtsm_local_runs/'
var='waterlevel'
# Iterate over the files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('his.nc'):
        # Construct the file path
        file_path = os.path.join(folder_path, filename)

        # Open the dataset
        ds = xr.open_dataset(file_path)

        stations_coords = np.arange(ds.dims['stations'])

        ds = ds.assign_coords(stations=stations_coords)
        ds = ds.drop_vars('station_name')
        
        # Select the variable
        da = ds[var]

        da = da.rename({'station_x_coordinate': 'x', 'station_y_coordinate': 'y'})
        da["y"].attrs.update({'units': 'degrees_north', 'short_name': 'latitude', 'long_name': 'latitude', 'crs': 'EPSG:4326'})
        da["x"].attrs.update({'units': 'degrees_east', 'long_name': 'longitude', 'short_name': 'longitude'})

        da.attrs.update({'geospatial_lat_units': 'degrees_north', 
                        'geospatial_lat_resolution': 'point', 
                        'geospatial_lon_units': 'degrees_east', 
                        'geospatial_lon_resolution': 'point', 
                        'geospatial_vertical_units': 'm',
                        'geospatial_vertical_positive': 'up', 
                        'crs': 'EPSG:4326'})
        
        # Construct the output file path
        output_file_path = os.path.join(folder_path, f'{os.path.splitext(filename)[0]}_{var}.nc')

        # Save to netCDF
        da.to_netcdf(output_file_path)