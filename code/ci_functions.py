import os,sys
import glob
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
import xarray as xr
import dask_geopandas
from tqdm import tqdm 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def raster_to_vector(xr_raster):
    """
    Convert a raster to a vector representation.

    Args:
        xr_raster (xarray.DataArray): Input raster data as xarray.DataArray.

    Returns:
        gpd.GeoDataFrame: Vector representation of the input raster.
    """

    # Convert xarray raster to pandas DataFrame
    df = xr_raster.to_dataframe()

    # Filter DataFrame to select rows where band_data is 1
    df_1 = df.dropna().reset_index()

    # Create a Shapely Point geometry column from x and y values
    df_1['geometry'] = shapely.points(df_1.x.values, df_1.y.values)

    # Remove unnecessary columns from the DataFrame
    df_1 = df_1.drop(['x', 'y','spatial_ref'], axis=1)

    # Calculate the resolution of the raster
    resolution = xr_raster.x[1].values - xr_raster.x[0].values

    # Buffer the Point geometries by half of the resolution with square caps
    df_1.geometry = shapely.buffer(df_1.geometry, distance=resolution/2, cap_style='square').values

    # Convert the DataFrame to a GeoDataFrame
    return gpd.GeoDataFrame(df_1)     

def building_flood_intersect(vector, raster_in):
    """
    Calculate zonal statistics of a raster dataset based on a vector dataset.
    
    Parameters:
    - vector_in: Vector dataset file (as GeoPandas GeoDataFrame).
    - raster_in (str): Path to the raster dataset file (in NetCDF format).
    
    Returns:
    - pandas.Series: A series containing the zonal statistics values corresponding to each centroid point in the vector dataset.
    """
    
    # Open the raster dataset using the xarray library
    raster = xr.open_dataset(raster_in, engine="netcdf4")   
    
    # Progress bar setup for obtaining values
    tqdm.pandas(desc='obtain values')
       
    # Convert the clipped raster dataset to a vector representation
    raster_vector = raster_to_vector(raster)

    # set crs
    raster_vector = raster_vector.set_crs(32618)
    raster_vector = raster_vector.to_crs(4326)    
    
    # Create a dictionary mapping each index to its corresponding band data value
    band_data_dict = dict(zip(list(raster_vector.index), raster_vector['hmax'].values))

    # Construct an STRtree from the vector geometry values
    tree = shapely.STRtree(raster_vector.geometry.values)

    # Apply a function to calculate zonal statistics for each centroid point in the vector dataset
    df_intersect = pd.DataFrame(tree.query(vector.geometry, predicate='intersects').T,columns=['building_index','flood_data'])
    df_intersect.flood_data = df_intersect.flood_data.apply(lambda x: band_data_dict[x])
    
    return vector.merge(df_intersect,left_index=True,right_on='building_index')

def cis_flood_intersect(vector, raster_in):
    """
    Calculate zonal statistics of a raster dataset based on a vector dataset.
    
    Parameters:
    - vector_in: Vector dataset file (as GeoPandas GeoDataFrame).
    - raster_in (str): Path to the raster dataset file (in NetCDF format).
    
    Returns:
    - pandas.Series: A series containing the zonal statistics values corresponding to each centroid point in the vector dataset.
    """
    
    # Open the raster dataset using the xarray library
    raster = xr.open_dataset(raster_in, engine="netcdf4")   
    
    # Progress bar setup for obtaining values
    tqdm.pandas(desc='obtain values')
       
    # Convert the clipped raster dataset to a vector representation
    raster_vector = raster_to_vector(raster)

    # set crs
    raster_vector = raster_vector.set_crs(32618)
    
    # Create a dictionary mapping each index to its corresponding band data value
    band_data_dict = dict(zip(list(raster_vector.index), raster_vector['hmax'].values))

    # Construct an STRtree from the vector geometry values
    tree = shapely.STRtree(raster_vector.geometry.values)

    # Apply a function to calculate zonal statistics for each centroid point in the vector dataset
    df_intersect = pd.DataFrame(tree.query(vector.geometry.centroid, predicate='intersects').T,columns=['building_index','flood_data'])
    df_intersect.flood_data = df_intersect.flood_data.apply(lambda x: band_data_dict[x])
    
    return vector.merge(df_intersect,left_index=True,right_on='building_index')
