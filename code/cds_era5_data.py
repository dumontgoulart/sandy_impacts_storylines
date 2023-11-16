'''
Import Copernicus data from the web based on requirements set by the list of parameters.
Data can be a string ex:'10' or a list of strings ex:['10','11']

Created by Henrique Goulart
'''
import os
os.chdir('D:/paper_4/code')
import cdsapi
import xarray as xr

################################################################################
### Requirements 
# Different versions of reanalysis
storm = 'idai'#'Sandy'#'xynthia' # 'xaver'
list_of_reanalysis = {'single':'reanalysis-era5-single-levels','land':'reanalysis-era5-land'}
key = 'single'
# Variables
variables = ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
            'total_precipitation'] #'total_precipitation'
# Dates
year = '2019' # String
month = '03' # String, It can also be a list of strings ['01','02']
days = list(range(4, 19+1, 1)) # or string or list of strings (up to 31)
# Boundary limits of the area to be analysed (integers)  -89.472656,7.532249,-30.078125,56
lon_min, lon_max = 31, 42
lat_min, lat_max = -22, -13
bbox = [lat_max, lon_min, lat_min, lon_max]

# Set output destination
if type(year) == list:
    year_str = f'{year[0]}_{year[-1]}'
else:
    year_str = year
if type(month) == list:
    month_str = f'{month[0]}_{month[-1]}'
else:
    month_str = month
# OUTPUT
output_name_loc = f'../data/era5/era5_hourly_vars_{storm}_{key}_{year_str}_{month_str}.nc'
print(output_name_loc)
################################################################################

################################################################################
# Run client function
c = cdsapi.Client()
# ERA5 single levels and land
c.retrieve(
    list_of_reanalysis[key], #'reanalysis-era5-single-levels','reanalysis-era5-land'
    {
        'product_type': 'reanalysis',
        'variable': variables,
        'year': year,
        'month': month,
        'day': days,
        'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                 '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                 '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                 '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',],
        'area': bbox,
        'format': 'netcdf',
    },
    output_name_loc)

################################################################################
# Check output to see if it's working nicely:
with xr.open_dataset(output_name_loc) as ds:

    print(ds.keys())