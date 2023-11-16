import cdsapi
import xarray as xr
c = cdsapi.Client()

c.retrieve(
    'sis-european-risk-extreme-precipitation-indicators',
    {
        'format': 'zip',
        'variable': 'precipitation_at_fixed_return_periods',
        'product_type': 'eca_d', #eca_d, era5, e_obs
        'spatial_coverage': 'europe',
        'temporal_aggregation': '30_year',
        'return_period': [
            '10-yrs', '100-yrs', '25-yrs',
            '50-yrs',
        ],
        'period': '1989-2018',
    },
    '../data/extreme_precip_rp/download_eca_d.zip')


    ################################################################################
# Check output to see if it's working nicely:
with xr.open_dataset('../data/extreme_precip_rp/precipitation-at-fixed-return-period_eobs_europe_era5_30-year_50-yrs_1989-2018_v1.nc') as ds:

    print(ds.keys())

