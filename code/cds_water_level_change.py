import os
os.chdir('D:/paper_3/code')
import cdsapi

def global_water_level_changes(year, month, type = 'timeseries'):

    if type == 'indicators':

        c = cdsapi.Client()

        c.retrieve(
            'sis-water-level-change-indicators-cmip6',
            {
                'variable': [
                    'surge_level', 'total_water_level',
                ],
                'derived_variable': 'absolute_value',
                'period': '1985-2014',
                'format': 'zip',
                'statistic': [
                    '100_year', '10_year', '50_year',
                ],
                'product_type': 'reanalysis',
                'confidence_interval': 'best_fit',
            },
            '../data/sis-water-level-change-indicators-era5_gtsm.zip')

    elif type == 'timeseries':

        c = cdsapi.Client()

        c.retrieve(
            'sis-water-level-change-timeseries-cmip6',
            {
                'variable': [
                    'storm_surge_residual', 'total_water_level',
                ],
                'experiment': 'reanalysis',
                'temporal_aggregation': [
                    'hourly',
                ],
                'year': year,
                'month': month,
                'format': 'zip',
            },
            f'../data/sis-water-level-change-timeseries-{year[0]:02d}-{month[0]:02d}-era5_gtsm.zip')
    else:
        raise ValueError('Only timeseries or indicator')


if __name__ == '__main__':
    # Dates
    year = [2012] # String
    month = [10,11] # String, It can also be a list of strings ['01','02']
    global_water_level_changes(year = year, month = month, type = 'timeseries')