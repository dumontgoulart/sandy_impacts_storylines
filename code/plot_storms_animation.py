import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs # for map projections
import cartopy.feature as cfeature
import os
os.chdir('D:/paper_3/code')
import sn_storm_preprocessing as prep
import xarray as xr
import numpy as np

# Define a function that takes the 'ds' variable as an argument, and optionally other parameters; animation is generated
def create_animation(ds, variable = 'Ptot', threshold=5, file_name='precip_animation.gif'):

    # Create a masked array to mask values below the threshold
    masked_da = ds[variable].where(ds[variable] > threshold)

    # Define a function that creates the base map with cartopy features and returns a figure and an axis object
    def make_figure():
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    # 
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        return fig, ax

    # Define the vmin, vmax and cmap values for the different variables
    if variable == 'Ptot':
        vmin, vmax, cmap = 0, 10, 'Blues'
    elif variable == 'U10m':
        vmin, vmax, cmap = 10, 30, 'Blues'
    elif variable == 'mslp':
        vmin, vmax, cmap = 970, 1030, 'RdBu'

    # Call the make_figure function and assign the returned values to fig and ax variables
    fig, ax = make_figure()

    # Define a function that draws the data on the map for a given frame (time index) and optionally adds a colorbar
    def draw(frame, add_colorbar):
        # Clear the previous plot
        ax.clear()
        # Re-add the cartopy features
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Plot the grid for the current frame
        grid = masked_da.isel(time=frame)
        contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = cmap, add_colorbar=add_colorbar, vmin=vmin, vmax=vmax)
        title = f"Ptot - {str(masked_da.time[frame].values)[:19]}"
        ax.set_title(title)
        return contour

    # Define an init function that calls the draw function for the first frame and adds a colorbar
    def init():
        return draw(0, add_colorbar=True)

    # Define an animate function that calls the draw function for subsequent frames without adding a colorbar
    def animate(frame):
        return draw(frame, add_colorbar=False)

    # Create an animation object using FuncAnimation with the figure, animate, init_func, frames, interval and repeat arguments
    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(masked_da.time), interval=200, repeat=False)

    # Save the animation as a video file using the writer and fps arguments
    ani.save(file_name)

    plt.close()

    # Return the animation object
    return ani

# Define a function that takes the 'ds' variable as an argument, and optionally other parameters; animation is generated
def create_animation_surge(ds, file_name='surge_animation.gif'):

    # Create a masked array to mask values below the threshold
    if 'x' in ds.coords:
        ds = ds.rename({'x':'station_x_coordinate', 'y':'station_y_coordinate'})
    if 'lat' in ds.coords:
        ds = ds.rename({'lon':'station_x_coordinate', 'lat':'station_y_coordinate'})
    masked_da = ds

    # Define a function that creates the base map with cartopy features and returns a figure and an axis object
    def make_figure():
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())    # 
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        return fig, ax


    # Call the make_figure function and assign the returned values to fig and ax variables
    fig, ax = make_figure()

    # Define a function that draws the data on the map for a given frame (time index) and optionally adds a colorbar
    def draw(frame, add_colorbar):
        # Clear the previous plot
        ax.clear()
        # Re-add the cartopy features
        ax.coastlines(resolution='10m') 
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        # Plot the grid for the current frame
        grid = masked_da.isel(time=frame)
        # contour = grid.plot(ax=ax, transform=ccrs.PlateCarree(), x='lon', y='lat', cmap = 'Blues', vmin=2, vmax=30, add_colorbar=add_colorbar)
        contour = grid.plot.scatter(
                        x='station_x_coordinate', 
                        y='station_y_coordinate', 
                        hue = 'waterlevel',
                        s=100,
                        edgecolor='none',
                        vmin=-2,
                        vmax=2,
                        transform=ccrs.PlateCarree(),
                        ax=ax,
                        cmap = 'RdBu',
                        cbar_kwargs={'shrink':0.6},
                        add_colorbar=add_colorbar);
        
        title = f"Surge - {str(masked_da.time[frame].values)[:19]}"
        ax.set_title(title)
        return contour

    # Define an init function that calls the draw function for the first frame and adds a colorbar
    def init():
        return draw(0, add_colorbar=True)

    # Define an animate function that calls the draw function for subsequent frames without adding a colorbar
    def animate(frame):
        return draw(frame, add_colorbar=False)

    # Create an animation object using FuncAnimation with the figure, animate, init_func, frames, interval and repeat arguments
    ani = animation.FuncAnimation(fig=fig, func=animate, init_func=init, frames=len(masked_da.time), interval=100, repeat=False)

    # Save the animation as a video file using the writer and fps arguments
    ani.save(file_name)

    plt.close()

    # Return the animation object
    return ani

################################################################################
from scipy.ndimage import laplace

if __name__ == '__main__':
    # Dict of storms
    dict_storms = {
        'xynthia':{'lat':slice(56,38), 'lon':slice(-15,8),'tstart':'2010-02-26','tend':'2010-03-01'}, 
        'sandy':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_shifted':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_historical':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'sandy_era5':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-26','tend':'2012-10-30'},
        'xaver':{'lat':slice(79.99,40), 'lon':slice(-26,23),'tstart':'2013-12-04','tend':'2013-12-06'},
        'idai':{'lat':slice(-10,-30), 'lon':slice(20,55),'tstart':'2019-03-10','tend':'2019-03-16'},
        'irma':{'lat':slice(30,10), 'lon':slice(-90,-60),'tstart':'2017-09-04','tend':'2017-09-14'},
        'andrew':{'lat':slice(30,10), 'lon':slice(-90,-60),'tstart':'1992-08-21','tend':'1992-08-29'},
        }
    

    storm = 'idai_era5'
    variable = 'Ptot_mmh' #Ptot_mmh  mslp_hPa
    scenarios = ["counter_1",
                "counter_2",
                "counter_3",
                "factual_1",
                "factual_2",
                "factual_3",
                "plus2_1",
                "plus2_2",
                "plus2_3" ]
    
    if "historical" in storm or "era5" in storm:
        scenarios = ["historical"]

    for scenario in scenarios:
        if 'shifted' in storm:
            ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.nc')
            ds = ds[[variable.split("_")[0]]]
        
        elif 'historical' in storm:
            ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\historical\CLI_MPI-ESM-XR_t255l95_echam_sandy_{variable}.nc')
        
        elif 'era5' in storm:
            variable_mapping = {'u10': 'u10m', 'v10': 'v10m', 'msl': 'mslp', 'tp': 'Ptot'}
            ds = xr.open_dataset(rf'D:\paper_4\data\era5\era5_hourly_vars_{storm.split("_")[0]}_single_{dict_storms[storm.split("_")[0]]["tstart"].split("-")[0]}_{dict_storms[storm.split("_")[0]]["tstart"].split("-")[1]}.nc')
            ds['msl'] = ds['msl']/100
            ds['tp'] = ds['tp']*1000
            ds = ds.rename(variable_mapping)
            if 'latitude' in ds.coords:
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds = ds[[variable.split("_")[0]]]

        else:
            ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\{variable}\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.{variable}.nc')
     
        ds = ds.sel(time = slice(dict_storms[storm.split("_")[0]]['tstart'], dict_storms[storm.split("_")[0]]['tend']), lat = dict_storms[storm.split("_")[0]]['lat'], lon=dict_storms[storm.split("_")[0]]['lon'])
        
        create_animation(ds, variable = variable.split("_")[0], threshold=2, file_name=rf'D:\paper_4\data\Figures\animations\{storm}\{storm}_{scenario}_{variable}_animation.gif')
    
    # Wind speed animation
    for scenario in scenarios:
        if 'shifted' in storm:
            ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.nc')
            ds_u10 = ds[['u10m']]
            ds_v10 = ds[['v10m']]
        
        elif 'historical' in storm:
            ds_u10 = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\historical\CLI_MPI-ESM-XR_t255l95_echam_sandy_u10m_m_s.nc')
            ds_v10 = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\historical\CLI_MPI-ESM-XR_t255l95_echam_sandy_v10m_m_s.nc')
            
        elif 'era5' in storm:
            variable_mapping = {'u10': 'u10m', 'v10': 'v10m','msl': 'mslp','tp': 'Ptot'}
            ds = xr.open_dataset(rf'D:\paper_4\data\era5\era5_hourly_vars_{storm.split("_")[0]}_single_{dict_storms[storm.split("_")[0]]["tstart"].split("-")[0]}_{dict_storms[storm.split("_")[0]]["tstart"].split("-")[1]}.nc')
            ds['msl'] = ds['msl']/100
            ds['tp'] = ds['tp']*1000
            ds = ds.rename(variable_mapping)
            if 'latitude' in ds.coords:
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
            ds_u10 = ds[['u10m']]
            ds_v10 = ds[['v10m']]

        else:
            ds_u10 = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\u10m_m_s\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.u10m_m_s.nc')
            ds_v10 = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\v10m_m_s\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.v10m_m_s.nc')

        ds_merge = xr.merge([ds_u10, ds_v10])
        prep.U10m_totalwind(ds_merge)

        ds_merge = ds_merge.sel(time = slice(dict_storms[storm.split("_")[0]]['tstart'],dict_storms[storm.split("_")[0]]['tend']), lat = dict_storms[storm.split("_")[0]]['lat'],lon=dict_storms[storm.split("_")[0]]['lon'])

        create_animation(ds_merge, variable = 'U10m', threshold=5, file_name=rf'D:\paper_4\data\Figures\animations\{storm}\{storm}_{scenario}_U10m_animation.gif')

# now plot animations for surge
    dict_storms_surge = {
            'xynthia':{'lat':slice(54,38), 'lon':slice(-15,3),'tstart':'2010-02-26','tend':'2010-02-28'}, 
            'sandy':{'lat':slice(52,20), 'lon':slice(-85,-55),'tstart':'2012-10-28','tend':'2012-10-30'}
            }
    storm = 'sandy_shifted'
    # scenario = 'counter_1'
    scenarios = ["counter_1",
                "counter_2",
                "counter_3",
                "factual_1",
                "factual_2",
                "factual_3",
                "plus2_1",
                "plus2_2",
                "plus2_3" ]

    for scenario in scenarios:
        ds_surge = xr.open_dataset(rf'D:\paper_3\data\gtsm_local_runs\{storm}_fine_grid\gtsm_fine_{storm}_{scenario}_0000_his_waterlevel.nc')
        ds_surge = ds_surge.sel(time = slice(dict_storms_surge[storm.split("_")[0]]['tstart'],dict_storms_surge[storm.split("_")[0]]['tend']))

        create_animation_surge(ds_surge, file_name=rf'D:\paper_3\data\Figures\animations\{storm}\surge_{storm}_{scenario}_animation.gif')


# # FOr shifted storms
#     for variable in ['Ptot_mmh']:
#         for scenario in scenarios:
#             ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\BOT_t_HR255_Nd_SU_1015_{scenario}_2012_{storm}_storm.nc')
#             ds = ds.sel(time = slice(dict_storms[storm]['tstart'],dict_storms[storm]['tend']), lat = dict_storms[storm]['lat'],lon=dict_storms[storm]['lon'])

#             create_animation(ds, variable = variable.split("_")[0], threshold=1, file_name=rf'D:\paper_3\data\Figures\animations\{storm}\{storm}_{scenario}_{variable}_animation.gif')

#     # for scenario in scenarios:
#     #     variable = 'mslp_hPa' #Ptot_mm3h

#     #     ds = xr.open_dataset(rf'D:\paper_3\data\spectral_nudging_data\{storm}\{variable}\BOT_t_HR255_Nd_SU_1015_{scenario}_{dict_storms[storm]["tstart"].split("-")[0]}_{storm}_storm.{variable}.nc')
#     #     ds = ds.sel(time = slice(dict_storms[storm]['tstart'],dict_storms[storm]['tend']), lat = dict_storms[storm]['lat'],lon=dict_storms[storm]['lon'])
            
#     #     # Find laplacian operator for MSLP
#     #     mslp_laplacian = laplace(ds)
#     #     da_mslp_laplacian = xr.DataArray(mslp_laplacian, dims=ds.dims, coords=ds.coords)
#     #     da_mslp_laplacian_mask = da_mslp_laplacian.where(da_mslp_laplacian > da_mslp_laplacian.quantile(q=0.90).values)

#     #     create_animation(ds, variable = variable.split("_")[0], threshold=2, file_name=rf'D:\paper_3\data\Figures\animations\{storm}\{storm}_{scenario}_{variable}_animation.gif')
