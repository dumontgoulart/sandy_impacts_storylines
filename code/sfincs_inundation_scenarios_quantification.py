from calendar import c
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs # for map projections
import cartopy.feature as cfeature
from hydromt_sfincs import SfincsModel, utils

import xarray as xr
import numpy as np
import pandas as pd
import os
import seaborn as sns
import re
import pyproj
from shapely.geometry import box
from shapely.geometry import mapping
import rioxarray
import geopandas as gpd
from scipy.stats import linregress

# plot configurations
# ESD figure sizes
singlecol = 8.3 * 0.393701
doublecol = 14 * 0.393701
fontsize=9

# Add text annotation in figure?
annotation = True
annotation_color = '#333333'
annotation_fontsize=7

plt.rcParams.update({'font.size': 9})

def get_hmax_datasets(storm):
    directory = rf"D:\paper_3\data\sfincs_inundation_results\{storm}\\"
    file_list = [f for f in os.listdir(directory) if f.endswith('.nc')]

    # Initialize an empty list to store datasets
    dataset_list = []

    # Iterate over each file, open it, modify variable name, and append to the dataset list
    for file in file_list:
        # Open the file
        ds = xr.open_dataset(os.path.join(directory, file))

        # Modify variable name
        var_name = 'hmax_' + file[:-3]  # Extract the file name without the extension
        ds = ds.rename_vars({'hmax': var_name})

        # Append dataset to the list
        dataset_list.append(ds)

    # Merge all datasets into a single dataset
    merged_dataset = xr.merge(dataset_list)

    return merged_dataset

def get_total_sum(storm, city_bbox, reduced_size = False):
    # Get a list of all the files in the directory and return a dataframe with the sum of inundation for each file

    directory = rf"D:\paper_3\data\sfincs_inundation_results\{storm}\\"
    file_list = [f for f in os.listdir(directory) if f.endswith('.nc')]


    df = pd.DataFrame(columns=['file_name','n_inundation_cells', 'mean_hmax_city'])

    # Iterate over each file, open it and compute the sum of its values
    for file in file_list:
        # Open the file
        ds = xr.open_dataset(os.path.join(directory, file))
        if reduced_size == True:
            if storm.split('_')[0] == 'sandy':
                # Clip the dataset to the bounding box of New York City    
                sf = gpd.read_file(r'D:\paper_3\data\us_dem\ny_city_shape\ny_city_shape.gpkg')

                ds_clipped = ds.rio.clip(sf.geometry.apply(mapping), sf.crs, drop=True)
                ds = ds_clipped
            
            else:
                epsg = pyproj.CRS(ds['spatial_ref'].attrs['crs_wkt']).to_epsg()

                # Clip the dataset to the bounding box of New York City    
                bbox_crs_ny = convert_latlon_to_xy(city_bbox[0], city_bbox[1], city_bbox[2], city_bbox[3], crs = epsg)

                ds_clipped = ds.where((ds.x >= bbox_crs_ny[0]) & (ds.x <= bbox_crs_ny[2]) & (ds.y >= bbox_crs_ny[1]) & (ds.y <= bbox_crs_ny[3]), drop=True)
                ds = ds_clipped
        # calculate the area of the grid cells:
        dx = ds['x'][1] - ds['x'][0]
        dy = ds['y'][1] - ds['y'][0]

        cell_area = dx * dy

        # Compute the total sum of the values in the dataset
        total_volume = (ds['hmax'] * cell_area).sum().item()

        # Compute the number of grid cells where hmax is >= 0.1
        n_inundation_cells = len(ds['hmax'].to_dataframe()['hmax'].dropna())

        # city mean inundation depth:
        mean_hmax_city = ds['hmax'].mean().item()

        
        # Append the file name and total sum to the dataframe
        df = pd.concat([df, pd.DataFrame.from_records([{'file_name': file, 'n_inundation_cells':n_inundation_cells,  'mean_hmax_city': mean_hmax_city, 'total_volume': total_volume}])])

        # create a regex pattern to capture the relevant parts
    # pattern = rf'^(hmax_forcing_)(rain|surge|rain_surge)(_)(counter|factual|plus2)?(_)(\d)?\.nc$'
    pattern = rf'^(hmax_)({storm})(_)(counter|factual|plus2)?(_)(\d)?(_)(rain|surge|rain_surge)\.nc$'


    # extract the relevant parts into new columns
    df[['storm','GW scenario', 'member', 'component']] = df['file_name'].str.extract(pattern)[[1,3,5,7]]
    df['n_inundation_cells'] = df['n_inundation_cells'].astype(float)
    df['mean_hmax_city'] = df['mean_hmax_city'].astype(float)
    df['total_volume'] = df['total_volume'].astype(float)
    df = df.set_index('file_name')
    print(df.sort_values(by='mean_hmax_city', ascending=False))

    return df

def convert_latlon_to_xy(lon1, lat1, lon2, lat2, crs):
    # Convert a bounding box from lat/lon to a projected coordinate system    
    # Create a shapely geometry object representing the bounding box
    bbox_wgs84 = box(lon1, lat1, lon2, lat2)

    # define the source and target projections
    src_proj = pyproj.Proj(proj="latlong", datum="WGS84")    
    tgt_proj = pyproj.Proj(crs)

    # Convert the bounding box to UTM zone 18N coordinates
    xmin, ymin = pyproj.transform(src_proj, tgt_proj, lon1, lat1)
    xmax, ymax = pyproj.transform(src_proj, tgt_proj, lon2, lat2)

    # Create a new bounding box
    bbox = (xmin, ymin, xmax, ymax)
    return bbox

def regression_lines(data, x,y):
    slope, intercept, r_value, p_value, std_err = linregress(data[x],data[y])
    x = data[x].sort_values()
    y = slope * x + intercept
    return x, y, slope

# parameters
bbox_ny = [-74.025879, 40.700422, -73.907089, 40.878218]
bbox_ny2 = [-74.252472,40.399902,-73.321381,40.837191]
bbox_larochelle = [-1.242828,46.125129,-1.038380,46.213101]
reduced_size = False

# get the datasets for each storm
ds_inundation_sandy = get_hmax_datasets('sandy')
ds_inundation_sandy_shifted = get_hmax_datasets('sandy_shifted')
ds_inundation_sandy_slr71 = get_hmax_datasets('sandy_slr71')
ds_inundation_sandy_slr101 = get_hmax_datasets('sandy_slr101')
ds_inundation_sandy_historical = get_hmax_datasets('sandy_historical')

# get the dataframes for each storm
df_inundation_sandy = get_total_sum('sandy',bbox_ny, reduced_size = reduced_size)
df_inundation_sandy_shifted = get_total_sum('sandy_shifted',bbox_ny, reduced_size = reduced_size)
df_inundation_sandy_slr71 = get_total_sum('sandy_slr71', bbox_ny, reduced_size = reduced_size)
df_inundation_sandy_slr101 = get_total_sum('sandy_slr101', bbox_ny, reduced_size = reduced_size)
df_inundation_sandy_historical = get_total_sum('sandy_historical', bbox_ny, reduced_size = reduced_size)
# variable to be plotted:
variable = 'total_volume' # 'total_volume', mean_hmax_city, n_inundation_cells

# merge the three dataframes over a new column called 'mode':
df_inundation_sandy_slr = pd.concat([df_inundation_sandy, df_inundation_sandy_slr71,df_inundation_sandy_slr101], keys=['normal', 'slr71','slr101'])
df_inundation_sandy_slr = df_inundation_sandy_slr.reset_index().rename(columns={'level_0': 'mode'})

# merge the 4 dataframes over a new column called 'mode':
df_inundation_sandy_allmodes = pd.concat([df_inundation_sandy, df_inundation_sandy_slr71,df_inundation_sandy_slr101,df_inundation_sandy_shifted], keys=['normal', 'slr71','slr101','shifted'])
df_inundation_sandy_allmodes = df_inundation_sandy_allmodes.reset_index().rename(columns={'level_0': 'mode'})


################ paper figures


# open sfincs model with hydromt
mod = SfincsModel(r'd:\paper_3\data\sfincs_ini\spectral_nudging\sandy_base', mode='r')
ds_sfx = mod.grid.copy()
mod.data_catalog.from_yml("d:\paper_3\data\sfincs_ini\spectral_nudging\data_deltares_sandy\data_catalog.yml")
ds_fabdem = mod.data_catalog.get_rasterdataset("fabdem")
ds_fabdem_raster = ds_fabdem.raster.reproject_like(mod.grid)
# ds_dep = ds_sfx['dep'].where(ds_sfx['dep'] >= 0)

# hillshade background
from matplotlib import colors, patheffects
import pyproj
res = 50
da = ds_fabdem_raster
ls = colors.LightSource(azdeg=210, altdeg=45)
hs = ls.hillshade(np.ma.masked_equal(da.values, -9999), vert_exag=1, dx=res, dy=res)
da_hs = xr.DataArray(dims=da.raster.dims, data=hs, coords=da.raster.coords).where(da > 0)

utm_zone = ds_sfx.raster.crs.to_wkt().split("UTM zone ")[1][:3]
utm = ccrs.UTM(int(utm_zone[:2]), "S" in utm_zone)
ylab = f"y coordinate UTM zone {utm_zone} [m]"
xlab = f"x coordinate UTM zone {utm_zone} [m]" 
ratio = ds_sfx.raster.ycoords.size / (ds_sfx.raster.xcoords.size * 1.2)

# convert the bounding box from lat/lon to UTM coordinates
latlon_proj = pyproj.CRS.from_string('EPSG:4326')
transformer = pyproj.Transformer.from_crs(latlon_proj, utm, always_xy=True)
utm_dot1 = list(transformer.transform(bbox_ny2[0], bbox_ny2[1]))
utm_dot2 = list(transformer.transform(bbox_ny2[2], bbox_ny2[3]))
bbox_utm = [utm_dot1[0],utm_dot2[0],utm_dot1[1],utm_dot2[1]]


simple_df = df_inundation_sandy.reset_index().drop(['file_name','n_inundation_cells', 'mean_hmax_city'], axis=1).copy()
rain_df = simple_df[simple_df['component'] == 'rain'].reset_index() #.rename(columns={'component': 'component_rain'})
surge_df = simple_df[simple_df['component'] == 'surge'].reset_index() #.rename(columns={'component': 'component_rain'})
rain_surge_direct_df = rain_df.copy()
rain_surge_direct_df[variable] = rain_df[variable] + surge_df[variable]
rain_surge_direct_df['component'] = 'superposition'
rain_surge_df = simple_df[simple_df['component'] == 'rain_surge'].reset_index()  #.rename(columns={'component': 'component_rain'})
rain_surge_df['component'] = 'Compound'

# create a new dataframe with the compound and superposition scenarios
df_compoudness_comparison = pd.concat([rain_surge_df, rain_surge_direct_df])
df_compoudness_comparison['total_volume'] = df_compoudness_comparison['total_volume'] * (10**-8) 
# Get the indices of the two highest points
highest_indices = np.argsort(df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'])[-2:]


# Figure 2 - create a barplot with all storms that have mode == 'normal' on the X axis and the total volume on the Y axis
import pyproj
utm_proj = pyproj.Proj(proj='utm', zone=18, ellps='WGS84')
manhattan_coords = (-74.0050, 40.712)
staten_island_coords = (-74.1530, 40.5205)  # Geographic coordinates for Staten Island
long_island_coords = (-73.650, 40.6828)  # Geographic coordinates for Long Island
new_jersey_coords = (-74.1690, 40.7458)  # Geographic coordinates for New Jersey

manhattan_utm_coords = utm_proj(manhattan_coords[0], manhattan_coords[1])
staten_island_utm_coords = utm_proj(staten_island_coords[0], staten_island_coords[1])
long_island_utm_coords = utm_proj(long_island_coords[0], long_island_coords[1])
new_jersey_utm_coords = utm_proj(new_jersey_coords[0], new_jersey_coords[1])


# FIGURE 3 
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(doublecol, doublecol*1.4), constrained_layout=True, gridspec_kw={'height_ratios': [0.5, 1, 1]})
x,y, slope = regression_lines(df_compoudness_comparison[df_compoudness_comparison['component'] == 'superposition'], 'total_volume', 'total_volume')
# Add now bar plots of rain_df on the x axis to the previous figure
width = 0.045
# Plot the first figure in the leftmost subplot
# axs[0].plot(x, y, linestyle='--', color='black', label='1:1')
axs[0].bar(df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'], rain_df.total_volume* (10**-8) , width, color='violet', alpha=0.8, label='Univariate: rain')
axs[0].bar(df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'], surge_df.total_volume* (10**-8) , width, bottom=rain_df.total_volume * (10**-8) , color='lightblue', alpha=0.8, label='Univariate: surge')
axs[0].scatter(df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'], df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'], color='maroon', label='Compound')
for x_val, y_val in zip(df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'], df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume']):
    axs[0].plot([x_val, x_val], [0, y_val], color='gainsboro', linestyle = 'dotted', zorder=0)

# Add gray annotations for the two highest points
annotations = ['b', 'c']
for i, index in enumerate(highest_indices):
    x_val = df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'].iloc[index]
    y_val = df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'].iloc[index]
    axs[0].annotate(annotations[i], xy=(x_val, y_val), xytext=(7, -20), textcoords='offset points', color='dimgray', ha='center', va='top',
                   arrowprops=dict(arrowstyle='->', color='dimgray',shrinkB=5,shrinkA=0.1))
# axs[0].set_xticks([])
axs[0].set_xlabel('Baseline scenario flood volume [10⁸ m³]')
axs[0].set_ylabel('Flood volume [10⁸ m³]') # Rain and surge superposition volume [10⁸ m³]
axs[0].set_title(f'a) Compound and univariate flood volumes', loc='left') # Flood volume from combined univariate scenarios
axs[0].set_ylim(0, 1.01*df_compoudness_comparison['total_volume'].max())
axs[0].set_yticks([0, 1, 2])
axs[0].legend(frameon=False, bbox_to_anchor=(-0.0, -0.26), loc='upper left', ncol = 4, columnspacing=0.8)

# Plot the second figure in the middle subplot
da_hs.plot(ax=axs[1], cmap='Greys', add_colorbar=False, alpha=0.2)
cs1 = ds_inundation_sandy['hmax_hmax_sandy_factual_2_rain_surge'].plot(vmin=0, vmax=1, cmap='Blues',add_colorbar=False, ax=axs[1]) # cbar_kwargs=cbar_kwargs, 
axs[1].set_title(f'', loc='center')
axs[1].set_title(f'b) Surge dominated flood', loc='left')
axs[1].set_xlabel('')
axs[1].set_xticklabels([])
# axs[1].set_ylabel('')
# axs[1].set_yticklabels([])

axs[1].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')

axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))

axs[1].text(manhattan_utm_coords[0], manhattan_utm_coords[1], 'Manhattan', ha='center', va='bottom', fontsize=8, color='black')
axs[1].text(staten_island_utm_coords[0], staten_island_utm_coords[1], 'Staten Island', ha='center', va='bottom', fontsize=8, color='black')
axs[1].text(long_island_utm_coords[0], long_island_utm_coords[1], 'Long Island', ha='center', va='bottom', fontsize=8, color='black')
axs[1].text(new_jersey_utm_coords[0], new_jersey_utm_coords[1], 'New Jersey', ha='center', va='bottom', fontsize=8, color='black')

# Plot the third figure in the rightmost subplot
da_hs.plot(ax=axs[2], cmap='Greys', add_colorbar=False, alpha=0.2)
cs2 = ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'].plot(vmin=0, vmax=1, cmap='Blues',add_colorbar=False,  ax=axs[2]) #cbar_kwargs=cbar_kwargs,
axs[2].set_title(f'', loc='center')
axs[2].set_title(f'c) Precipitation dominated flood', loc='left')
# axs[2].set_ylabel('')
# axs[2].set_xticklabels([])
# axs[2].set_yticklabels([])
axs[2].set_xlabel(xlab)
axs[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
axs[2].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')

# adjust ticks and whitespaces
axs[1].set_xticks(axs[1].get_xticks()[::2])
axs[1].set_xlim(bbox_utm[0], bbox_utm[1])
axs[1].set_ylim(bbox_utm[2], bbox_utm[3])
axs[1].xaxis.set_visible(True)
axs[1].yaxis.set_visible(True)
axs[2].set_xticks(axs[2].get_xticks()[::2])
axs[2].set_xlim(bbox_utm[0], bbox_utm[1])
axs[2].set_ylim(bbox_utm[2], bbox_utm[3])
axs[2].xaxis.set_visible(True)
axs[2].yaxis.set_visible(True)

# plt.subplots_adjust(hspace=(0.1)

# Draw the colorbar
cbar_ax = fig.add_axes([1.001, 0.3, 0.02, 0.25])
ticks = np.linspace(0, 1, 3)
ticks[0] = 0
cbar = fig.colorbar(cs2, cax=cbar_ax, orientation='vertical', ticks=ticks, extend='max', label='Flood depth [m]')
cbar_ax.set_yticklabels(ticks)

# Save the figure
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_flood_hazard_map.png', dpi=300, bbox_inches='tight')
plt.show()



######
# SI figure - map of the study area
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(doublecol, doublecol), subplot_kw={'projection': utm} )
# Plot the second figure in the middle subplot
msk_array = np.where(mod.grid.msk == 0, np.nan, mod.grid.msk)
cs = axs.pcolormesh(mod.grid.msk.x, mod.grid.msk.y, msk_array, cmap=colors.ListedColormap(['black', 'white']), alpha=0.4)
# axs.scatter(-74.0060, 40.7128, color='black', marker='o', s=40, transform=ccrs.PlateCarree())
# add state boundaries
axs.add_feature(cfeature.STATES, edgecolor='black', linestyle = 'dashed' , linewidth=0.5)
axs.text(manhattan_utm_coords[0], manhattan_utm_coords[1], 'New York City', ha='center', va='bottom', fontsize=12, color='black')
# axs.text(staten_island_utm_coords[0], staten_island_utm_coords[1], 'Staten Island', ha='center', va='bottom', fontsize=8, color='black')
# axs.text(long_island_utm_coords[0], long_island_utm_coords[1], 'Long Island', ha='center', va='bottom', fontsize=8, color='black')
# axs.text(new_jersey_utm_coords[0], new_jersey_utm_coords[1], 'New Jersey', ha='center', va='bottom', fontsize=8, color='black')

axs.set_xlim(mod.grid.msk.x.min()*0.90, mod.grid.msk.x.max()* 1.2)
axs.set_ylim(mod.grid.msk.y.min()*0.98,mod.grid.msk.y.max()* 1.02)
axs.add_feature(cfeature.OCEAN, facecolor=("lightblue"))
axs.add_feature(cfeature.LAND, color='gainsboro')
axs.add_feature(cfeature.RIVERS, linewidth=0.5)

axs_secondary_y = axs.secondary_yaxis('left', functions=(lambda y: y, lambda y: y))
axs_secondary_y.set_ylabel('y coordinate UTM zone 18N [10⁶ m]')
axs_secondary_y.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
axs_secondary_x = axs.secondary_xaxis('bottom', functions=(lambda x: x, lambda x: x))
axs_secondary_x.set_xlabel(xlab)
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_map_mask.png', dpi=300, bbox_inches='tight')
plt.show()


# SI FIGURE - differences between maximum compound and maximum single driver
maximum_da = xr.where(ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'] > -10000, 0, ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'])
maximum_da = xr.where(ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain'] > maximum_da,
                      ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain'],maximum_da)
maximum_da = xr.where(ds_inundation_sandy['hmax_hmax_sandy_factual_3_surge'] > maximum_da,
                      ds_inundation_sandy['hmax_hmax_sandy_factual_3_surge'],maximum_da)

ds_dif = ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'] - maximum_da
mask = ds_dif.notnull() & maximum_da.notnull()
binary_dif = ds_dif.where(ds_dif >= 0, 1).where(ds_dif < 0, 0).where(mask, np.nan)

# Create a new figure with a single subplot
fig, ax = plt.subplots(figsize=(10, 8.5), subplot_kw={'projection': utm})
da_hs.plot(ax=ax, cmap='Greys', add_colorbar=False, alpha=1)
ds1 = ds_dif.plot(robust=True, cmap='RdBu', add_colorbar=True, ax=ax)
ax.set_title(f'Compound flooding')
ax.set_ylabel(ylab)
ax.set_xlabel(xlab)
ax.set_xticks(ax.get_xticks()[::2])
ax.set_extent(bbox_utm, crs=utm)
plt.show()


volume_ratio = df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume']/df_compoudness_comparison[df_compoudness_comparison['component'] == 'Compound']['total_volume'].min()
print(volume_ratio.sort_values())

## Figure 3 - add to figure 1 the sea level rise scenarios on top of the existing normal bars
columns = ['n_inundation_cells', 'mean_hmax_city', 'total_volume']
df_extra_slr71 = df_inundation_sandy_slr71.copy().reset_index().drop('file_name', axis = 1)
df_extra_slr71[columns] = df_inundation_sandy_slr71.reset_index().drop('file_name', axis = 1)[columns] - df_inundation_sandy.reset_index().drop('file_name', axis = 1)[columns]
df_extra_slr101 = df_inundation_sandy_slr101.copy().reset_index().drop('file_name', axis = 1)
df_extra_slr101[columns] = df_inundation_sandy_slr101.reset_index().drop('file_name', axis = 1)[columns] - df_inundation_sandy.reset_index().drop('file_name', axis = 1)[columns]

df_slr_increase= pd.concat([df_extra_slr71, df_extra_slr101], axis = 0, keys=['slr71','slr101'])
df_slr_increase = df_slr_increase.reset_index().rename(columns={'level_0': 'mode'}).drop('level_1', axis = 1)

# Use pivot to create two columns for variable based on 'mode'
df_combined_slr = df_slr_increase[df_slr_increase['component']=='rain_surge'].pivot(index =["GW scenario", "member" ], columns='mode',
                                values=variable).reset_index()
df_combined_slr = df_combined_slr.rename(columns={'slr71': 'total_volume_slr71', 'slr101': 'total_volume_slr101'})
avg_increase_flood_depth = np.mean(df_slr_increase[df_slr_increase['component'] == 'rain_surge'][df_slr_increase['mode']=='slr101']['mean_hmax_city'] * (10/101) * 100)
print(f"Average increase of {avg_increase_flood_depth} cm of flooding per 10cm of sea level rise")


# Calculate the surge and rain_surge values for the normal and slr71 and slr101 scenarios
df_inundation_slr_comparison = df_inundation_sandy_slr[df_inundation_sandy_slr['component'] == 'rain_surge'].drop(['file_name', 'storm'], axis = 1) 
df_inundation_slr_comparison['total_volume'] = df_inundation_slr_comparison['total_volume'] * (10**-8)
# change names for the paper clarity
names_dict = {'normal': 'baseline', 'slr71': 'SLR71', 'slr101': 'SLR101'}
df_inundation_slr_comparison['mode'] = df_inundation_slr_comparison['mode'].replace(names_dict)
# Create a mask to filter rows where mode == 'baseline'
mask_normal = df_inundation_slr_comparison['mode'] == 'baseline'
# Create a new column 'normal_volume' and assign the total_volume from the rows where mode == 'baseline'
df_inundation_slr_comparison['normal_volume'] = df_inundation_slr_comparison.loc[mask_normal, 'total_volume']
# Get the unique combinations of GW scenario, member, and component for the normal mode
normal_combinations = df_inundation_slr_comparison.loc[mask_normal, ['GW scenario', 'member', 'component']].drop_duplicates()
# Iterate over the normal_combinations and assign the normal_volume to corresponding rows in the dataframe
for _, row in normal_combinations.iterrows():
    gw_scenario = row['GW scenario']
    member = row['member']
    component = row['component']
    mask = (df_inundation_slr_comparison['GW scenario'] == gw_scenario) & (df_inundation_slr_comparison['member'] == member) & (df_inundation_slr_comparison['component'] == component)
    df_inundation_slr_comparison.loc[mask, 'normal_volume'] = df_inundation_slr_comparison.loc[mask_normal & (df_inundation_slr_comparison['GW scenario'] == gw_scenario) & (df_inundation_slr_comparison['member'] == member) & (df_inundation_slr_comparison['component'] == component), 'total_volume'].item()

# regression lines
x_normal, y_normal, slope_normal = regression_lines(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'baseline'], 'normal_volume', 'total_volume')
x_slr71, y_slr71, slope_slr71 = regression_lines(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR71'], 'normal_volume', 'total_volume')
x_slr101, y_slr101, slope_slr101 = regression_lines(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101'], 'normal_volume', 'total_volume')

# surge volumes
df_surges_slr = df_inundation_sandy_slr[df_inundation_sandy_slr['component'] == 'surge'].drop(['file_name', 'storm'], axis = 1)
df_surges_slr['total_volume'] = df_surges_slr['total_volume'] * (10**-8)
df_surges_slr71_dif = df_surges_slr[df_surges_slr['mode'] == 'slr71'].reset_index()['total_volume'] - df_surges_slr[df_surges_slr['mode'] == 'normal'].reset_index()['total_volume']
df_surges_slr101_dif = df_surges_slr[df_surges_slr['mode'] == 'slr101'].reset_index()['total_volume'] - df_surges_slr[df_surges_slr['mode'] == 'slr71'].reset_index()['total_volume']

# # Figure 2 - plot the difference between the normal and slr71 and slr101 scenarios
# fig, ax = plt.subplots(figsize=(doublecol, doublecol*0.7))
# custom_palette = ['lightblue', 'royalblue', 'darkslateblue']
# alpha = 0.8
# for x_val, y_val in zip(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['normal_volume'], df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['total_volume']):
#     ax.vlines(x=x_val, ymin=0, ymax=y_val, color='gainsboro', linestyle='dotted', zorder=0)

# ax.bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['normal_volume'], df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'].values, width*0.8, color=custom_palette[0], alpha=alpha, label='baseline surge')
# ax.bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['normal_volume'], df_surges_slr71_dif, width*0.8, bottom = df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'], color=custom_palette[1], alpha=alpha, label='SLR71 surge')
# ax.bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['normal_volume'], df_surges_slr101_dif, width*0.8, bottom = df_surges_slr71_dif + df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'].values, color=custom_palette[2], alpha=alpha, label='SLR101 surge')

# plt.plot(x_normal, y_normal, color=custom_palette[0], linestyle='dashed', alpha=alpha, label=f'baseline (coeff {slope_normal})')
# plt.plot(x_slr71, y_slr71, color=custom_palette[1], linestyle='dashed', alpha=alpha, label=f'SLR71 (coeff {round(slope_slr71,2)})')
# plt.plot(x_slr101, y_slr101, color=custom_palette[2], linestyle='dashed', alpha=alpha, label=f'SLR101 (coeff {round(slope_slr101,2)})')

# sns.scatterplot(data=df_inundation_slr_comparison, x='normal_volume', y='total_volume', style='mode', s=60, hue='mode', edgecolor='black', palette=custom_palette)

# plt.title('Increase in flood volume due to SLR')
# plt.xlabel('Flood volume [10⁸ m³]')
# plt.ylabel('Flood volume [10⁸ m³]')
# plt.ylim(0, 1.04*df_inundation_slr_comparison['total_volume'].max())

# # plot legend outside and below the figure
# plt.legend(frameon=False, bbox_to_anchor=(0.1, -0.11), loc='upper left', ncol=3, columnspacing=0.1, handletextpad=0.1, borderpad=0.1)

# plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_slr_scenarios.png', dpi=300, bbox_inches='tight')
# plt.show()

# Paper figure 4 - with flood mapping
ds_sandy_surge_dif_slr101 = ds_inundation_sandy_slr101['hmax_hmax_sandy_slr101_factual_2_rain_surge'] - ds_inundation_sandy['hmax_hmax_sandy_factual_2_rain_surge'].where(ds_inundation_sandy['hmax_hmax_sandy_factual_2_rain_surge'] >= 0, 0)

df_inundation_slr_comparison['mode'] = df_inundation_slr_comparison['mode'].replace({'baseline': 'baseline (compound)', 'SLR71': 'SLR71 (compound)', 'SLR101': 'SLR101 (compound)'})

# Create a new figure with three subplots horizontally
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(doublecol, doublecol*1.4))
# Plot the first figure in the leftmost subplot
custom_palette = ['lightblue', 'royalblue', 'darkslateblue']
alpha = 0.8
for x_val, y_val in zip(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['normal_volume'], df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101']['total_volume']):
    axs[0].vlines(x=x_val, ymin=0, ymax=y_val, color='gainsboro', linestyle='dotted', zorder=0)

axs[0].bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101 (compound)']['normal_volume'], df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'].values, width*0.8, color=custom_palette[0], alpha=alpha, label='baseline (surge)')
axs[0].bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101 (compound)']['normal_volume'], df_surges_slr71_dif, width*0.8, bottom = df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'], color=custom_palette[1], alpha=alpha, label='SLR71 (surge)')
axs[0].bar(df_inundation_slr_comparison[df_inundation_slr_comparison['mode'] == 'SLR101 (compound)']['normal_volume'], df_surges_slr101_dif, width*0.8, bottom = df_surges_slr71_dif + df_surges_slr[df_surges_slr['mode'] == 'normal']['total_volume'].values, color=custom_palette[2], alpha=alpha, label='SLR101 (surge)')

axs[0].plot(x_normal, y_normal, color=custom_palette[0], linestyle='dashed', alpha=alpha, label=f'baseline (coeff {round(slope_normal,2)})') # 
axs[0].plot(x_slr71, y_slr71, color=custom_palette[1], linestyle='dashed', alpha=alpha, label=f'SLR71 (coeff {round(slope_slr71,2)})') # 
axs[0].plot(x_slr101, y_slr101, color=custom_palette[2], linestyle='dashed', alpha=alpha, label=f'SLR101 (coeff {round(slope_slr101,2)})') #  

sns.scatterplot(data=df_inundation_slr_comparison, x='normal_volume', y='total_volume', style='mode', s=60, hue='mode', edgecolor='black', palette=custom_palette, ax=axs[0])

axs[0].set_title('a) Compound flood volumes for sea level rise (SLR) scenarios', loc = 'left')
axs[0].set_xlabel('Baseline scenario flood volume [10⁸ m³]') #axs[0].set_xlabel('Storylines in ascending order of flood volume')
axs[0].set_ylabel('Flood volume [10⁸ m³]')
axs[0].set_ylim(0, 1.04*df_inundation_slr_comparison['total_volume'].max())
# plot legend outside of the plot
axs[0].legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')

# Add gray annotations for the two highest points
annotations = ['b']
x_val = df_inundation_slr_comparison.loc[67]['normal_volume']
y_val = df_inundation_slr_comparison.loc[67]['total_volume']
axs[0].annotate('b', xy=(x_val, y_val), xytext=(7, -20), textcoords='offset points', color='gray', ha='center', va='top', arrowprops=dict(arrowstyle='->', color='gray',shrinkB=5,shrinkA=0.1))

# Plot the second figure in the middle subplot
da_hs.plot(ax=axs[1], cmap='Greys', add_colorbar=False, alpha=0.2)
cslr = ds_sandy_surge_dif_slr101.plot(vmin=-1, vmax=1, cmap='RdBu',add_colorbar=False, ax=axs[1]) # cbar_kwargs=cbar_kwargs, 
axs[1].set_title(f'', loc='center')
axs[1].set_title(f'b) Changes in flood due to SLR101', loc='left')
axs[1].set_xlabel('')
# axs[1].set_xticklabels([])

# adjust ticks and whitespaces
# axs[1].set_xticks([])
axs[1].set_xlim(bbox_utm[0], bbox_utm[1])
axs[1].set_ylim(bbox_utm[2], bbox_utm[3])
axs[1].yaxis.set_visible(True)
axs[1].set_xticks(axs[1].get_xticks()[::2])

axs[1].set_xlabel(xlab)
axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
axs[1].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')

plt.subplots_adjust(hspace=0.3)

# Draw the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.25])
ticks = np.linspace(-1, 1, 5)
ticks[0] = -1
cbar = fig.colorbar(cslr, cax=cbar_ax, orientation='vertical', ticks=ticks, extend='max', label='Change in flood depth [m]')
cbar_ax.set_yticklabels(ticks)

plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_slr_scenarios_map.png', dpi=300, bbox_inches='tight')
plt.show()




###############################
# FIgure 5 - add now the shifted versions and compare with the normal ones
df_inundation_sandy_comparison_shift = pd.concat([df_inundation_sandy, df_inundation_sandy_shifted], keys=['normal', 'shifted'])
df_inundation_sandy_comparison_shift = df_inundation_sandy_comparison_shift.reset_index().rename(columns={'level_0': 'mode'})

df_extra_shifted = df_inundation_sandy_shifted.copy().reset_index().drop('file_name', axis = 1)
df_extra_shifted[columns] = (df_inundation_sandy_shifted.reset_index().drop('file_name', axis = 1)[columns] - df_inundation_sandy.reset_index().drop('file_name', axis = 1)[columns]) / df_inundation_sandy.reset_index().drop('file_name', axis = 1)[columns]


# # option c
# # plot a figure showing the extra shifted values on the y axis and the shifting distance csv on the x axis
# df_shifting_distance = pd.read_csv('D:\paper_3\data\shifting_distance_sandy.csv', index_col=0).reset_index()
# df_extra_shifted_precip = df_extra_shifted[df_extra_shifted['component']=='rain'].copy().reset_index()
# df_extra_shifted_precip['shifting distance'] = df_shifting_distance['shifting distance']
# sns.regplot(x='shifting distance', y=variable, data = df_extra_shifted_precip, x_ci = 'sd')
# plt.xlabel('Shifting Distance')
# plt.ylabel('Increase in flooding volume')
# plt.title('Shifting Distance vs. Total Volume')
# # Add grid lines
# plt.grid(True)
# # Add legend
# plt.legend()
# # Show the plot
# plt.show()


# option d - scatter plot
df_inundation_sandy_comparison_shift_precip_surge = df_inundation_sandy_comparison_shift[df_inundation_sandy_comparison_shift['component']=='rain_surge'].copy().reset_index()
df_inundation_sandy_comparison_shift_precip_surge['total_volume'] = df_inundation_sandy_comparison_shift_precip_surge['total_volume'] * (10**-8)
# change names for the paper clarity
names_dict_mp = {'normal': 'baseline', 'shifted': 'MP'}
df_inundation_sandy_comparison_shift_precip_surge['mode'] = df_inundation_sandy_comparison_shift_precip_surge['mode'].replace(names_dict_mp)

# Create a mask to filter rows where mode == 'baseline'
mask_normal = df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline'
# Create a new column 'normal_volume' and assign the total_volume from the rows where mode == 'baseline'
df_inundation_sandy_comparison_shift_precip_surge['normal_volume'] = df_inundation_sandy_comparison_shift_precip_surge.loc[mask_normal, 'total_volume']
# Get the unique combinations of GW scenario, member, and component for the normal mode
normal_combinations = df_inundation_sandy_comparison_shift_precip_surge.loc[mask_normal, ['GW scenario', 'member', 'component']].drop_duplicates()
# Iterate over the normal_combinations and assign the normal_volume to corresponding rows in the dataframe
for _, row in normal_combinations.iterrows():
    gw_scenario = row['GW scenario']
    member = row['member']
    component = row['component']
    mask = (df_inundation_sandy_comparison_shift_precip_surge['GW scenario'] == gw_scenario) & (df_inundation_sandy_comparison_shift_precip_surge['member'] == member) & (df_inundation_sandy_comparison_shift_precip_surge['component'] == component)
    df_inundation_sandy_comparison_shift_precip_surge.loc[mask, 'normal_volume'] = df_inundation_sandy_comparison_shift_precip_surge.loc[mask_normal & (df_inundation_sandy_comparison_shift_precip_surge['GW scenario'] == gw_scenario) & (df_inundation_sandy_comparison_shift_precip_surge['member'] == member) & (df_inundation_sandy_comparison_shift_precip_surge['component'] == component), 'total_volume'].item()

x_normal, y_normal, slope_normal = regression_lines(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline'], 'normal_volume', 'total_volume')
x_shifted, y_shifted, slope_shifted = regression_lines(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP'], 'normal_volume', 'total_volume')

# precipitation volumes
df_precip_shifted = df_inundation_sandy_comparison_shift[df_inundation_sandy_comparison_shift['component'] == 'rain'].drop(['file_name', 'storm'], axis = 1)
df_precip_shifted['total_volume'] = df_precip_shifted['total_volume'] * (10**-8)
df_precip_shifted_dif = df_precip_shifted[df_precip_shifted['mode'] == 'shifted'].reset_index()['total_volume'] - df_precip_shifted[df_precip_shifted['mode'] == 'normal'].reset_index()['total_volume']

# ratio of increase
shifted_ratio = df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['total_volume'].values/df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline']['normal_volume'].values
# calculate the coefficient of variation
from scipy.stats import variation
variation_normal = variation(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline']['total_volume'])
variation_shifted = variation(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['total_volume'])

# FIGURE 4
fig, ax = plt.subplots(figsize=(doublecol, doublecol*0.7))
custom_palette = ['violet', 'purple']
alpha = 0.8
width = 0.05

# for x_val, y_val in zip(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['normal_volume'], df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['total_volume']):
#     ax.vlines(x=x_val, ymin=0, ymax=y_val, color='gainsboro', linestyle='dotted', zorder=0)

# ax.bar(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline']['normal_volume'], df_precip_shifted[df_precip_shifted['mode'] == 'normal']['total_volume'].values, width*0.8, color=custom_palette[0], alpha = alpha, label='baseline precip.')
# ax.bar(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline']['normal_volume'], df_precip_shifted_dif, width*0.8, bottom = df_precip_shifted[df_precip_shifted['mode'] == 'normal']['total_volume'].values, color=custom_palette[1], alpha=alpha, label='MP precip.')

# plt.plot(x_normal, y_normal, color=custom_palette[0],alpha=alpha, linestyle='dashed', label = 'baseline (coeff {})'.format(round(slope_normal,2)))
# plt.plot(x_shifted, y_shifted, color=custom_palette[1],alpha=alpha, linestyle='dashed', label = 'MP (coeff {})'.format(round(slope_shifted,2)))

# sns.scatterplot(data=df_inundation_sandy_comparison_shift_precip_surge, x='normal_volume', y='total_volume', style = 'mode',alpha=1, s=60, hue='mode', palette = custom_palette, edgecolor='black')

# plt.title('Increase in flood volume due to maximum precipitation')
# plt.xlabel('Flood volume [10⁸ m³]')
# plt.ylabel('Flood volume [10⁸ m³]')
# plt.ylim(0, 1.04*df_inundation_sandy_comparison_shift_precip_surge['total_volume'].max())
# # plot legend outside of the plot
# plt.legend(frameon=False, bbox_to_anchor=(0.1, -0.11), loc='upper left', ncol=3, columnspacing=0.1, handletextpad=0.1, borderpad=0.1)
# plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_track_manipulation_scenarios.png', dpi=300, bbox_inches='tight')
# plt.show()



ds_sandy_surge_dif_fac_3_mp = ds_inundation_sandy_shifted['hmax_hmax_sandy_shifted_factual_3_rain_surge'] - ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'].where(ds_inundation_sandy['hmax_hmax_sandy_factual_3_rain_surge'] >= 0, 0)
ds_sandy_surge_dif_fac_1_mp = ds_inundation_sandy_shifted['hmax_hmax_sandy_shifted_factual_1_rain_surge'] - ds_inundation_sandy['hmax_hmax_sandy_factual_1_rain_surge'].where(ds_inundation_sandy['hmax_hmax_sandy_factual_1_rain_surge'] >= 0, 0)
ds_sandy_dif_counter_3_mp = ds_inundation_sandy_shifted['hmax_hmax_sandy_shifted_counter_3_rain_surge'] - ds_inundation_sandy['hmax_hmax_sandy_counter_3_rain_surge'].where(ds_inundation_sandy['hmax_hmax_sandy_counter_3_rain_surge'] >= 0, 0)

# change df_inundation_sandy_comparison_shift_precip_surge mode to include (compound) after
df_inundation_sandy_comparison_shift_precip_surge['mode'] = df_inundation_sandy_comparison_shift_precip_surge['mode'].replace({'baseline': 'baseline (compound)', 'MP': 'MP (compound)'})

# FIGURE 4
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(doublecol, doublecol*1.4))

# First subplot
for x_val, y_val in zip(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['normal_volume'], df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'MP']['total_volume']):
    axs[0].vlines(x=x_val, ymin=0, ymax=y_val, color='gainsboro', linestyle='dotted', zorder=0)

axs[0].bar(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline (compound)']['normal_volume'], df_precip_shifted[df_precip_shifted['mode'] == 'normal']['total_volume'].values, width*0.8, color=custom_palette[0], alpha = alpha, label='baseline (precip.)')
axs[0].bar(df_inundation_sandy_comparison_shift_precip_surge[df_inundation_sandy_comparison_shift_precip_surge['mode'] == 'baseline (compound)']['normal_volume'], df_precip_shifted_dif, width*0.8, bottom = df_precip_shifted[df_precip_shifted['mode'] == 'normal']['total_volume'].values, color=custom_palette[1], alpha=alpha, label='MP (precip.)')

# axs[0].plot(x_normal, y_normal, color=custom_palette[0],alpha=alpha, linestyle='dashed', label = 'baseline (coeff {})'.format(round(slope_normal,2)))
# axs[0].plot(x_shifted, y_shifted, color=custom_palette[1],alpha=alpha, linestyle='dashed', label = 'MP (coeff {})'.format(round(slope_shifted,2)))
# axs[0].axhspan(3.7, 4.3, alpha=0.1, color='purple', zorder=0,linestyle='dashed', linewidth=1.5, label = 'Range of most MP runs')

sns.scatterplot(data=df_inundation_sandy_comparison_shift_precip_surge, x='normal_volume', y='total_volume', style = 'mode',alpha=1, s=60, hue='mode', palette = custom_palette, edgecolor='black', ax=axs[0])

axs[0].set_title('a) Compound flood volumes for maximised precipitation (MP) scenario', loc='left')
axs[0].set_xlabel('Baseline scenario flood volume [10⁸ m³]')
axs[0].set_ylabel('Flood volume [10⁸ m³]')
axs[0].set_ylim(0, 1.04*df_inundation_sandy_comparison_shift_precip_surge['total_volume'].max())
# plot legend outside of the plot
axs[0].legend(frameon=False, bbox_to_anchor=(1,1), loc='upper left', ncol=1, columnspacing=0.1, handletextpad=0.1, borderpad=0)

# Add gray annotations for the two highest points
annotations = ['b']
x_val = df_inundation_sandy_comparison_shift_precip_surge.loc[14]['normal_volume']
y_val = df_inundation_sandy_comparison_shift_precip_surge.loc[14]['total_volume']
axs[0].annotate('b', xy=(x_val, y_val), xytext=(7, -20), textcoords='offset points', color='gray', ha='center', va='top', arrowprops=dict(arrowstyle='->', color='gray',shrinkB=5,shrinkA=0.1))


# Second subplot
da_hs.plot(ax=axs[1], cmap='Greys', add_colorbar=False, alpha=0.2)
cmp = ds_sandy_surge_dif_fac_3_mp.plot(vmin=-0.5, vmax=0.5, cmap='RdBu',add_colorbar=False, ax=axs[1]) # cbar_kwargs=cbar_kwargs, 
axs[1].set_title(f'', loc='center')
axs[1].set_title(f'b) Changes in flood due to MP', loc='left')
# axs[1].set_xlabel('')
# axs[1].set_xticklabels([])
axs[1].set_ylabel('')

# adjust ticks and whitespaces
# axs[1].set_xticks([])
axs[1].set_xlim(bbox_utm[0], bbox_utm[1])
axs[1].set_ylim(bbox_utm[2], bbox_utm[3])
axs[1].yaxis.set_visible(True)

axs[1].set_xlabel(xlab)
axs[1].set_xticks(axs[1].get_xticks()[::2])
axs[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
axs[1].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')

# add a horizontal space between the  two subplots
plt.subplots_adjust(hspace=0.3)
# Draw the colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.25])
ticks = np.linspace(-0.5, 0.5, 5)    
ticks[0] = -0.5
cbar = fig.colorbar(cmp, cax=cbar_ax, orientation='vertical', ticks=ticks, extend='max', label='Change in flood depth [m]')
cbar_ax.set_yticklabels(ticks)

plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_track_manipulation_scenarios_map.png', dpi=300, bbox_inches='tight')
plt.show()








# ############################################
# # Xynthia
# df_inundation_xynthia = get_total_sum('xynthia', bbox_larochelle)


# # Create a barplot with error bars and a legend 
# g = sns.catplot(x='GW scenario', y='total_volume', hue='member', col='component', col_order=['rain', 'surge', 'rain_surge'], data=df_inundation_xynthia, kind='bar', height=5, aspect=0.7, palette='colorblind',margin_titles=True)

# g.set_titles("{col_name}")
# # g.legend_.remove()
# # Show the plot
# plt.show()


