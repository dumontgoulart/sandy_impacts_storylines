import os,sys
import glob
import matplotlib
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import shapely
import pandas as pd
import xarray as xr
import dask_geopandas
from tqdm import tqdm 
import seaborn as sns
import matplotlib.pyplot as plt
os.chdir('D:/paper_3/code')
import ci_functions as cif
# Developed by E. Koks and adapted by H. Goulart, 2023 

# Set the input data path
input_data = rf'D:\paper_3\data\osm_data'
inundation_data_path = rf'D:\paper_3\data\sfincs_inundation_results'
storm = 'sandy'

### Load the netcdf files with inundation data
# Construct the pattern to search for subdirectories containing "sandy"
subdir_pattern = os.path.join(inundation_data_path, f"{storm}*")
# Use glob to find all subdirectories matching the pattern
matching_subdirs = glob.glob(subdir_pattern)
# Initialize a list to store the paths of the .nc files
netcdfs = []
# Iterate over the matching subdirectories
for subdir in matching_subdirs:
    if "sandy_historical" in subdir:
        continue
    # Construct the pattern to search for .nc files within each subdirectory
    file_pattern = os.path.join(subdir, "*_rain_surge.nc")
    # Use glob to find all .nc files within the subdirectory
    matching_files = glob.glob(file_pattern)
    # Extend the netcdfs list with the matching file paths
    netcdfs.extend(matching_files)

### Read single raster file to make sure we can clip the exposure data
raster_in = os.path.join(input_data,netcdfs[0])
vector_flood = cif.raster_to_vector(xr.open_dataset(raster_in, engine="netcdf4"))

# set crs
vector_flood = vector_flood.set_crs(32618)
vector_flood = vector_flood.to_crs(4326)    

### Read exposure data
cis_all = gpd.read_parquet(os.path.join(input_data,'us-northeast_cis.parquet'))
buildings = gpd.read_parquet(os.path.join(input_data,'buildings.parquet'))
ny_buildings = buildings.clip(tuple(vector_flood.total_bounds)).reset_index(drop=True)



# Actual processing of data and obtaining exposure of CI
### Overlay buildings with flood information
building_exposure = {}
for netcdf in tqdm(netcdfs,total=len(netcdfs)):
    flood_map = os.path.join(input_data,netcdf)
    building_exposure[netcdf.split('.')[0]] = cif.building_flood_intersect(ny_buildings,flood_map)

### Overlay critical infrastructure with flood information
cis = ['healthcare','education','oil & gas','telecom','water','wastewater','power','rail','road','air']

cis_exposure = {}
for i_cis in cis:
    if i_cis == 'oil & gas':
        sub_cis = cis_all.loc[['oil','gas']].reset_index().drop(['level_0','level_1'], axis =1)
    else:
        sub_cis = cis_all.loc[i_cis] 
    ny_sub_cis = sub_cis.clip(tuple(vector_flood.total_bounds)).reset_index(drop=True)
    ny_sub_cis = ny_sub_cis.to_crs(32618)    
    print(i_cis)
    for netcdf in tqdm(netcdfs,total=len(netcdfs)):
        flood_map = os.path.join(input_data,netcdf)
        cis_exposure[(netcdf.split('.')[0],i_cis)] = cif.cis_flood_intersect(ny_sub_cis,flood_map)

# test = cis_exposure[('D:\\paper_3\\data\\sfincs_inundation_results\\sandy_slr101\\hmax_sandy_slr101_factual_1_rain_surge','oil & gas')]

################################################################################
# Create datasets of CI with exposure and flooded levels - use for plots
################################################################################
# combine cis_exposure and building_exposure into a third dictionary
# divide CI in categories
index_mapping = {
    'sandy': 'baseline',
    'sandy_shifted': 'MP',
    'sandy_slr71': 'SLR71',
    'sandy_slr101': 'SLR101'
}

building_exposure_upd = {
    (key, 'buildings'): value 
    for key, value in building_exposure.items()
}

cis_build_exposure = {}
cis_build_exposure.update(building_exposure_upd)
cis_build_exposure.update(cis_exposure)    

### 1 - Create the dataset for exposed buildings per scenario and CI system
scenarios = []
ci_systems = []
exposed_buildings = []

# Iterate over the keys of the cis_exposure dictionary
for key in cis_build_exposure.keys():
    scenario, ci_system = key
    exposed_building_count = len(cis_build_exposure[key])

    # Append the values to the corresponding lists
    scenarios.append(scenario)
    ci_systems.append(ci_system)
    exposed_buildings.append(exposed_building_count)

# Create the new dataframe
data = {'scenario': scenarios,
    'CI system': ci_systems,
    'exposed buildings': exposed_buildings}
df_exposed_ci = pd.DataFrame(data)

# Replace the 'CI system' values using the mapping dictionary
# df_exposed_ci['CI system'] = df_exposed_ci['CI system'].replace(ci_mapping)

# Groupby 'scenario' and 'CI system' and sum the 'exposed buildings'
df_exposed_ci = df_exposed_ci.groupby(['scenario', 'CI system'])['exposed buildings'].sum().reset_index()



### Pivot the dataframe and extract the sum of exposed buildings per scenario and CI system
df_exposed_ci_pivot = df_exposed_ci.pivot(index='scenario', columns='CI system', values='exposed buildings')
df_exposed_ci_pivot = df_exposed_ci_pivot.assign(Sum=df_exposed_ci_pivot.sum(axis=1))
# Reset the index and rename the columns
df_exposed_ci_pivot = df_exposed_ci_pivot.reset_index().rename_axis(None, axis=1)
# Extract the "mode" from the "scenario" column
df_exposed_ci_pivot['mode'] = df_exposed_ci_pivot['scenario'].str.extract('hmax_(.*?)_(?:plus2|counter|factual)')
df_exposed_ci_pivot['member'] = df_exposed_ci_pivot.groupby('mode').cumcount() + 1

### 2- Create the dataset for mean flooded levels per scenario and CI system
scenarios = []
ci_systems = []
mean_flooded_levels = []

# Iterate over the keys of the cis_build_exposure dictionary
for key in cis_build_exposure.keys():
    scenario, ci_system = key
    flooded_levels = cis_build_exposure[key]['flood_data']
    mean_flooded_level = flooded_levels.mean()

    # Append the values to the corresponding lists
    scenarios.append(scenario)
    ci_systems.append(ci_system)
    mean_flooded_levels.append(mean_flooded_level)

# Create the new dataframe
data_mean_levels = {
    'scenario': scenarios,
    'CI system': ci_systems,
    'mean flooded level': mean_flooded_levels
}
df_mean_flooded_levels = pd.DataFrame(data_mean_levels)
# Create a pivot table of mean flooded levels by scenario and CI system
df_mean_flooded_levels_pivot = df_mean_flooded_levels.pivot(index='scenario', columns='CI system', values='mean flooded level')
df_mean_flooded_levels_pivot = df_mean_flooded_levels_pivot.assign(total_mean=df_mean_flooded_levels_pivot.mean(axis=1))
# Reset the index and rename the columns
df_mean_flooded_levels_pivot = df_mean_flooded_levels_pivot.reset_index().rename_axis(None, axis=1)
# Extract the "mode" from the "scenario" column
df_mean_flooded_levels_pivot['mode'] = df_mean_flooded_levels_pivot['scenario'].str.extract('hmax_(.*?)_(?:plus2|counter|factual)')
df_mean_flooded_levels_pivot['member'] = df_mean_flooded_levels_pivot.groupby('mode').cumcount() + 1


### 3- Create a new dataframe with the scenario and CI system columns based on water level categories
df_exposed_ci_cat = df_exposed_ci[['scenario', 'CI system']].copy()
# Initialize the total count columns
total_count_0_0_4m = []
total_count_0_4_1m = []
total_count_above_1m = []

# Plot histogram of buildings['flood_data']
buildings.flood_data.hist(bins=100)
# add x labels every 0.5
plt.xticks(np.arange(0, 2.5, 0.15))
plt.show()


# Iterate over the keys of the cis_build_exposure dictionary
for scenario, ci_system in zip(df_exposed_ci['scenario'].values, df_exposed_ci['CI system'].values):
    
    # Get the buildings for the current scenario and CI system
    buildings = cis_build_exposure[(scenario, ci_system)]
    
    # Count the buildings for each water level category
    count_0_0_4m = ((buildings['flood_data'] >= 0.15) & (buildings['flood_data'] < 0.5)).sum()
    count_0_4_1m = ((buildings['flood_data'] >= 0.5) & (buildings['flood_data'] < 1)).sum()
    count_above_1m = (buildings['flood_data'] >= 1).sum()
    
    # Append the total counts for the scenario
    total_count_0_0_4m.append(count_0_0_4m)
    total_count_0_4_1m.append(count_0_4_1m)
    total_count_above_1m.append(count_above_1m)

# Add the total count columns to the dataframe
df_exposed_ci_cat['Total buildings (0.15-0.5m)'] = total_count_0_0_4m
df_exposed_ci_cat['Total buildings (0.5-1m)'] = total_count_0_4_1m
df_exposed_ci_cat['Total buildings (>1m)'] = total_count_above_1m

# Calculate the sum of the counts for each scenario
df_exposed_ci_cat['Sum'] = df_exposed_ci_cat.iloc[:, 2:].sum(axis=1)

# create a new dataframe based on df_exposed_ci_cat where we have the mode column and the member column. 
df_exposed_ci_cat_scenarios = df_exposed_ci_cat.copy()
df_exposed_ci_cat_scenarios['mode'] = df_exposed_ci_cat['scenario'].str.extract('hmax_(.*?)_(?:plus2|counter|factual)')
# create a new column called member that counts the number of members in each CI system and mode
df_exposed_ci_cat_scenarios['member'] = df_exposed_ci_cat_scenarios.groupby(['CI system', 'mode']).cumcount() + 1
# Then get the mean of the total buildings in each level by the CI system and mode.
df_exposed_ci_cat_mean = df_exposed_ci_cat_scenarios.drop('scenario', axis=1).groupby(['CI system', 'mode']).median()#.reset_index()
df_exposed_ci_cat_mean.reset_index(inplace=True)
df_exposed_ci_cat_mean.set_index('mode', inplace=True)
#update index names index_mapping
df_exposed_ci_cat_mean.index = df_exposed_ci_cat_mean.index.map(index_mapping) 

### Create a new dataframe with total exposed buildings (combining all CI systems) per water level category for each scenario 
df_exposed_scenarios = pd.DataFrame({'scenario': df_exposed_ci_cat['scenario']})
# Initialize the total count columns
total_count_0_0_4m = []
total_count_0_4_1m = []
total_count_above_1m = []
total_count_sum = []

# Iterate over the scenarios in df_exposed_ci_cat
for scenario in df_exposed_ci_cat['scenario']:
    # Filter the rows in df_exposed_ci_cat for the current scenario
    scenario_rows = df_exposed_ci_cat[df_exposed_ci_cat['scenario'] == scenario]
    
    # Calculate the sum of counts for each category across all CI systems
    count_0_0_4m = scenario_rows['Total buildings (0.15-0.5m)'].sum()
    count_0_4_1m = scenario_rows['Total buildings (0.5-1m)'].sum()
    count_above_1m = scenario_rows['Total buildings (>1m)'].sum()
    count_sum = scenario_rows['Sum'].sum()
    
    # Append the total counts for the scenario
    total_count_0_0_4m.append(count_0_0_4m)
    total_count_0_4_1m.append(count_0_4_1m)
    total_count_above_1m.append(count_above_1m)
    total_count_sum.append(count_sum)

# Add the total count columns to the dataframe
df_exposed_scenarios['Total buildings (0.15-0.5m)'] = total_count_0_0_4m
df_exposed_scenarios['Total buildings (0.5-1m)'] = total_count_0_4_1m
df_exposed_scenarios['Total buildings (>1m)'] = total_count_above_1m
df_exposed_scenarios['Sum'] = total_count_sum

# Drop duplicate rows to keep only unique scenarios
df_exposed_scenarios = df_exposed_scenarios.drop_duplicates().reset_index(drop=True)
# Extract the "mode" from the "scenario" column
df_exposed_scenarios['mode'] = df_exposed_scenarios['scenario'].str.extract('hmax_(.*?)_(?:plus2|counter|factual)')
df_exposed_scenarios['member'] = df_exposed_scenarios.groupby('mode').cumcount() + 1



################################################################################
# Generate plots
################################################################################


desired_order = ['baseline', 'MP', 'SLR71', 'SLR101']

df_exposed_scenarios['mode'] = df_exposed_scenarios['mode'].replace(index_mapping)
# convert df_exposed_scenarios so that total buildings are values and a column "water category" is added to show each category
df_exposed_scenarios_test = df_exposed_scenarios.melt(id_vars=['scenario', 'mode', 'member', 'Sum'], value_vars=['Total buildings (0.15-0.5m)', 'Total buildings (0.5-1m)', 'Total buildings (>1m)'], var_name='water category', value_name='Buildings')
# now plot box plots of df_exposed_scenarios_test per mode and have the water category as hue and make sure the hue color palette is based on custom_palette:
custom_palette = ['salmon', 'brown', 'maroon']

df_ci_flood_0_0_5m = df_exposed_scenarios.groupby('mode')['Total buildings (0.15-0.5m)'].mean().reindex(desired_order)
df_ci_flood_0_5_1m = df_exposed_scenarios.groupby('mode')['Total buildings (0.5-1m)'].mean().reindex(desired_order)
df_ci_flood_above_1m = df_exposed_scenarios.groupby('mode')['Total buildings (>1m)'].mean().reindex(desired_order)

# print the each column of total buildings divided by Sum of df_exposed_scenarios per mode (sandy, sandy_shifted, sandy_slr71, sandy_slr101)
print(df_exposed_scenarios.groupby('mode')['Total buildings (0.15-0.5m)'].mean()/df_exposed_scenarios.groupby('mode')['Sum'].mean())
print(df_exposed_scenarios.groupby('mode')['Total buildings (0.5-1m)'].mean()/df_exposed_scenarios.groupby('mode')['Sum'].mean())
print(df_exposed_scenarios.groupby('mode')['Total buildings (>1m)'].mean()/df_exposed_scenarios.groupby('mode')['Sum'].mean())

# divide the sum of df_exposed_scenarios per mode by the sum of df_exposed_scenarios for mode == sandy
df_increase_flood = df_exposed_scenarios.groupby('mode')['Sum'].median()/df_exposed_scenarios.groupby('mode')['Sum'].median()['baseline']
# divide each of the total buildings of df_exposed_scenarios per mode by the corresponding total buildings of df_exposed_scenarios for mode == sandy
df_increase_0_0_5m = df_exposed_scenarios.groupby('mode')['Total buildings (0.15-0.5m)'].median()/df_exposed_scenarios.groupby('mode')['Total buildings (0.15-0.5m)'].median()['baseline']
df_increase_0_5_1m = df_exposed_scenarios.groupby('mode')['Total buildings (0.5-1m)'].median()/df_exposed_scenarios.groupby('mode')['Total buildings (0.5-1m)'].median()['baseline']
df_increase_above_1m = df_exposed_scenarios.groupby('mode')['Total buildings (>1m)'].median()/df_exposed_scenarios.groupby('mode')['Total buildings (>1m)'].median()['baseline']

# print the increase per water level category for each mode and explain in the text each category
print(df_increase_flood)
print(df_increase_0_0_5m)
print(df_increase_0_5_1m)
print(df_increase_above_1m)

# general setup for plots
singlecol = 8.3 * 0.393701
doublecol = 14 * 0.393701
fontsize=9
custom_palette = ['salmon', 'brown', 'maroon']
width = 0.5
alpha = 0.7
plt.rcParams.update({'font.size': 9})
# Rename columns
column_name_mapping = {
    'Total buildings (0.15-0.5m)': '0.15-0.5m',
    'Total buildings (0.5-1m)': '0.5-1m',
    'Total buildings (>1m)': '>1m'
}

coldict = {'0.15-0.5m':'salmon', '0.5-1m':'brown', '0.15-0.5m':'maroon'}
markdict={'MP':'v','SLR71':'o','SLR101':'o'}

# Melt the dataframe to reshape it for Seaborn
df_ci_melted = pd.melt(df_exposed_scenarios, id_vars=['member', 'mode'], value_vars=['Total buildings (0.15-0.5m)', 'Total buildings (0.5-1m)', 'Total buildings (>1m)'])
df_ci_melted['variable'] = df_ci_melted['variable'].replace(column_name_mapping)

# Create a boxplot using Seaborn
fig, ax = plt.subplots(figsize=(doublecol, doublecol))
sns.boxplot(data=df_ci_melted, x='mode', y='value', hue='variable', palette=custom_palette, order=desired_order, 
            whis = (0, 100),showfliers=False, showcaps=False, boxprops=dict(alpha=alpha), linewidth=0.7)
plt.xlabel('')
plt.ylabel('Number of assets')
plt.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_ci_flood.png', dpi=300, bbox_inches='tight')
plt.show()


df_exposed_ci_cat = pd.melt(df_exposed_ci_cat_scenarios, id_vars=['member', 'mode', 'CI system'], value_vars=['Total buildings (0.15-0.5m)', 'Total buildings (0.5-1m)', 'Total buildings (>1m)'])
df_exposed_ci_cat['variable'] = df_exposed_ci_cat['variable'].replace(column_name_mapping)
df_exposed_ci_cat['mode'] = df_exposed_ci_cat['mode'].replace(index_mapping)

df_exposed_ci_cat_scenarios_total = pd.melt(df_exposed_ci_cat_scenarios, id_vars=['member', 'mode', 'CI system'], value_vars=['Sum'])
df_exposed_ci_cat_scenarios_total['mode'] = df_exposed_ci_cat_scenarios_total['mode'].replace(index_mapping)
df_exposed_ci_cat_scenarios_total = df_exposed_ci_cat_scenarios_total[df_exposed_ci_cat_scenarios_total['CI system'] != 'air']

fig, axs = plt.subplots(5, 2, figsize=(doublecol, 8), sharex=True)
axs = axs.flatten()
# Loop through each 'mode' column and create the bar plot for each subplot
for i, mode_col in enumerate(df_exposed_ci_cat['CI system'].unique()[1:]):
    ax = axs[i]
    sns.boxplot(data=df_exposed_ci_cat[df_exposed_ci_cat['CI system'] == mode_col], x='mode', y='value', hue='variable', 
                ax=ax, palette=custom_palette, order=desired_order, whis = (0, 100),showfliers=False, showcaps=False, boxprops=dict(alpha=alpha), linewidth=0.7)
    # set title per panel
    ax.set_title(mode_col)
    ax.set_xlabel('')
    ax.set_ylabel('')
    # set extent as 1.1 max value
    # ax.set_ylim(0, 1.1*df_exposed_ci_cat[df_exposed_ci_cat['CI system'] == mode_col]['value'].max())
    # remove legend
    ax.get_legend().remove()
# set a common Y label for all plots called "flooded CI"
plt.legend(frameon=False, bbox_to_anchor=(-1, -0.55), loc='lower left', ncol=3)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Number of assets", labelpad=20)
# add vertical and horizontal space between the two main vertical columns of the figure
plt.subplots_adjust(hspace=0.4, wspace=0.4)
# plt.tight_layout()

plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_ci_cat_flood.png', dpi=300, bbox_inches='tight')
plt.show()



# Filter the baseline data
baseline_data = df_exposed_ci_cat[df_exposed_ci_cat['mode'] == 'baseline']
baseline_data = baseline_data[baseline_data['CI system'] != 'air']
# change baseline_data minimum value to 1 to avoid division by zero
baseline_data['value'] = baseline_data['value'].replace(0, 1)

# Merge the baseline data with other scenarios and calculate normalized values
normalized_values = []
for mode in df_exposed_ci_cat['mode'].unique():
    if mode != 'baseline':
        scenario_data = df_exposed_ci_cat[df_exposed_ci_cat['mode'] == mode]
        scenario_data = scenario_data[scenario_data['CI system'] != 'air']
        merged_data = pd.merge(scenario_data, baseline_data, on=['CI system', 'variable', 'member'], suffixes=('', '_baseline'))
        normalized_data = merged_data.copy()
        normalized_data['normalized_value'] = merged_data['value'] / merged_data['value_baseline']
        normalized_values.append(normalized_data)

# Concatenate the normalized values into a new dataframe
df_normalized = pd.concat(normalized_values, ignore_index=True)
# find the meadian normalized value per CI system and per mode and per variable
df_normalized_median = df_normalized.groupby(['CI system', 'mode', 'variable'])['normalized_value'].median().reset_index()
df_normalized_median.rename(columns={'mode': 'Scenario'}, inplace=True)
df_normalized_median.rename(columns={'variable': 'Water level'}, inplace=True)
df_normalized_median.rename(columns={'normalized_value': 'Increase'}, inplace=True)


baseline_data_sum = df_exposed_ci_cat_scenarios_total[df_exposed_ci_cat['mode'] == 'baseline']
# Merge the baseline data with other scenarios and calculate normalized values
normalized_values_sum = []
for mode in df_exposed_ci_cat_scenarios_total['mode'].unique():
    if mode != 'baseline':
        scenario_data = df_exposed_ci_cat_scenarios_total[df_exposed_ci_cat_scenarios_total['mode'] == mode]
        merged_data = pd.merge(scenario_data, baseline_data_sum, on=['CI system', 'member'], suffixes=('', '_baseline'))
        normalized_data = merged_data.copy()
        normalized_data['normalized_value'] = merged_data['value'] / merged_data['value_baseline']
        normalized_values_sum.append(normalized_data)
df_normalized_sum = pd.concat(normalized_values_sum, ignore_index=True)



# Create a scatterplot using Seaborn
plt.figure(figsize=(doublecol, doublecol))
sns.boxplot(data=df_normalized, x='CI system', y='normalized_value', hue='mode',  whis = (0, 100), showfliers=False, showcaps=False, palette = custom_palette, dodge=True) #  style='Scenario',s=100, 
plt.xlabel('')
plt.ylabel('Increase with respect to baseline')
plt.title('Relative increase in exposed assets')
plt.legend(frameon =False, bbox_to_anchor=(0.7, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()

# create a new dataframe showing the increase per water level category for each CI system: divide each of the total buildings of df_exposed_ci_cat_mean per CI system and per index by the corresponding total buildings of df_exposed_scenarios for mode == sandy
df_increase_cat_0_0_5m = df_exposed_ci_cat_mean.pivot_table(index='CI system', columns='mode', values='Total buildings (0.15-0.5m)')
df_increase_cat_0_0_5m['MP_increase'] = df_increase_cat_0_0_5m['MP'] / df_increase_cat_0_0_5m['baseline']
df_increase_cat_0_0_5m['SLR101_increase'] = df_increase_cat_0_0_5m['SLR101'] / df_increase_cat_0_0_5m['baseline']
df_increase_cat_0_0_5m['SLR71_increase'] = df_increase_cat_0_0_5m['SLR71'] / df_increase_cat_0_0_5m['baseline']

df_increase_cat_0_5_1m = df_exposed_ci_cat_mean.pivot_table(index='CI system', columns='mode', values='Total buildings (0.5-1m)')
df_increase_cat_0_5_1m['MP_increase'] = df_increase_cat_0_5_1m['MP'] / df_increase_cat_0_5_1m['baseline']
df_increase_cat_0_5_1m['SLR101_increase'] = df_increase_cat_0_5_1m['SLR101'] / df_increase_cat_0_5_1m['baseline']
df_increase_cat_0_5_1m['SLR71_increase'] = df_increase_cat_0_5_1m['SLR71'] / df_increase_cat_0_5_1m['baseline']

df_increase_cat_above_1m = df_exposed_ci_cat_mean.pivot_table(index='CI system', columns='mode', values='Total buildings (>1m)')
df_increase_cat_above_1m['MP_increase'] = df_increase_cat_above_1m['MP'] / df_increase_cat_above_1m['baseline']
df_increase_cat_above_1m['SLR101_increase'] = df_increase_cat_above_1m['SLR101'] / df_increase_cat_above_1m['baseline']
df_increase_cat_above_1m['SLR71_increase'] = df_increase_cat_above_1m['SLR71'] / df_increase_cat_above_1m['baseline']


# Filter out the 'baseline' rows
df_exposed_ci_cat_mean2 = df_exposed_ci_cat_mean.reset_index()

# Filter out the 'baseline' rows
baseline_df = df_exposed_ci_cat_mean2[df_exposed_ci_cat_mean2['mode'] == 'baseline']

# Merge 'MP', 'SLR101', and 'SLR71' with 'baseline' for each CI system
increase_order = ['MP', 'SLR71', 'SLR101']  # Define the desired order of the categories
merged_df = pd.merge(
    baseline_df[['CI system', 'Total buildings (0.15-0.5m)', 'Total buildings (0.5-1m)', 'Total buildings (>1m)']],
    df_exposed_ci_cat_mean2[['mode', 'CI system', 'Total buildings (0.15-0.5m)', 'Total buildings (0.5-1m)', 'Total buildings (>1m)']],
    on='CI system',
    suffixes=('_baseline', '_comparison')
)

# Calculate the divisions
merged_df['0.15-0.5m'] = merged_df['Total buildings (0.15-0.5m)_comparison'] / merged_df['Total buildings (0.15-0.5m)_baseline']
merged_df['0.5-1m'] = merged_df['Total buildings (0.5-1m)_comparison'] / merged_df['Total buildings (0.5-1m)_baseline']
merged_df['>1m'] = merged_df['Total buildings (>1m)_comparison'] / merged_df['Total buildings (>1m)_baseline']

# Drop unnecessary columns and rearrange the columns
result_df = merged_df[['mode', 'CI system', '0.15-0.5m', '0.5-1m', '>1m']]
result_df = result_df[result_df['mode'] != 'baseline']
result_df = result_df[result_df['CI system'] != 'air']

# melt the result_df to have a column with the water level category and a column with the increase
result_df_melt = pd.melt(result_df, id_vars=['mode', 'CI system'], value_vars=['0.15-0.5m', '0.5-1m', '>1m'], var_name='Water level', value_name='Increase')
# change the column name 'mode' for 'scenario'
result_df_melt.rename(columns={'mode': 'Scenario'}, inplace=True)




# Define the shift amounts for each 'Scenario'
shift_amount = {'MP': -0.2, 'SLR71': 0, 'SLR101': 0.2}
# Convert 'CI system' categories to numerical values
unique_ci_systems = result_df_melt['CI system'].unique()
ci_system_mapping = {system: i for i, system in enumerate(unique_ci_systems)}
# Apply shifts to the numerical values
result_df_melt['Shifted_CI_system'] = result_df_melt['CI system'].map(ci_system_mapping)
result_df_melt['Shifted_CI_system'] += result_df_melt['Scenario'].map(shift_amount)

# Create a scatterplot using Seaborn
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the scatterplot using numerical x-values
sns.scatterplot(data=result_df_melt, x='Shifted_CI_system', y='Increase', hue='Water level', style='Scenario', style_order = increase_order, palette = custom_palette, alpha=1, ax=ax)
# Customize plot labels, legend, and layout
plt.xlabel('CI system')
plt.ylabel('Increase with respect to baseline')
plt.title('Relative increase in exposed assets')
ax.set_xticks(list(ci_system_mapping.values()))
ax.set_xticklabels(list(ci_system_mapping.keys()), rotation=45, ha='right')
# Show the plot
plt.legend(frameon =False, bbox_to_anchor=(0.7, 1), loc='upper left')
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_si_ci_cat_flood_increase.png', dpi=300, bbox_inches='tight')
plt.show()

df_normalized_median['Shifted_CI_system'] = df_normalized_median['CI system'].map(ci_system_mapping)
df_normalized_median['Shifted_CI_system'] += df_normalized_median['Scenario'].map(shift_amount)

fig, ax = plt.subplots(figsize=(10, 6))
# Plot the scatterplot using numerical x-values
sns.scatterplot(data=df_normalized_median, x='Shifted_CI_system', y='Increase', hue='Water level', style='Scenario', style_order = increase_order, palette = custom_palette, alpha=1, ax=ax)
# Customize plot labels, legend, and layout
plt.xlabel('CI system')
plt.ylabel('Increase with respect to baseline')
plt.title('Median increase in exposed assets')
ax.set_xticks(list(ci_system_mapping.values()))
ax.set_xticklabels(list(ci_system_mapping.keys()), rotation=45, ha='right')
# Show the plot
plt.legend(frameon =False, bbox_to_anchor=(0.7, 1), loc='upper left')
# plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_si_ci_cat_flood_increase.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(doublecol, doublecol))
sns.boxplot(data=df_normalized_sum, x='CI system', y='normalized_value', hue='mode',  whis = (0, 100), linewidth=0.7, hue_order=increase_order ,showfliers=False, showcaps=False,  dodge=True) #  style='Scenario',s=100, 
plt.xlabel('')
plt.ylabel('Increase with respect to baseline')
plt.title('Relative increase in exposed assets across all water levels')
plt.legend(frameon =False, bbox_to_anchor=(0.7, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.show()



# COMBINED FIGURE
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(doublecol, doublecol*1.4))

# Plot the first figure in the first subplot
axs[0].bar(df_ci_flood_0_0_5m.index, df_ci_flood_0_0_5m, width*0.8, color=custom_palette[0], alpha=alpha, label='0.15-0.5m')
axs[0].bar(df_ci_flood_0_5_1m.index, df_ci_flood_0_5_1m, width*0.8, bottom=df_ci_flood_0_0_5m, color=custom_palette[1], alpha=alpha, label='0.5-1m')
axs[0].bar(df_ci_flood_above_1m.index, df_ci_flood_above_1m, width*0.8, bottom=df_ci_flood_0_0_5m + df_ci_flood_0_5_1m, color=custom_palette[2], alpha=alpha, label='>1m')
axs[0].set_xlabel('')
axs[0].set_ylabel('Number of assets')
axs[0].set_title('a) Flooded CI assets and buildings')
axs[0].legend(frameon=False, bbox_to_anchor=(1.001, 1), loc='upper left')

# Plot the second figure in the second subplot
sns.scatterplot(data=result_df_melt, x='Shifted_CI_system', y='Increase', hue='Water level', palette=custom_palette, style_order=increase_order, style='Scenario', s=70, alpha=0.7, ax=axs[1])
axs[1].set_xlabel('')
axs[1].set_ylabel('Increase with respect to baseline')
axs[1].set_title('b) Relative increase in flooded CI assets and buildings')
axs[1].legend(frameon=False, bbox_to_anchor=(1.001, 1), loc='upper left')
axs[1].set_xticks(list(ci_system_mapping.values()))
axs[1].set_xticklabels(list(ci_system_mapping.keys()), rotation=45, ha='right')

# plt.tight_layout()
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_ci_combined.png', dpi=300, bbox_inches='tight')

plt.show()


################################################################################
# Plot a map with the CI assets (healthcare) and the flooded buildings per category
################################################################################
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hydromt_sfincs import SfincsModel, utils

bbox_ny2 = [-74.252472,40.399902,-73.321381,40.837191]

# open sfincs model with hydromt
mod = SfincsModel(r'd:\paper_3\data\sfincs_ini\spectral_nudging\sandy_base', mode='r')
ds_sfx = mod.grid.copy()
mod.data_catalog.from_yml("d:\paper_3\data\sfincs_ini\spectral_nudging\data_deltares_sandy\data_catalog.yml")
ds_fabdem = mod.data_catalog.get_rasterdataset("fabdem")
ds_fabdem_raster = ds_fabdem.raster.reproject_like(mod.grid)

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


# Healthcare changes
system = 'power'
healthcare_exposure_factual_2_slr101 = cis_exposure[('D:\\paper_3\\data\\sfincs_inundation_results\\sandy_slr101\\hmax_sandy_slr101_factual_2_rain_surge', system)]
healthcare_exposure_factual_2 = cis_exposure[('D:\\paper_3\\data\\sfincs_inundation_results\\sandy\\hmax_sandy_factual_2_rain_surge', system)]

merged_healthcare = pd.merge(healthcare_exposure_factual_2_slr101, healthcare_exposure_factual_2, on='building_index', how='outer', suffixes=('_slr101', '_normal'))
# Fill missing flood_data values with 0
merged_healthcare['flood_data_slr101'].fillna(0, inplace=True)
merged_healthcare['flood_data_normal'].fillna(0, inplace=True)
merged_healthcare['flood_data_difference'] = merged_healthcare['flood_data_slr101'] - merged_healthcare['flood_data_normal']
merged_healthcare = merged_healthcare.set_geometry('geometry_slr101')
# Find the unique rows representing new assets flooded in SLR 101 scenario
new_healthcare_assets = merged_healthcare[merged_healthcare['flood_data_normal'] == 0]

# Step 3: Plot the difference
fig, ax = plt.subplots(figsize=(doublecol , doublecol), subplot_kw={'projection': utm})
da_hs.plot(ax=ax, cmap='Greys', add_colorbar=False, alpha=0.2)
ds_healthcare = merged_healthcare.plot(ax=ax, column='flood_data_difference', cmap='RdBu', legend=True, vmin=-1.5, vmax=1.5, legend_kwds={'label': "Difference in flood depth [m]"}, markersize = 10)
# new_healthcare_assets.plot(ax=ax, color='none', edgecolor='black', linewidth=1, markersize = 10)
ax.set_xticks(ax.get_xticks()[::2])
ax.set_ylabel('y coordinate UTM zone 18N [10⁶ m]')
ax.set_xlim(bbox_utm[0], bbox_utm[1])
ax.set_ylim(bbox_utm[2], bbox_utm[3])
ax.yaxis.set_visible(True)
plt.title(f"Change in flooded {system} assets between SLR101 and baseline")
plt.show()


# plot MP scenarios for telecom
system_mp = 'road'
telecom_exposure_factual_2_shifted = cis_exposure[('D:\\paper_3\\data\\sfincs_inundation_results\\sandy_shifted\\hmax_sandy_shifted_factual_3_rain_surge', system_mp)]
telecom_exposure_factual_2 = cis_exposure[('D:\\paper_3\\data\\sfincs_inundation_results\\sandy\\hmax_sandy_factual_3_rain_surge', system_mp)]

merged_telecom = pd.merge(telecom_exposure_factual_2_shifted, telecom_exposure_factual_2, on='building_index', how='outer', suffixes=('_mp', '_normal'))
# Fill missing flood_data values with 0
merged_telecom['flood_data_mp'].fillna(0, inplace=True)
merged_telecom['flood_data_normal'].fillna(0, inplace=True)
merged_telecom['flood_data_difference'] = merged_telecom['flood_data_mp'] - merged_telecom['flood_data_normal']
merged_telecom = merged_telecom.set_geometry('geometry_mp')
# Find the unique rows representing new assets flooded in SLR 101 scenario
new_telecom_assets = merged_telecom[merged_telecom['flood_data_normal'] == 0]



fig, ax = plt.subplots(figsize=(doublecol , doublecol), subplot_kw={'projection': utm})
da_hs.plot(ax=ax, cmap='Greys', add_colorbar=False, alpha=0.2)
merged_telecom.plot(ax=ax, column='flood_data_difference', cmap='RdBu', legend=True, vmin=-0.5, vmax=0.5, legend_kwds={"shrink":.5,'label': "Difference in flood depth [m]"})
new_telecom_assets.plot(ax=ax, color='none', edgecolor='black', linewidth=1)
ax.set_xticks(ax.get_xticks()[::2])
ax.set_xlim(bbox_utm[0], bbox_utm[1])
ax.set_ylim(bbox_utm[2], bbox_utm[3])
ax.yaxis.set_visible(True)
plt.title(f"Change in flooded {system_mp} assets between MP and baseline")
plt.show()



df_ci_melted_copy = df_ci_melted.copy()
df_ci_melted_copy['value'] = df_ci_melted_copy['value']*10**-5

# obtain the median per ci system of df_exposed_ci_cat
df_exposed_ci_cat_median = df_exposed_ci_cat.groupby(['CI system'])['value'].median()
# now order the df_exposed_ci_cat_median according to the descending order of df_exposed_ci_cat_median
df_exposed_ci_cat_median = df_exposed_ci_cat_median.sort_values(ascending=False)




# COMBINED FIGURE
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(doublecol*1.6, doublecol*1.4), gridspec_kw={'height_ratios': [0.4,1]}) #
# decrease the text size of the subplots
plt.rcParams.update({'font.size': 8})

# Plot the first figure in the first subplot
sns.boxplot(data=df_ci_melted_copy, x='mode', y='value', hue='variable', palette=custom_palette, order=desired_order, 
            whis = (0, 100),showfliers=False, showcaps=False, boxprops=dict(alpha=alpha), linewidth=0.7, ax=axs[0,0])
axs[0,0].set_xlabel('')
axs[0,0].set_ylabel('Number of assets (10⁵)')
axs[0,0].set_title('a) Flooded CI assets and buildings', loc = 'left')
axs[0,0].get_legend().remove()

# Plot the second figure in the second subplot # df_normalized_median or result_df_melt
sns.scatterplot(data=df_normalized_median, x='Shifted_CI_system', y='Increase', hue='Water level', palette=custom_palette, style_order=increase_order, style='Scenario', s=30, alpha=1, ax=axs[0,1])
axs[0,1].set_xlabel('')
axs[0,1].set_ylabel('Increase wrt baseline')
axs[0,1].set_title('b) Median increase in flooded CI systems', loc = 'left')
axs[0,1].legend(frameon=False, bbox_to_anchor=(0.98, 1), loc='upper left')
axs[0,1].set_xticks(list(ci_system_mapping.values()))
axs[0,1].set_xticklabels(list(ci_system_mapping.keys()), rotation=40, ha='center')
axs[0,1].set_yticks([0, 5, 10, 15])


da_hs.plot(ax=axs[1,0], cmap='Greys', add_colorbar=False, alpha=0.2)
merged_healthcare.plot(ax=axs[1,0], column='flood_data_difference', cmap='RdBu', vmin=-1.0, vmax=1.0, markersize = 3) # ,legend=True, legend_kwds={"location":"bottom", "shrink":.5,'label': "Difference in flood depth [m]"}
# axs[1,0].set_xticks([])
# axs[1,0].set_xticklabels([])
# axs[1,0].set_ylabel('')
axs[1,0].set_title('')
axs[1,0].set_xticks(axs[1,0].get_xticks()[::2])
# axs[1,0].set_yticks(axs[1,0].get_yticks()[::2])
axs[1,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
axs[1,0].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')
axs[1,0].set_xlabel('x coordinate UTM zone 18N [10 m]')


axs[1,0].set_title(f'c) Power: SLR101 increase', loc = 'left')
axs[1,0].set_xlim(bbox_utm[0], bbox_utm[1])
axs[1,0].set_ylim(bbox_utm[2], bbox_utm[3])
# axs[1,0].yaxis.set_visible(False)
# axs[1,0].set_xlabel('')


da_hs.plot(ax=axs[1,1], cmap='Greys', add_colorbar=False, alpha=0.2)
fig_colorbar = merged_telecom.plot(ax=axs[1,1], column='flood_data_difference', cmap='RdBu', legend=False, vmin=-1.0, vmax=1.0, markersize = 1 ) # , legend_kwds={"location":"right","shrink":.5,'label': "Difference in flood depth [m]"},
# new_telecom_assets.plot(ax=axs[1,1], color='none', edgecolor='black', linewidth=1)
axs[1,1].set_ylabel('')
axs[1,1].set_yticklabels([])
axs[1,1].set_title('')
axs[1,1].set_xticks(axs[1,0].get_xticks())
axs[1,1].set_title(f'd) Road: MP increase', loc = 'left')
axs[1,1].set_xlim(bbox_utm[0], bbox_utm[1])
axs[1,1].set_ylim(bbox_utm[2], bbox_utm[3])
# axs[1,1].set_xlabel('')
# axs[1,1].yaxis.set_visible(True)

# axs[1,0].set_yticks(axs[1,0].get_yticks()[::2])
# axs[1,1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:.2f}'.format(x/1e6)))
# axs[1,1].set_ylabel('y coordinate UTM zone 18N [10⁶ m]')
axs[1,1].set_xlabel('x coordinate UTM zone 18N [10 m]')

cax = fig.add_axes([0.92, 0.3, 0.01, 0.2])
cmap = plt.cm.get_cmap('RdBu')  # Replace 'RdBu' with your desired colormap
vmin, vmax = -1.0, 1.0  # Replace these values with your desired colorbar range
dummy_scatter = axs[1, 1].scatter([], [], c=[], cmap=cmap, vmin=vmin, vmax=vmax)  
cbar = plt.colorbar(mappable=dummy_scatter, cax=cax, orientation='vertical', extend='max', label='Flood depth change [m]', cmap=cmap)

plt.subplots_adjust(hspace=-0.1, wspace=0.15)

# plt.tight_layout()
plt.savefig(r'd:\paper_3\data\Figures\paper_figures\fig_ci_combined_maps.png', dpi=300, bbox_inches='tight')

plt.show()
