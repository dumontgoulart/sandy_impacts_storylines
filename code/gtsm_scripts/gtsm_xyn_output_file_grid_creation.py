import os
import glob
import pandas as pd
import numpy as np
#from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

df = pd.read_csv(r"C:\Users\morenodu\Downloads\selected_output_OR.csv",names=['lon','lat','id'], sep = ";",)
df['id'] = df['id'].str.replace("'", "")

# Define your boundary box coordinates
# xynthia  Box -12.963867,40.010787,2.812500,51.096623 
# Sandy Box -75.805664,38.548165,-69.708252,42.892064
lat_min = 38.548165
lat_max = 42.892064
lon_min = -75.805664
lon_max = -69.708252

df_bounded = df[(df['lat'] >= lat_min) & (df['lat'] <= lat_max) & (df['lon'] >= lon_min) & (df['lon'] <= lon_max)]
df_bounded.to_csv('D:/paper_3/data/gtsm_local_runs/selected_output_OR_sandy_snapped_1p25eu_unique_obs.xyn', header = False, index = False)