# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:24:06 2021

@author: veenstra
updated by hg from Irene
"""

import os
#import datetime as dt
import glob
import pandas as pd
import numpy as np
#from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
plt.close('all')
import contextily as ctx
from scipy.spatial import KDTree
import xugrid as xu
import dfm_tools as dfmt
# from dfm_tools.get_nc import plot_netmapdata #, get_netdata#, plot_background
# from dfm_tools.get_nc_helpers import get_ncvardimlist#, get_hisstationlist  
#from dfm_tools.regulargrid import scatter_to_regulargrid
#from dfm_tools.io.bc import read_bcfile
# from dfm_tools.io.polygon import Polygon
#from dfm_tools.get_nc_helpers import get_timesfromnc

def xlonylat2xyzcartesian(data):
    """
    necessary to calculate cartesian distances, otherwise nearest neigbour can fail.
    https://stackoverflow.com/questions/45127141/find-the-nearest-point-in-distance-for-all-the-points-in-the-dataset-python
    """
    R = 6367
    phi = np.deg2rad(data['y'])
    theta = np.deg2rad(data['x'])
    data = pd.DataFrame()
    data['x_cart'] = R * np.cos(phi) * np.cos(theta)
    data['y_cart'] = R * np.cos(phi) * np.sin(theta)
    data['z_cart'] = R * np.sin(phi)
    return data
def dist_to_arclength(chord_length):
    """
    https://stackoverflow.com/questions/45127141/find-the-nearest-point-in-distance-for-all-the-points-in-the-dataset-python
    """
    R = 6367 # earth radius
    central_angle = 2*np.arcsin(chord_length/(2.0*R)) 
    arclength = R*central_angle*1000
    return arclength


dir_obsoriginal = r'D:\paper_3\data\gtsm_local_runs'  # Location of the observation file I defined! Still need to put it in Snellius
obspointfile_list = [os.path.basename(x) for x in glob.glob(os.path.join(dir_obsoriginal,'selected_output_OR_sandy.xyn'))]

#settings
also_unique_set = True
plot_result = False
threshold_wlrange = 0.05#0.05 # minimum variability water levels
threshold_mindepth = 3.0 #[m] bed level of the point
threshold_movedist = 20e3 #[m] max distance to qualify for selection (that we want to move the points). beware that grid resolution is 25-50km for ocean stations like FES, but 20e3 is fine for 1p25eu and 5km since FES is not snapped. for 25km grid, min dist is 20km (dist from corner to center). For 50km grid, min dist is 38km (30e3 results in 4 deep water FES stations being too far).
list_nosnapping = ['selected_output_OR_sandy.xyn']  # I do not have more files
#dir_modelrun = r'/gpfs/home4/benitoli/gtsm_d3dfm/model2'  
dir_modelrun = r'D:\paper_3\data\gtsm_local_runs'  

# file_ldb = os.path.join(dir_modelrun,'world.ldb')  # confirm this location is correct!
identifier = '1p25eu'
#obspointfile_list = ['selected_output_OR.xyn']  # change name by the one in Snellius
	
#retrieve obs data
dir_obssnapped = os.path.join(dir_obsoriginal,'observation_locations_snapped_%s'%(identifier))
if not os.path.exists(dir_obssnapped):
    os.mkdir(dir_obssnapped)

#retrieve modeldata
file_nc = os.path.join(dir_modelrun,'gtsm_model_0000_fou.nc')
#file_nc = os.path.join(dir_modelrun,'output','gtsm_fine_0000_fou.nc')
data_xr_map = dfmt.open_partitioned_dataset(file_nc)
vars_pd = dfmt.get_ncvarproperties(data_xr_map)
print('retrieving data from fourier file')
data_cellcenx = data_xr_map['mesh2d_face_x'] #dfmt.open_partitioned_dataset(file_nc, varname='mesh2d_face_x', silent=True)
data_cellceny = data_xr_map['mesh2d_face_y'] #get_ncmodeldata(file_nc, varname='mesh2d_face_y', silent=True)
data_minwl = data_xr_map['mesh2d_fourier001_min'] #get_ncmodeldata(file_nc, varname='mesh2d_fourier001_min', silent=True)
data_mindep = data_xr_map['mesh2d_fourier001_min_depth'] #get_ncmodeldata(file_nc, varname='mesh2d_fourier001_min_depth', silent=True)
data_maxwl = data_xr_map['mesh2d_fourier002_max'] #get_ncmodeldata(file_nc, varname='mesh2d_fourier002_max', silent=True)
print('..finished')

data_meanwl = (data_minwl+data_maxwl)/2
data_wlrange = data_maxwl-data_minwl
bool_valid_cells = ((data_wlrange>threshold_wlrange) & (data_mindep>threshold_mindepth))

#creating kdtree with valid cell centers (cartesian coordinates)
data_celcenxy_valid = pd.DataFrame({'x':data_cellcenx[bool_valid_cells],'y':data_cellceny[bool_valid_cells]})#,'area':data_cellarea[bool_valid_cells]})
data_celcenxy_valid_cart = xlonylat2xyzcartesian(data_celcenxy_valid)
tree = KDTree(data_celcenxy_valid_cart[['x_cart','y_cart','z_cart']])

for iOP, obspointfilename in enumerate(obspointfile_list):
    file_obsorg = os.path.join(dir_obsoriginal,obspointfilename)
    print('processing %s'%(obspointfilename))
    if obspointfilename.endswith('_obs.xyn'):
        filepostfix = '_obs.xyn'
    else:
        filepostfix = '.xyn'
    obspointfilename_out = obspointfilename.replace(filepostfix,'_snapped_%s_obs.xyn'%(identifier))
    obspointfilename_outorig = obspointfilename.replace(filepostfix,'_original_obs.xyn')
    obspointfilename_outuniq = obspointfilename.replace(filepostfix,'_snapped_%s_unique_obs.xyn'%(identifier))
    obspointfilename_outtoofar = obspointfilename.replace(filepostfix,'_toofar_%s_obs.xyn'%(identifier))
    if obspointfilename=='sealevel_obs_ioc.xyn':
        data_obsorg = pd.read_csv(file_obsorg,delim_whitespace=True,names=['x','y','name','#','comment'], quotechar="'")
    else:
        data_obsorg = pd.read_csv(file_obsorg,names=['x','y','name'], sep = ',', quotechar="'")
    
    #check for spaces in names and add quotes if necessary
    if data_obsorg['name'].str.contains('"').sum()>0:
        raise Exception('station names contain double quotes ("), use single quotes (or no quotes) instead')
    if data_obsorg['name'].str.contains(' ').sum()>0:
        print('names contains spaces, so quotes are added in outputfile')
        data_obsorg['name'] = "'"+data_obsorg['name']+"'"
    data_obsorg_cart = xlonylat2xyzcartesian(data_obsorg)
    
    #finding nearest cellcenter-neighbors of each obspoint in file
    distance_cart, index = tree.query(data_obsorg_cart.loc[:,['x_cart','y_cart','z_cart']], k=1)
    #distance_meter = distToMeter(distance_cart)
    distance_meter = dist_to_arclength(distance_cart)
    
    data_celcenxy_validsel = data_celcenxy_valid.loc[index,:]
    data_obsorg['x_snapped'] = data_celcenxy_validsel['x'].values
    data_obsorg['y_snapped'] = data_celcenxy_validsel['y'].values
    data_obsorg['distance_cart'] = distance_cart
    data_obsorg['distance_meter'] = distance_meter
    
    bool_toofar = data_obsorg['distance_meter'] > threshold_movedist
    data_obsoriginal = data_obsorg[['x','y','name']]
    data_obssnapped = data_obsorg.loc[~bool_toofar,['x_snapped','y_snapped','name']]
    data_obssnapped_uniq = data_obssnapped.drop_duplicates(subset=['x_snapped','y_snapped'])
    data_obstoofar = data_obsorg.loc[bool_toofar,['x','y','name']]
    '''
    if obspointfilename in list_nosnapping:
        print('no snapping applied, writing %d original stations'%(len(data_obssnapped)))
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outorig),data_obsoriginal, fmt='%11.6f %11.6f  %-s')
    else:
        print('discarded %d stations, results in %d stations with non-unique coordinates'%(bool_toofar.sum(),len(data_obssnapped)-len(data_obssnapped_uniq)))
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_out),data_obssnapped, fmt='%11.6f %11.6f  %-s')
        if also_unique_set:
            np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outuniq),data_obssnapped_uniq, fmt='%11.6f %11.6f  %-s')
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outtoofar),data_obstoofar, fmt='%11.6f %11.6f  %-s')
    '''
    if obspointfilename in list_nosnapping:
        print('no snapping applied, writing %d original stations'%(len(data_obssnapped)))
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outorig),data_obsoriginal, fmt='      %11.6f     %11.6f %-s')
    else:
        print('discarded %d stations, results in %d stations with non-unique coordinates'%(bool_toofar.sum(),len(data_obssnapped)-len(data_obssnapped_uniq)))
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_out),data_obssnapped, fmt='      %11.6f     %11.6f %-s')
        if also_unique_set:
            np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outuniq),data_obssnapped_uniq, fmt='      %11.6f     %11.6f %-s')
        np.savetxt(os.path.join(dir_obssnapped,obspointfilename_outtoofar),data_obstoofar, fmt='      %11.6f     %11.6f %-s')

if plot_result:

    test = xu.open_dataset(file_nc)
    ugrid = test.grid.face_node_coordinates

    # ugrid = get_ugrid_verts(data_xr_map)
    # ugrid.mask = np.isnan(ugrid)
    bool_nobackconnect = (ugrid[:,:,0]>-180).all(axis=1)
    bool_equ = ((ugrid[:,:,0]>-100) & (ugrid[:,:,0]<175) & (ugrid[:,:,1]>-20) & (ugrid[:,:,1]<20)).all(axis=1)
    bool_eur = ((ugrid[:,:,0]>-10) & (ugrid[:,:,0]<42.5) & (ugrid[:,:,1]>30) & (ugrid[:,:,1]<70)).all(axis=1)
    bool_isr = ((ugrid[:,:,0]>33) & (ugrid[:,:,0]<36) & (ugrid[:,:,1]>31) & (ugrid[:,:,1]<34)).all(axis=1)
    bool_south = ((ugrid[:,:,0]>-75.6) & (ugrid[:,:,0]<-74.3) & (ugrid[:,:,1]>-65.3) & (ugrid[:,:,1]<-64.6)).all(axis=1)
    bool_blacksea = ((ugrid[:,:,0]>26) & (ugrid[:,:,0]<42.5) & (ugrid[:,:,1]>40) & (ugrid[:,:,1]<48)).all(axis=1)
    
    def plot_obspoints(ax1):
        ax1.plot(data_obsorg.loc[:,'x'],data_obsorg.loc[:,'y'],'xk')
        ax1.plot(data_obsorg.loc[~bool_toofar,'x_snapped'],data_obsorg.loc[~bool_toofar,'y_snapped'],'xg')
        ax1.plot(data_obsorg.loc[bool_toofar,'x_snapped'],data_obsorg.loc[bool_toofar,'y_snapped'],'xr')
    
#     #retrieve ldb data
#     ldb_world = Polygon.fromfile(file_ldb)[0][0]
    
    if 1:
        fig,ax1 = plt.subplots(figsize=(12,8))
        ax1.set_xlim(-12,3)
        ax1.set_ylim(40,55)
        plot_obspoints(ax1)
        # ctx.add_basemap(ax1, crs="EPSG:4326", zoom=9, attribution_size=5, reset_extent=True)      
        fig.tight_layout()
        
        plt.show()
        print()
        # fig.savefig(os.path.join(dir_obssnapped,'bool_world'))
        # ax1.set_xlim(-15,42.5)
        # ax1.set_ylim(30,70)
        # fig.savefig(os.path.join(dir_obssnapped,'bool_europe'))
        # ax1.set_xlim(6,15)
        # ax1.set_ylim(53,59)
        # fig.savefig(os.path.join(dir_obssnapped,'bool_Denmark'))
        # ax1.set_xlim(26,42.5)
        # ax1.set_ylim(40,48)
        # fig.savefig(os.path.join(dir_obssnapped,'bool_blacksea'))
        # ax1.set_xlim(0,10)
        # ax1.set_ylim(50,56)
        # fig.savefig(os.path.join(dir_obssnapped,'bool_netherlands'))

#     if 0:
#         fig,ax1 = plt.subplots()
#         pc = plot_netmapdata(ugrid[bool_nobackconnect,:,:], values=None, ax=ax1, linewidth=0.5, color="crimson", facecolor="None")
#         ax1.plot(ldb_world[:,0],ldb_world[:,1],'k-',linewidth=0.5)
#         plot_obspoints(ax1)

   
#     if 0:
#         fig,ax1 = plt.subplots()
#         pc = plot_netmapdata(ugrid[bool_eur,:,:], values=bool_valid_cells[bool_eur].astype(int), ax=ax1, edgecolor='none')
#         cbar = fig.colorbar(pc,ax=ax1)
#         ax1.plot(ldb_world[:,0],ldb_world[:,1],'k-',linewidth=0.5)
#         #ax1.set_xlim(26,42.5)
#         #ax1.set_ylim(40,48)
        
#     if 0:
#         fig,ax1 = plt.subplots()
#         pc = plot_netmapdata(ugrid[bool_nobackconnect,:,:], values=bool_valid_cells[bool_blacksea].astype(int), ax=ax1, edgecolor='none')
#         cbar = fig.colorbar(pc,ax=ax1)
#         ax1.plot(ldb_world[:,0],ldb_world[:,1],'k-',linewidth=0.5)
#         ax1.set_xlim(26,42.5)
#         ax1.set_ylim(40,48)
#         #ax1.plot(data_cellcenx[bool_south],data_cellceny[bool_south],'o')
#         #for i in range(bool_south.sum()):
#         #    ax1.text(data_cellcenx[bool_south][i],data_cellceny[bool_south][i],'%d'%np.where(bool_south)[0][i])
#         plot_obspoints(ax1)
#         pc.set_clim(0,1)
#         ax1.set_aspect('equal')
#         #ax1.set_xlim(-10,37)
#         #ax1.set_ylim(30,70)
#         ax1.set_xlim(-3,0)
#         ax1.set_ylim(44,48)
        
#         fig,ax1 = plt.subplots()
#         pc = plot_netmapdata(ugrid[bool_nobackconnect,:,:], values=bool_valid_cells[bool_nobackconnect].astype(int), ax=ax1, edgecolor='face')
#         cbar = fig.colorbar(pc,ax=ax1)
#         ax1.plot(ldb_world[:,0],ldb_world[:,1],'k-',linewidth=0.5)
    
#         # fig,ax1 = plt.subplots()
#         # pc = plot_netmapdata(ugrid[bool_nobackconnect,:,:], values=data_wlrange[bool_nobackconnect], ax=ax1, edgecolor='face')
#         # pc.set_clim([0,3])
#         # cbar = fig.colorbar(pc,ax=ax1)
#         # source_list = [ctx.providers.Stamen.Terrain, #default source
#         #                ctx.providers.Esri.WorldImagery,
#         #                ctx.providers.CartoDB.Voyager,
#         #                #ctx.providers.NASAGIBS.ViirsEarthAtNight2012,
#         #                ctx.providers.Stamen.Watercolor]
        
#         # #ctx.add_basemap(ax1, source=ctx.providers.Esri.WorldImagery, crs="EPSG:4326", attribution_size=5)
    


