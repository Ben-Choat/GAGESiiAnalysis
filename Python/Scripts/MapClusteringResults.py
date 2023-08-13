'''
BChoat 2023/08/10

Plot maps of clustering results.


'''

'''
BChoat 10/20/2022

Script to create maps for dissertation
'''


# %%
# import libraries
################

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
# import seaborn as sns
# from shapely.geometry import Point
import numpy as np


# %%
# read in data
##################

# spatial file of us
states = gpd.read_file(
    'D:/DataWorking/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
)
# filter out Alaska, Hawaii, and Puerto Rico
states = states[~states['NAME'].isin(['Alaska', 'Puerto Rico', 'Hawaii'])]


# read in id data with lat, long, aggecoregion, and reference vs non-reference
df_IDtrain = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'
    'ID_train.csv',
    dtype = {'STAID': 'string'}
)

df_IDvalnit = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'
    'ID_valnit.csv',
    dtype = {'STAID': 'string'}
)

df_IDtrain = df_IDtrain.astype(str)
df_IDvalnit = df_IDvalnit.astype(str)

# directory where to write figs
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'

# %%
# prep data
##########

# make points from lat long in id files
# training
long_train = df_IDtrain['LNG_GAGE']
lat_train = df_IDtrain['LAT_GAGE']
points_train = gpd.points_from_xy(long_train, lat_train, crs = states.crs)
geo_df_train = gpd.GeoDataFrame(geometry = points_train)
geo_df_train['STAID'] = df_IDtrain['STAID']
geo_df_train = geo_df_train.merge(df_IDtrain, on = 'STAID')

# valnit
long_valnit = df_IDvalnit['LNG_GAGE']
lat_valnit = df_IDvalnit['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit['STAID'] = df_IDvalnit['STAID']
geo_df_valnit = geo_df_valnit.merge(df_IDvalnit, on = 'STAID')



# %%
# map
############
# import matplotlib
# matplotlib.use('Agg')
# columns in geo_df_train
# ['geometry', 'STAID', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE',
#        'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site', 'CAMELS', 'HLR',
#        'All_0', 'All_1', 'All_2', 'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
#        'Anth_0', 'Anth_1']

# for clust_meth in [
#        'Class', 'AggEcoregion', 'CAMELS', 'HLR',
#        'All_0', 'All_1', 'All_2', 
#        'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
#        'Anth_0', 'Anth_1'
#        ]:
    
for clust_meth in [
       'Nat_3'
       ]:

# clust_meth = 'Anth_0'


    # define colormap
    cm_org = plt.get_cmap('tab20_r')
    if clust_meth in ['All_0', 'All_1', 'All_2', 
                    'Nat_0','Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
                    'Anth_0', 'Anth_1']:   
        add_colors = ['black', 'sandybrown', 'peru', 
                            'hotpink', 'cyan', 'aquamarine', 'blueviolet', 'lime']
    else:
        add_colors = ['sandybrown', 'peru', 
                    'hotpink', 'cyan', 'aquamarine', 'blueviolet', 'lime']
    add_rgb = [mcolors.to_rgb(col) for col in add_colors]
    all_colors = add_rgb + list(cm_org.colors)
    cm_cust = mcolors.ListedColormap(all_colors)
    cmap_in = cm_cust

    nlegcols_in = int(len(geo_df_train[clust_meth].unique())/3)



    # cmap_in[0]

    # define base axes as state boundaries
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 12), sharex = True)
    states.boundary.plot(
        ax = ax1,
        figsize = (12, 9), 
        color = 'Gray',
        linewidth = 1,
        zorder = 0
    )
    states.boundary.plot(
        ax = ax2,
        figsize = (12, 9), 
        color = 'Gray',
        linewidth = 1,
        zorder = 0
    )
    # plot training points
    geo_df_train.plot(
        ax = ax1,
        column = clust_meth, 
        markersize = 20, 
        marker = 'o',
        legend = True,
        legend_kwds = {'loc': 'lower center', 
                            'ncol': nlegcols_in},
        cmap = cmap_in,
        zorder = 5)

    ax1.set(title = 'Training Catchments')

    # plot valnit points
    geo_df_valnit.plot(
        ax = ax2,
        column = clust_meth, 
        markersize = 20, 
        marker = 'o',
        cmap = cmap_in,
        zorder = 5)

    ax2.set(
        title = 'Unseen Testing Catchments', 
        zorder = 5)

    # turn off axis tick labels
    ax1.tick_params(axis = 'both', which = 'both', length = 0)
    ax2.tick_params(axis = 'both', which = 'both', length = 0)

    # turn off plot frame
    for spine in ax1.spines.values():
        spine.set_visible(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # legend position
    leg = ax1.get_legend()
    leg.set_bbox_to_anchor((0.5, -0.19))
    leg.get_frame().set_edgecolor('none')

    plt.show()