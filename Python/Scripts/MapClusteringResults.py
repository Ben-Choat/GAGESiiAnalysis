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

alpha_in = 1
marker_in = 'o' 
# ',' # 'x'
msize_in = 1.5 # markersize

# replace '-1' with 'noise
# geo_df_train = geo_df_train.replace('-1', 'noise')
# geo_df_valnit = geo_df_valnit.replace('-1', 'noise')

# import matplotlib
# matplotlib.use('Agg')
# columns in geo_df_train
# ['geometry', 'STAID', 'DRAIN_SQKM', 'HUC02', 'LAT_GAGE', 'LNG_GAGE',
#        'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site', 'CAMELS', 'HLR',
#        'All_0', 'All_1', 'All_2', 'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
#        'Anth_0', 'Anth_1']

clust_meths_in =  [
       'Class', 'AggEcoregion', 'CAMELS', 'HLR',
       'All_0', 'All_1', 'All_2', 
       'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
       'Anth_0', 'Anth_1'
       ]

# for clust_meth in clust_meths_in:
    
# for clust_meth in [
#        'Nat_3'
#        ]:

# clust_meth = 'Anth_0'



    


# cmap_in[0]

# define base axes as state boundaries
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (3, 3), sharex = True)
fig, axs = plt.subplots(5, 3, figsize = (8, 9), 
                                sharex = True, sharey = True)

# fig = plt.figure(figsize = (12, 9))
# ax = fig.add_subplot(111)

# new_fig = plt.figure()
axes = axs.flatten()

for i in range(14):
    clust_meth = clust_meths_in[i]
    # get number of clusters
    Nclust  = len(geo_df_train[clust_meth].unique())

    # define colormap
    cm_org = plt.get_cmap('tab20c')
    
    if clust_meth in ['All_0', 'All_1', 'All_2', 
                    'Nat_0','Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
                    'Anth_0', 'Anth_1']:   
        add_colors = ['black', 'sandybrown', 'cyan', 'peru', 'hotpink',  'aquamarine', 
                    'blueviolet', 'lime']
    else:
        add_colors = ['sandybrown', 'cyan', 'peru', 
                    'hotpink', 'aquamarine', 'blueviolet', 'lime']
    add_rgb = [mcolors.to_rgb(col) for col in add_colors]
    all_colors = add_rgb + list(cm_org.colors)
    all_colors = all_colors[0:Nclust]
    cm_cust = mcolors.ListedColormap(all_colors)
    cmap_in = cm_cust

    # define title for each subplot
    sp_title = f'{clust_meth} ({Nclust} clusters)'

    # if len(geo_df_train[clust_meth].unique()) > 14:
    #     nlegcols_in = 5 # int(len(geo_df_train[clust_meth].unique())/7)
    #     anch_in = (1.3, 0.5)
    # else:
    #     nlegcols_in = 3
        
    #     anch_in = (1.2, 0.5)

    ax1 = axes[i]
    
#     ax1 = ax.inset_axes([0.1, 0.5, 0.5, 0.5]) # x, y, width, height
#     ax2 = ax.inset_axes([0.1, 0.0, 0.5, 0.5])

# for outer_ax in axs.flatten():
        
# new_fig = plt.figure()
# ax1 = new_fig.add_subplot(211)
# ax2 = new_fig.add_subplot(212)

# create inner grid
# inner_axs = outer_ax.subgridspec(2, 1)

# access inner subplots
# ax1 = inner_axs[0]
# ax2 = inner_axs[1]

    states.boundary.plot(
        ax = ax1,
        # figsize = (12, 9), 
        color = 'Gray',
        linewidth = 1,
        zorder = 0
    )
    # states.boundary.plot(
    #     ax = ax2,
    #     # figsize = (12, 9), 
    #     color = 'Gray',
    #     linewidth = 1,
    #     zorder = 0
    # )
    # plot training points
    geo_df_train.plot(
        ax = ax1,
        column = clust_meth, 
        markersize = msize_in, 
        marker = marker_in,
        legend = False,
        # legend_kwds = {'loc': 'center left', 
        #                     'ncol': nlegcols_in},
        cmap = cmap_in,
        alpha = alpha_in,
        zorder = 5)

    # ax1.set(title = 'Training Catchments')
    # title = ax1.set_title('Training Catchments', fontsize = 8)
    # title.set_fontsize(8)
    title = ax1.set_title(sp_title, fontsize = 8)

    # plot valnit points
    # geo_df_valnit.plot(
    #     ax = ax2,
    #     column = clust_meth, 
    #     markersize = msize_in, 
    #     marker = marker_in,
    #     cmap = cmap_in,
    #     alpha = alpha_in,
    #     zorder = 5)

    # ax2.set(
    #     title = 'Unseen Testing Catchments', 
    #     zorder = 5)
    # title = ax2.set_title('Unseen Testing Catchments', fontsize = 8)
    # title.set_fontsize(8)

    # turn off axis tick labels
    ax1.tick_params(axis = 'both', which = 'both', length = 0)
    # ax2.tick_params(axis = 'both', which = 'both', length = 0)

    # turn off plot frame
    for spine in ax1.spines.values():
        spine.set_visible(False)
    # for spine in ax2.spines.values():
    #     spine.set_visible(False)

    # legend position
    # leg = ax1.get_legend()
    # leg.set_bbox_to_anchor(anch_in) # anch_in, transform = fig.transFigure
    # leg.set_frame_on(False)
    # # leg.get_frame().set_edgecolor('none')
    # for label in leg.get_texts():
    #     label.set_fontsize(6)
    # for handle in leg.legendHandles:
    #     handle.set_markersize(6)

    # leg.set_zorder(6)

    plt.suptitle('Clustering Results for Training Data', fontsize = 9)

    fig.tight_layout()

plt.show()
# %%






# states.boundary.plot(
#         ax = ax1,
#         figsize = (12, 9), 
#         color = 'Gray',
#         linewidth = 1,
#         zorder = 0
#     )
#     states.boundary.plot(
#         ax = ax2,
#         figsize = (12, 9), 
#         color = 'Gray',
#         linewidth = 1,
#         zorder = 0
#     )
#     # plot training points
#     geo_df_train.plot(
#         ax = ax1,
#         column = clust_meth, 
#         markersize = msize_in, 
#         marker = marker_in,
#         legend = True,
#         legend_kwds = {'loc': 'center', 
#                             'ncol': nlegcols_in},
#         cmap = cmap_in,
#         alpha = alpha_in,
#         zorder = 5)

#     # ax1.set(title = 'Training Catchments')
#     title = ax1.set_title('Training Catchments')
#     title.set_fontsize(8)

#     # plot valnit points
#     geo_df_valnit.plot(
#         ax = ax2,
#         column = clust_meth, 
#         markersize = msize_in, 
#         marker = marker_in,
#         cmap = cmap_in,
#         alpha = alpha_in,
#         zorder = 5)

#     # ax2.set(
#     #     title = 'Unseen Testing Catchments', 
#     #     zorder = 5)
#     title = ax2.set_title('Unseen Testing Catchments')
#     title.set_fontsize(8)

#     # turn off axis tick labels
#     ax1.tick_params(axis = 'both', which = 'both', length = 0)
#     ax2.tick_params(axis = 'both', which = 'both', length = 0)

#     # turn off plot frame
#     for spine in ax1.spines.values():
#         spine.set_visible(False)
#     for spine in ax2.spines.values():
#         spine.set_visible(False)

#     # legend position
#     leg = ax1.get_legend()
#     leg.set_bbox_to_anchor(anch_in, transform = fig.transFigure)
#     leg.set_frame_on(False)
#     # leg.get_frame().set_edgecolor('none')
#     for label in leg.get_texts():
#         label.set_fontsize(6)
#     for handle in leg.legendHandles:
#         handle.set_markersize(6)

#     leg.set_zorder(6)

#     plt.suptitle(clust_meth, fontsize = 9)

#     plt.show()