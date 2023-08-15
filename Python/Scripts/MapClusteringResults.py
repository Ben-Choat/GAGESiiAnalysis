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
# import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import seaborn as sns
# from shapely.geometry import Point
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score


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




# %% Input vars for plotting
################################

# 'training' or 'valint' data?
# part_in = 'training'
part_in = 'valnit'

alpha_in = 1
marker_in = 'o' 
# ',' # 'x'
msize_in = 2 # markersize
cmap_str = 'tab20b'


clust_meths_in =  [
       'Class', 'CAMELS', 'AggEcoregion', 'HLR',
       'All_0', 'All_1', 'All_2', 
       'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
       'Anth_0', 'Anth_1'
       ]


# %% Calculate AMI for plotting
###################################

if part_in == 'training':
    df_id = df_IDtrain
    geo_df_in = geo_df_train
else:
    df_id = df_IDvalnit
    geo_df_in = geo_df_valnit


# create an empty dataframe to hold results
df_ami = pd.DataFrame(columns = clust_meths_in, index = clust_meths_in)

# loop through all combinations of two dataframe names (i.e., cluster results)
for c1 in df_ami.columns:
    for c2 in df_ami.index:
        # print(c1, c2)
        # calc adjusted mutual information index
        # AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
        ami_temp = adjusted_mutual_info_score(df_id[c1], df_id[c2])
        
        # add ami_tempto df_ami
        df_ami.loc[c1, c2] = ami_temp

# convert df_ami dtype to float for heatmap
df_ami = df_ami.astype(float)

# ht_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.75]
# wd_ratios = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.75]

# define base axes as state boundaries
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (3, 3), sharex = True)
fig, axs = plt.subplots(5, 3, figsize = (10, 11)) # ,
                        # gridspec_kw = {'height_ratios': ht_ratios, 
                        #                'width_ratios': wd_ratios}) # , 
                                # sharex = True, sharey = True)

# fig = plt.figure(figsize = (12, 9))
# ax = fig.add_subplot(111)

# new_fig = plt.figure()
axes = axs.flatten()

for i, ax1 in enumerate(axes): # in range(15):
    # ax1 = axes[i]
    if i != 14:
        row, col = divmod(i, 3)
        ax1.sharex(axes[col])
        ax1.sharey(axes[row])
        clust_meth = clust_meths_in[i]
        # get number of clusters
        Nclust  = len(geo_df_train[clust_meth].unique())

        # define colormap
        cm_org = plt.get_cmap(cmap_str)
        
        if clust_meth in ['All_0', 'All_1', 'All_2', 
                        'Nat_0','Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
                        'Anth_0', 'Anth_1']:   
            add_colors = ['black', 'orange', 'cyan', 'peru', 'magenta',  'aquamarine', 
                        'blueviolet', 'lime']
        else:
            add_colors = ['orange', 'cyan', 'peru', 
                        'magenta', 'aquamarine', 'blueviolet', 'lime']
        add_rgb = [mcolors.to_rgb(col) for col in add_colors]
        all_colors = add_rgb + list(cm_org.colors)
        all_colors = all_colors[0:Nclust]
        cm_cust = mcolors.ListedColormap(all_colors)
        cmap_in = cm_cust

        # define title for each subplot
        sp_title = f'{clust_meth} ({Nclust} clusters)'

        states.boundary.plot(
            ax = ax1,
            # figsize = (12, 9), 
            color = 'Gray',
            linewidth = 1,
            zorder = 0
        )
    
        # plot training points
        geo_df_in.plot(
            ax = ax1,
            column = clust_meth, 
            markersize = msize_in, 
            marker = marker_in,
            legend = False,
            cmap = cmap_in,
            alpha = alpha_in,
            zorder = 5)

        # ax1.set(title = 'Training Catchments')
        title = ax1.set_title(sp_title, fontsize = 8)

        # turn off axis tick labels
        ax1.set_xticks([])
        ax1.set_yticks([])

        # turn off plot frame
        for spine in ax1.spines.values():
            spine.set_visible(False)

    else:
        mask = np.tri(*df_ami.shape, k = -1).T

        # Plot the correlogram heatmap
        # plt.figure(figsize=(10, 8))
        hm = sns.heatmap(df_ami, 
                    mask = mask,
                    annot = False, # show ami value
                    fmt = '.2f', # '.2f'two decimal places
                    cmap = 'seismic',
                    vmin = 0,
                    vmax = 1,
                    ax = ax1,
                    annot_kws = {'fontsize': 10})
        # xticks
        xticks = np.arange(0.5, len(clust_meths_in), step=1)
        hm.set_xticks(xticks)

        hm.set_xticklabels(
            labels = clust_meths_in,
            rotation = 45,
            ha = 'right',
            rotation_mode = 'anchor'
        )
        # yticks
        hm.set_yticks(xticks)

        hm.set_yticklabels(
            labels = clust_meths_in,
            rotation = 35,
            ha = 'right',
            rotation_mode = 'anchor'
        )

        # make cbar labels larger
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize = 7)
        cbar.set_label('Adjusted Mutual Information', fontsize = 7)


        # make tick labels larger
        plt.xticks(fontsize = 7)
        plt.yticks(fontsize = 7)
        

        plt.title('Similarity Between All Approaches' , fontsize = 8)
                        

    if part_in == 'training':
        lab_in = 'Training'
    else:
        lab_in = 'Testing'
    plt.suptitle(f'Clustering Results for {lab_in} Data', fontsize = 9,
                 y = 0.9)

# adjust white space
plt.subplots_adjust(wspace = 0.25, hspace = 0.0)

# fig.tight_layout()

plt.show()









# %% plot original HLR data
###################

# get number of clusters
Nclust  = len(geo_df_train['HLR'].unique())

# define colormap
cm_org = plt.get_cmap(cmap_str)

add_colors = ['black', 'sandybrown', 'cyan', 'peru', 'hotpink',  'aquamarine', 
                'blueviolet', 'lime']

add_rgb = [mcolors.to_rgb(col) for col in add_colors]
all_colors = add_rgb + list(cm_org.colors)
all_colors = all_colors[0:Nclust]
cm_cust = mcolors.ListedColormap(all_colors)
cmap_in = cm_cust


# HLC dir
dir_hl = 'D:/DataWorking/hlrshape/hlrshape'

hl_shp = gpd.read_file(
    f'{dir_hl}/hlrus.shp'
)

hl_shp['HLR'] = hl_shp['HLR'].astype(str)


ax = hl_shp.plot(column = 'HLR', cmap = cmap_in)
ax.set_xlim(-2.1e6, 2.7e6)
ax.set_ylim(-2.2e6, 1e6)

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
