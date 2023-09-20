'''
BChoat 2023/09/13

plot maps of performance metrics.


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
import os
from itertools import product

# %%
# read in data
##################

# save figure with maps w/cluster results (True or False)
save_maps  = True

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



# read in any data needed
dir_in = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results'
# mean annual
df_mnan_results = pd.read_csv(
                f'{dir_in}/PerfMetrics_MeanAnnual.csv',
                dtype = {'STAID': 'string',
                         'region': 'string'}
            )

# df_mnan_results['residuals'] = df_mnan_results.residuals.abs()

df_mnan_results = df_mnan_results[[
                'STAID', 'residuals', 'clust_method', 'region',\
                            'model', 'time_scale', 'train_val'
            ]]

# annual, montly
df_anm_results = pd.read_csv(
                f'{dir_in}/NSEComponents_KGE.csv',
                dtype = {'STAID': 'string',
                         'region': 'string'}
            )


df_IDtrain = df_IDtrain.astype(str)
df_IDvalnit = df_IDvalnit.astype(str)

# directory where to write figs
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/Maps_Performance Metrics'
if not os.path.exists(dir_figs):
    os.mkdir(dir_figs)

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


# 'train' or 'valint' data?
parts_in =['train', 'valnit']
# part_in = 'valnit'

timescales = ['mean_annual', 'annual', 'monthly']

# as list
models_in = ['XGBoost', 'regr_precip', 'strd_mlr'] # ['regr_precip', 'strd_mlr', 'XGBoost]

metrics_in = ['residuals', 'KGE', 'NSE']

remove_noises = [True, False]

alpha_in = 1
marker_in = 'o' 
# markers_in = ['d', 'o', '+']
# ',' # 'x'
msize_in = 1.5 # markersize
cmap_str = 'Spectral' # 'RdYlBu' #'terrain' 'viridis' #  # 'bwr_r' # 'cool' # 'gist_heat' # 'hsv' # 'plasma' #  # 


clust_meths_in =  [
       'Class', 'CAMELS', 'AggEcoregion', 'HLR',
       'All_0', 'All_1', 'All_2', 
       'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4',
       'Anth_0', 'Anth_1', 'None'
       ]




# %% plot performance metrics
################################################

for part_in, timescale, model_in, remove_noise, metric_in in \
    product(parts_in, timescales, models_in, remove_noises, metrics_in):

    print(part_in, timescale, model_in, remove_noise, metric_in)

    if (timescale == 'mean_annual') & (metric_in != 'residuals'):
        continue
    if (timescale != 'mean_annual') & (metric_in == 'residuals'):
        continue
        

    if part_in == 'train':
        df_id = df_IDtrain
        geo_df_in = geo_df_train

    else:
        df_id = df_IDvalnit
        geo_df_in = geo_df_valnit

        
    if timescale == 'mean_annual':
        df_temp = df_mnan_results.loc[
            (df_mnan_results['train_val'] == part_in) &\
                (df_mnan_results['model'] == model_in), 
                ['STAID', metric_in, 'clust_method', 'region']
            ]
        
        vmin_in = -7.5
        vmax_in = 7.5
        ticks_in = np.arange(vmin_in, vmax_in, 1)
        # metric_in = 'residuals'
    else:
        df_temp = df_anm_results.loc[
            (df_anm_results['train_val'] == part_in) &\
                (df_anm_results['model'] == model_in) &\
                (df_anm_results['time_scale'] == timescale), 
                ['STAID', metric_in, 'clust_method', 'region']
        ]

        vmin_in = -1
        vmax_in = 1
        ticks_in = np.arange(vmin_in, vmax_in, 0.25)



    geo_df_in = pd.merge(geo_df_in[['STAID', 'LAT_GAGE', 'LNG_GAGE', 'geometry']], df_temp, 
                            on = ['STAID'])

    if remove_noise:
        geo_df_in = geo_df_in[geo_df_in['region'] != '-1']

    fig, axs = plt.subplots(5, 3, figsize = (10, 10))

    axes = axs.flatten()

    for i, ax1 in enumerate(axes):

        row, col = divmod(i, 3)
        ax1.sharex(axes[col])
        ax1.sharey(axes[row])

        if i < 14:
            clust_meth = clust_meths_in[i]
            geo_df_plot = geo_df_in[geo_df_in['clust_method'] == clust_meth]

            # get number of clusters
            Nclust  = len(geo_df_train[clust_meth].unique())
            # get median of per metric for reporting
            q50 = geo_df_plot[metric_in].median().round(4)
            # define title for each subplot
            sp_title = f'{clust_meth}: median({metric_in})={q50})' # ({Nclust} clusters,
            

        else:
            if timescale == 'mean_annual':
                # define function to return closest to zero
                # def closest_to_zero(series):
                #     return series.abs().min()
                geo_df_plot = geo_df_in.copy()
                geo_df_plot['DiffZero'] = geo_df_plot['residuals'].abs()
                geo_df_plot.sort_values(by = 'DiffZero', inplace = True)
                geo_df_plot = geo_df_plot.groupby(
                    ['STAID', 'LAT_GAGE', 'LNG_GAGE']
                    )[metric_in].first().reset_index() # agg(closest_to_zero).reset_index() # max().
                # geo_df_plot.drop('DiffZero')
                # if '_r' in cmap_in:
                #     cmap_in = cmap_in.split('_')[0]
                # else:
                #     cmap_in = cmap_in + '_r'
            else:
                geo_df_plot = geo_df_in.groupby('STAID').max()
            geo_df_plot = gpd.GeoDataFrame(
                geo_df_plot, geometry = gpd.points_from_xy(
                    geo_df_plot['LNG_GAGE'], geo_df_plot['LAT_GAGE']
                ),
                crs = geo_df_in.crs
            )
            # define title for each subplot
            sp_title = f'Best Scores'

        # define colormap
        cmap_in = cmap_str

        states.boundary.plot(
            ax = ax1,
            # figsize = (12, 9), 
            color = 'black',
            linewidth = 0.2,
            zorder = 0
        )

        # if i == 14:
        #     legend_in = True
        # else:
        legend_in = False

        # plot training points
        geo_df_plot.plot(
            ax = ax1,
            column = metric_in, 
            markersize = msize_in, 
            marker = marker_in,
            legend = legend_in,
            cmap = cmap_in,
            vmin = vmin_in, vmax = vmax_in,
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

    fig.subplots_adjust(bottom=0.025, top=0.95, left=0.025, right=0.85,
                        wspace=0., hspace=0.0)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.90, 0.3, 0.015, 0.4])
    p = ax1.scatter([], [], c=[], cmap=cmap_in, vmin=vmin_in, vmax=vmax_in)
    cbar = fig.colorbar(p, cax=cb_ax)
    
    #  set the colorbar ticks and tick labels
    if timescale == 'mean_annual':
        cbar.set_ticks(ticks_in)
        cbar.set_ticklabels(ticks_in)
        cbar.ax.text(0.9, 1.05, metric_in + '(cm)', transform=cbar.ax.transAxes,
                va='center', ha='center', fontsize=12)  # Adjust the position as needed
    else:
        cbar.set_ticks(ticks_in)
        cbar.set_ticklabels(ticks_in)
        cbar.ax.text(0.9, 1.05, metric_in, transform=cbar.ax.transAxes,
                va='center', ha='center', fontsize=12)  # Adjust the position as needed
    

    if part_in == 'train': 
        part_name = 'training'
    if part_in == 'valnit': 
        part_name = 'testing'
    if remove_noise: 
        ns_name = 'w/o Noise'
        fl_name = 'woNoise'
    else:
        ns_name = 'w/Noise'
        fl_name = 'wNoise'

    plt.suptitle(f'{metric_in} for {timescale} {part_name} data using {model_in} {ns_name}')

    if save_maps:
        plt.savefig(f'{dir_figs}/MapOfPerMetrics_{timescale}_{model_in}_{metric_in}_{part_name}_{fl_name}.png',
                    dpi = 300)
    else:
        plt.show()

# %%
