'''
BChoat 10/20/2022

Same as GAGESii_Mapping.py except  plots more maps in one fig

Plots NSEs of ecoregions, class, and none all as individual maps
'''


# %%
# import libraries
################

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
# from shapely.geometry import Point
import numpy as np


# %%
# read in data
##################

# spatial file of us
states = gpd.read_file(
    # 'D:/DataWorking/ _20m/cb_2018_us_state_20m.shp'
    'C:/Users/bench/OneDrive/Data/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
)
# filter out Alaska, Hawaii, and Puerto Rico
states = states[~states['NAME'].isin(['Alaska', 'Puerto Rico', 'Hawaii'])]


# read in id data with lat, long, aggecoregion, and reference vs non-reference
df_IDtrain = pd.read_csv(
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files'
    '/GAGES_Work/data_work/GAGESiiVariables/ID_train.csv',
    dtype={'STAID': 'string'}
)

df_IDvalnit = pd.read_csv(
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files'
    '/GAGES_Work/data_work/GAGESiiVariables/ID_valnit.csv',
    dtype={'STAID': 'string'}
)

# directory where to write figs
# dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'
# dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures'
dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/Figures/Manuscript'

# %%
# prep data
##########

# get df_ID with all catchments
df_ID = pd.concat([df_IDtrain, df_IDvalnit], axis=0).reset_index(drop=True)


# %%
# create map of NSE results
################

# create list defining if working with ['train'], ['valint'], or both ['train', 'valnit']
# part_wrk = ['train', 'valnit']
# part_wrk = ['train']
part_wrk = ['valnit']

# create list defining which clustering approach to consider
clust_meths = ['AggEcoregion', 'Class', 'None']

# which models to include
# models_in = ['regr_precip', 'strd_mlr', 'XGBoost']
# models_in = ['regr_precip']
# models_in = ['strd_mlr']
models_in = ['XGBoost']
# # read in shap results which hold best score
# df_shaptrain = pd.read_csv(
#     'D:/Projects/GAGESiiANNstuff/Data_Out/SHAP_OUT/MeanShap_mean_annual.csv'
# )


save_fig = True

# colormap
# cmap_in = 'Spectral'
# cmap_in = 'BrBG'
# cmap_in = 'PuOr'
# cmap_in = 'gnuplot'
# cmap_in = 'cividis'
# Define the colors for the colormap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib as mpl
colors = ['#ff7f0e', 'white', '#1f77b4']  # blue, white, orange
n_bins = 100  # Use a large number of bins for smooth color transitions

# Create a colormap object
cmap_name = 'blue_white_orange'
cmap_in = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
# cmap_in = 'magma'
# cmap_in = 'rainbow'

norm1 = mpl.colors.Normalize(vmin=-5, vmax=5)
norm2 = mpl.colors.Normalize(vmin=-1, vmax=1)

# pallete for barplot inset
# palette_in = 'tab20c_r'
# palette_in = 'tab20b'
palette_in = 'cividis'


########################
# mean annual
# training
df_indres = pd.read_pickle(
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/mean_annual/'
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work'
    '/data_out/mean_annual/combined/All_IndResults_mean_annual.pkl'
)
# subset to clustering methods specified
df_indres = df_indres.query("clust_method in @clust_meths")

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]

# valnit
df_bestvalnit_mannual = df_indres[df_indres['train_val'].isin(part_wrk)]
df_bestvalnit_mannual = df_bestvalnit_mannual[
    df_bestvalnit_mannual['model'].isin(models_in)
    ]

########################

# annual
# training
df_indres = pd.read_pickle(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work'
    '/data_out/annual/combined/All_IndResults_annual.pkl'
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/annual/'
    # 'combined/All_IndResults_annual.pkl'
)

# subset to clustering methods specified
df_indres = df_indres.query("clust_method in @clust_meths")

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]

# valnit
df_bestvalnit_annual = df_indres[df_indres['train_val'].isin(part_wrk)]
df_bestvalnit_annual = df_bestvalnit_annual[
    df_bestvalnit_annual['model'].isin(models_in)
    ]

########################

# monthly
# training
df_indres = pd.read_pickle(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work'
    '/data_out/monthly/combined/All_IndResults_monthly.pkl'
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/monthly/'
    # 'combined/All_IndResults_monthly.pkl'
)

# subset to clustering methods specified
df_indres = df_indres.query("clust_method in @clust_meths")

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]


# valnit
df_bestvalnit_monthly = df_indres[df_indres['train_val'].isin(part_wrk)]
df_bestvalnit_monthly = df_bestvalnit_monthly[
    df_bestvalnit_monthly['model'].isin(models_in)
    ]

#######################
# update
######################

# merge best scores nad models with id dataframe
df_ID_mannual = pd.merge(df_ID, df_bestvalnit_mannual)
df_ID_mannual['temp_best'] = df_ID_mannual['residuals']
df_ID_annual = pd.merge(df_ID, df_bestvalnit_annual)
df_ID_annual['temp_best'] = df_ID_annual['NSE']
df_ID_monthly = pd.merge(df_ID, df_bestvalnit_monthly)
df_ID_monthly['temp_best'] = df_ID_monthly['NSE']


# %%
# prep data
##########

# make points from lat long in id files
# valnit
# mean annul
long_valnit = df_ID_mannual['LNG_GAGE']
lat_valnit = df_ID_mannual['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_mannual = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_mannual['STAID'] = df_ID_mannual['STAID']
geo_df_valnit_mannual = geo_df_valnit_mannual.merge(df_ID_mannual, on = 'STAID')

# annual
long_valnit = df_ID_annual['LNG_GAGE']
lat_valnit = df_ID_annual['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_annual = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_annual['STAID'] = df_ID_annual['STAID']
geo_df_valnit_annual = geo_df_valnit_annual.merge(df_ID_annual, on = 'STAID')

# monthly
long_valnit = df_ID_monthly['LNG_GAGE']
lat_valnit = df_ID_monthly['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_monthly = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_monthly['STAID'] = df_ID_monthly['STAID']
geo_df_valnit_monthly = geo_df_valnit_monthly.merge(df_ID_monthly, on = 'STAID')


# %% define plotting functions
#################################

# define function for plotting a map to be used in each axis
def plot_map(geo_df, region, ax, vrange, marker, zorder, show_legend):
    '''plot scatter of performance metrics'''
    # print(f'region: {region}')
    # print(f'geo_df: {geo_df.head()}')
    plot = geo_df[geo_df['clust_method'] == region].plot(
        ax=ax,
        column='temp_best',
        markersize=25,
        marker=marker,
        legend=show_legend,
        cmap=cmap_in,
        vmin=vrange[0],
        vmax=vrange[1],
        edgecolor='k',
        linewidth=0.5,
        zorder=zorder
    )

    # Return the last collection, which will be used for the colorbar
    mappable = ax.collections[-1]
    
    return mappable


def plot_states(df, ax):
    '''plot states'''
    df.boundary.plot(
        ax=ax,
        color='Gray',
        linewidth=1,
        zorder=0
    )

# %% map
######################

# Map settings
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.2, wspace=0.05)  # Control spacing between subplots

# Flatten axis array for easier indexing
axs = axs.flatten()

# Define markers for legend
downtri = mlines.Line2D([], [], color='black', marker='v', linestyle='None', markersize=5, label='AggEcoregion')
uptri = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=5, label='Class')
circle = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label='None')

# Loop over axes to plot maps
for i, ax in enumerate(axs):
    # Define dataset and visualization ranges based on index
    if i in [0, 1, 2]:
        geo_df_in = geo_df_valnit_mannual
        vrange_in = [-5, 5]
    elif i in [3, 4, 5]:
        geo_df_in = geo_df_valnit_annual
        vrange_in = [-1, 1]
    else:
        geo_df_in = geo_df_valnit_monthly
        vrange_in = [-1, 1]

    # Define region and marker based on subplot index
    if i in [0, 3, 6]:
        region_in = 'AggEcoregion'
        marker_in = 'o'
    elif i in [1, 4, 7]:
        region_in = 'Class'
        marker_in = 'o'
    else:
        region_in = 'None'
        marker_in = 'o'

    # Plot state boundaries and map data
    plot_states(states, ax)
    mappable = plot_map(geo_df_in, region_in, ax, vrange_in, 
                        marker_in, zorder=5, show_legend=False)
    
    # Create colorbars only for certain subplots
    if i in [1, 4, 7]:
        if i == 1:
            y_in = 0.645
        elif i == 4:
            y_in = 0.375
        elif i == 7:
            y_in = 0.05
        # Customize colorbar placement using add_axes()
        cbar_ax = fig.add_axes([0.415, y_in, 0.2, 0.01])  # Custom position [left, bottom, width, height]
        cbar = plt.colorbar(mappable, cax=cbar_ax, extend='both',
                            orientation='horizontal', shrink=0.8, aspect=20)
        fs = 15 # fontsize
        cbar.set_label('Residuals (cm)' if i == 1 else 'NSE', fontsize=fs)

# Label the subplots
axs[0].set_title('AggEcoregion', fontsize=fs)
axs[0].set_ylabel('Mean Annual\nLatitude', fontsize=fs)
axs[1].set_title('Class', fontsize=fs)
axs[2].set_title('None', fontsize=fs)
axs[3].set_ylabel('Annual\nLatitude', fontsize=fs)
axs[6].set_ylabel('Monthly\nLatitude', fontsize=fs)
for i in [6, 7, 8]:
    axs[i].set_xlabel('Longitude', fontsize=15)

name_in = 'And'.join(part_wrk)+'_'+'And'.join(models_in)
# save fig
# plt.tight_layout()
if save_fig:
    plt.savefig(
        # f'{dir_figs}/NSE_Map_valnit.png', 
        # f'{dir_figs}/NSE_Map_trainAndValnit.png', 
        f'{dir_figs}/NSE_Map_SepGrouping_{name_in}.png',
        dpi=300,
        bbox_inches='tight'
        )
else:
    plt.show()

# %%