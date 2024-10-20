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
dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures'

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


save_fig = False

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


# define function for plotting a map to be used in each axis
def plot_map(geo_df, region, ax, vrange, marker, zorder, show_legend):
    # print(f'region: {region}')
    # print(f'geo_df: {geo_df.head()}')
    geo_df[geo_df['clust_method'] == region].plot(
        ax=ax,
        column='temp_best',
        markersize=15,
        marker=marker,
        legend=show_legend,
        cmap=cmap_in,
        vmin=vrange[0],
        vmax=vrange[1],
        edgecolor='k',
        linewidth=0.5,
        zorder=zorder
    )


def plot_states(df, ax):
    df.boundary.plot(
        ax=ax,
        color='Gray',
        linewidth=1,
        zorder=0
    )


# Map
##################

# %%
# make markers legend
downtri = mlines.Line2D([], [], color = 'black', marker = 'v',
                        linestyle = 'None', markersize = 5,
                        label = 'AggEcoregion')

uptri = mlines.Line2D([], [], color = 'black', marker = '^',
                        linestyle = 'None', markersize = 5,
                        label = 'Class')

circle = mlines.Line2D([], [], color='black', marker='o',
                        linestyle='None', markersize=5,
                        label = 'None')


# define base axes as state boundaries
fig, axs = plt.subplots(3, 3,
                        figsize=(12, 12),
                        sharex=True,
                        sharey=True)

fig.subplots_adjust(hspace=0.01, wspace=0.01)

axs = axs.flatten()

for i, ax in enumerate(axs):
    if i in [0, 1, 2]:
        geo_df_in = geo_df_valnit_mannual
        vrange_in = [-5, 5]
    if i in [3, 4, 5]:
        geo_df_in = geo_df_valnit_annual
        vrange_in = [-1, 1]
    if i in [6, 7, 8]:
        geo_df_in = geo_df_valnit_monthly
        vrange_in = [-1, 1]
    if i in [0, 3, 6]:
        region_in = 'AggEcoregion'
        marker_in = 'v'
    if i in [1, 4, 7]:
        region_in = 'Class'
        marker_in = '^'
    if i in [2, 5, 8]:
        region_in = 'None'
        marker_in = 'o'

    # print(f'i: {i}')
    # print(f'geo_df_in: {geo_df_in}')

    plot_states(states, ax)
    plot_map(geo_df_in, region_in, ax, vrange_in, marker_in, zorder=5, show_legend=False)

axs[0].set(
    title = 'AggEcoregion', 
    # xlabel = 'Longitude',
    ylabel = 'Mean Annual\nLatitude')
axs[1].set(title='Class')
axs[2].set(title='None')
axs[3].set(ylabel = 'Annual\nLatitude')
axs[6].set(ylabel = 'Monthly\nLatitude')
for i in [6, 7, 8]:
    axs[i].set(
        # title = 'Monthly', 
        xlabel = 'Longitude'
    ) # ,
        # ylabel = 'Monthly\nLatitude')

# axs[2].legend(
#     handles = [downtri, uptri, circle],
#     labels = ['Eco', 'Class', 'None'],
#     loc = 'upper right',
#     fontsize=9,
#     bbox_to_anchor = (0.95, 0.27),
#     frameon=False)
# axs[2].annotate('Grouping Method', xy = (-79.5, 30.4))

# modify the colorbars
# Create a ScalarMappable object
sm1 = plt.cm.ScalarMappable(cmap=cmap_in, norm=norm1)
sm1._A = [] # ensure scalarmappable is properly initialized

sm2 = plt.cm.ScalarMappable(cmap=cmap_in, norm=norm2)
sm2._A = [] # ensure scalarmappable is properly initialized

# Add colorbars with extended arrow tips to each subplot
cbar1 = fig.colorbar(sm1, ax=axs[2], extend='both')
cbar1.set_label('Residuals (cm)')

cbar2 = fig.colorbar(sm2, ax=axs[5], extend='min')
cbar2.set_label('NSE')

cbar3 = fig.colorbar(sm2, ax=axs[8], extend='min')
cbar3.set_label('NSE')

# name_in = 'And'.join(part_wrk)+'_'+'And'.join(models_in)
# print(f'saving NSE_Map_{name_in}.png')
# # save fig
# if save_fig:
#     plt.savefig(
#         # f'{dir_figs}/NSE_Map_valnit.png', 
#         # f'{dir_figs}/NSE_Map_trainAndValnit.png', 
#         f'{dir_figs}/NSE_Map_BestOfGrouping{name_in}.png',
#         dpi = 300,
#         bbox_inches = 'tight'
#         )

# %%
