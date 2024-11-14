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
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work'\
        '/data_work/GAGESiiVariables/ID_train.csv',
    dtype = {'STAID': 'string'}
)

df_IDvalnit = pd.read_csv(
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work'\
        '/data_work/GAGESiiVariables/ID_valnit.csv',
    dtype = {'STAID': 'string'}
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
# create map of NSE results
################

# create list defining if working with ['train'], ['valint'], or both ['train', 'valnit']
# part_wrk = ['train', 'valnit']
part_wrk = ['train']
# part_wrk = ['valnit']

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

# read in mean discharge for annual and mean_annual, so can normlize residuals
df_aQ_train = pd.read_csv(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'
    f'data_work/USGS_discharge/annual/WY_Ann_train.csv', # {part_wrk[0]}.csv',
    dtype={'site_no': str}
)
df_aQ_test = pd.read_csv(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'
    f'data_work/USGS_discharge/annual/WY_Ann_valnit.csv', # {part_wrk[0]}.csv',
    dtype={'site_no': str}
)
df_aQ = pd.concat([df_aQ_train, df_aQ_test], axis=0)
df_aQ = df_aQ.rename({'site_no': 'STAID'}, axis=1)
# get mean by STAID
df_maQ = df_aQ.groupby('STAID')['Ann_WY_cm'].mean().reset_index()

# combine with id df
df_maQ = pd.merge(df_maQ, df_ID, on='STAID', how='left')
df_aQ = pd.merge(df_aQ, df_ID, on='STAID', how='left')

# calc coef of var for annual data
df_aQ = (
    df_aQ.groupby('STAID')['Ann_WY_cm'].std() /
        df_aQ.groupby('STAID')['Ann_WY_cm'].mean()
    ).reset_index()
df_aQ.columns = ['STAID', 'CV']

# monthly
df_moQ_train = pd.read_csv(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'
    'data_work/USGS_discharge/monthly/WY_Mnth_train.csv',
    dtype={'site_no': str}
)
df_moQ_test = pd.read_csv(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'
    'data_work/USGS_discharge/monthly/WY_Mnth_valnit.csv',
    dtype={'site_no': str}
)
df_moQ = pd.concat([df_moQ_train, df_moQ_test], axis=0)
df_moQ = df_moQ.rename({'site_no': 'STAID'}, axis=1)

# combine with id df
df_moQ = pd.merge(df_moQ, df_ID, on='STAID', how='left')

# calc coef of var for annual data
df_moQ = (
    df_moQ.groupby('STAID')['Mnth_WY_cm'].std() /
        df_moQ.groupby('STAID')['Mnth_WY_cm'].mean()
    ).reset_index()
df_moQ.columns = ['STAID', 'CV']


# convert each df to geodataframes
points_train = gpd.points_from_xy(long_train, lat_train, crs = states.crs)
geo_df_train = gpd.GeoDataFrame(geometry = points_train)
points_in = gpd.points_from_xy(
    df_maQ['LNG_GAGE'], df_maQ['LAT_GAGE'], crs=states.crs
    )
df_maQ = gpd.GeoDataFrame(df_maQ, geometry=points_in)
df_aQ = gpd.GeoDataFrame(df_aQ, geometry=points_in)
df_moQ = gpd.GeoDataFrame(df_moQ, geometry=points_in)

# %%
# Map
##################

# colormap
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

ma_min = df_maQ['Ann_WY_cm'].min()  #  -5 # min for color ramp
ma_max = df_maQ['Ann_WY_cm'].max()  # 5 # max for color ramp
a_min = 0.0663  # df_aQ['CV'].quantile(0.001) # .min()  #  -5 # min for color ramp
a_max = 2.62 #  df_aQ['CV'].quantile(0.99) # max()  # 5 # max for color ramp
mo_min = df_moQ['CV'].quantile(0.01)  # min() #  -5 # min for color ramp
mo_max = df_moQ['CV'].quantile(0.99)  # max()  # 5 # max for color ramp
norm1 = mpl.colors.Normalize(vmin=ma_min, vmax=ma_max)
norm2 = mpl.colors.Normalize(vmin=-1, vmax=1)

# pallete for barplot inset
# palette_in = 'tab20c_r'
# palette_in = 'tab20b'
# palette_in = 'cividis'
# palette_in = 'plasma'
# palette_in = 'viridis_r'
palette_in = 'inferno'
# markersize
ms = 50
ms2 = 8  # legend marker size
# make markers legend
downtri = mlines.Line2D([], [], color = 'black', marker = 'o',
                        linestyle = 'None', markersize = ms2,
                        label = 'AggEcoregion')

uptri = mlines.Line2D([], [], color = 'black', marker = '^',
                        linestyle = 'None', markersize = ms2,
                        label = 'Class')

circle = mlines.Line2D([], [], color = 'black', marker = 'o',
                        linestyle = 'None', markersize = ms2,
                        label = 'None')


# define base axes as state boundaries
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, 
                        figsize = (10, 12), 
                        sharex = True)

fig.subplots_adjust(hspace = 0.1, wspace = 0.0)

# fig, ax1 = plt.subplots(3, 1, figsize = (10, 12))

# state boarders
states.boundary.plot(
    ax = ax1,
    # figsize = (12, 9), 
    color = 'Gray',
    linewidth = 1,
    zorder = 0
)

states.boundary.plot(
    ax = ax2,
    # figsize = (12, 9), 
    color = 'Gray',
    linewidth = 1,
    zorder = 0
)

states.boundary.plot(
    ax = ax3,
    # figsize = (12, 9), 
    color = 'Gray',
    linewidth = 1,
    zorder = 0
)

from matplotlib.colors import LogNorm
# plot mean annual
ma_p = df_maQ.plot(
    ax = ax1,
    column = 'Ann_WY_cm', 
    markersize = ms, 
    marker = 'o',
    legend = True,
    legend_kwds = {
        'label': 'Mean Annual Discharge (cm)',
        'extend': 'both'
        },
    norm=LogNorm(vmin=df_maQ['Ann_WY_cm'].min(),
                 vmax=df_maQ['Ann_WY_cm'].max()),
    cmap = palette_in,  # cmap_in,
    vmin = ma_min, # -5,
    vmax = ma_max, # 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)
# plt.show()
################

a_p = df_aQ.plot(
    ax = ax2,
    column = 'CV', 
    markersize = ms, 
    marker = 'o',
    legend = True,
    legend_kwds = {
        'label': 'C.V. Annual Discharge (cm/cm)',
        'extend': 'both'
        },
    # norm=LogNorm(vmin=df_aQ['CV'].min(),
    #              vmax=df_aQ['CV'].max()),
    cmap = palette_in + '_r',  # cmap_in,
    vmin = a_min, # -5,
    vmax = a_max, # 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)

mo_p = df_moQ.plot(
    ax = ax3,
    column = 'CV', 
    markersize = ms, 
    marker = 'o',
    legend = True,
    legend_kwds = {
        'label': 'C.V. Monthly Discharge (cm/cm)',
        'extend': 'both'
        },
    # norm=LogNorm(vmin=df_moQ['CV'].min(),
    #              vmax=df_moQ['CV'].max()),
    cmap = palette_in + '_r',  # cmap_in,
    vmin = mo_min, # -5,
    vmax = mo_max, # 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)

# add a, b, c annotations
pos_in = [-127, 49]  # annotation location
ax1.annotate('(a)', pos_in)
ax2.annotate('(b)', pos_in)
ax3.annotate('(c)', pos_in)

ax1.set(
    title = 'Mean Annual Discharge', 
    # xlabel = 'Longitude',
    ylabel = 'Latitude')

ax2.set(
    title = 'Coefficient of Variation of Annual Discharge', 
    # xlabel = 'Longitude',
    ylabel = 'Latitude')

ax3.set(
    title = 'Coefficient of Variation of Monthly Discharge', 
    xlabel = 'Longitude',
    ylabel = 'Latitude')

name_in = 'And'.join(part_wrk)+'_'+'And'.join(models_in)
print(f'saving NSE_Map_{name_in}.png')
# save fig
if save_fig:
    plt.savefig(
        f'{dir_figs}/DischargeAndCV_AllBasins.png',
        dpi = 300,
        bbox_inches = 'tight'
        )

# %%

# calculate some basic stats

# df_in = geo_df_valnit_monthly.copy()
df_in = geo_df_valnit_annual.copy()
# df_in = geo_df_valnit_mannual.copy()

print(f'max: {df_in.temp_best.max()}')
print(f'min: {df_in.temp_best.min()}')
print(f'2.5: {np.percentile(df_in.temp_best, 2.5)}')
print(f'97.5: {np.percentile(df_in.temp_best, 97.5)}')
print(f'1: {np.percentile(df_in.temp_best, 1)}')
print(f'99: {np.percentile(df_in.temp_best, 99)}')

# %% some stats
#####################################


ma_min = df_maQ['Ann_WY_cm'].min()  #  -5 # min for color ramp
ma_max = df_maQ['Ann_WY_cm'].max()  # 5 # max for color ramp
a_min = df_aQ['CV'].min()  #  -5 # min for color ramp
a_max = df_aQ['CV'].max()  # 5 # max for color ramp
mo_min = df_moQ['CV'].min() #  -5 # min for color ramp
mo_max = df_moQ['CV'].max()

print(f'min meanQ: {ma_min}')
print(f'max meanQ: {ma_max}')
print(f'min annual CV: {a_min}')
print(f'max annual CV: {a_max}')
print(f'min monthly CV: {mo_min}')
print(f'max monthly CV: {mo_max}')