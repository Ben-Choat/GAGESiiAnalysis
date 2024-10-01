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
dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures'

# %%
# prep data
##########

# get df_ID with all catchments
df_ID = pd.concat([df_IDtrain, df_IDvalnit], axis = 0).reset_index(drop=True)


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

# add color and marker columns to data frame based on aggecoregion
# and class, respectively
# geo_df_train['Marker'] = 0
# geo_df_train.loc[
#     geo_df_train['Class'] == 'Ref', 'Marker'
#     ] = '^'
# geo_df_train.loc[
#     geo_df_train['Class'] == 'Non-ref', 'Marker'
#     ] = 'o'

# %%
# map
############

# make markers legend
cross = mlines.Line2D([], [], color = 'black', marker = 'x',
                        linestyle = 'None', markersize = 5,
                        label = 'Reference')

circle = mlines.Line2D([], [], color = 'black', marker = 'o',
                        linestyle = 'None', markersize = 5,
                        label = 'Non-Reference')



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
geo_df_train[geo_df_train['Class'] == 'Ref'].plot(
    ax = ax1,
    column = 'AggEcoregion', 
    markersize = 20, 
    marker = 'x',
    legend = True,
    legend_kwds = {'loc': 'lower center', 
                        'ncol': 4},
    zorder = 5)

# ax1.get_legend().set_bbox_to_anchor((20, 25, -120, -80))
geo_df_train[geo_df_train['Class'] == 'Non-ref'].plot(
    ax = ax1,
    column = 'AggEcoregion', 
    markersize = 5, 
    marker = 'o',
    zorder = 5) # '*')
ax1.set(
    title = 'Training Catchments',
    ylabel = 'Latitude')

# plot valnit points
geo_df_valnit[geo_df_valnit['Class'] == 'Ref'].plot(
    ax = ax2,
    column = 'AggEcoregion', 
    markersize = 20, 
    marker = 'x',
    zorder = 5)

geo_df_valnit[geo_df_valnit['Class'] == 'Non-ref'].plot(
    ax = ax2,
    column = 'AggEcoregion', 
    markersize = 5, 
    marker = 'o') # '*')
ax2.set(
    title = 'Unseen Testing Catchments', 
    xlabel = 'Longitude',
    ylabel = 'Latitude',
    zorder = 5)

# annotations 
ax1.annotate("n=2,240", (-125, 25), fontsize=15)
ax2.annotate("n=960", (-125, 25), fontsize=15)
# legend position
leg = ax1.get_legend()
leg.set_bbox_to_anchor((0.37, -0.19))
leg.get_frame().set_edgecolor('none')

ax2.legend(
    handles = [cross, circle],
    loc = 'upper right',
    bbox_to_anchor = (0.95, 1.2))

ax2.get_legend().get_frame().set_edgecolor('none')

# save fig
# plt.savefig(
#     f'{dir_figs}/Map_AggEcoregion_RefOrNot.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )






#####################################################
#####################################################














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

# temp_cmap = ListedColormap(['purple', 'cyan', 'gray'])
# try: 
#     pal_in_str = mpl.colormaps.register(cmap=temp_cmap)
# except:
#     print('cmap already registered')

# palette_in = 'palette_in'

# read in results from independent catchments


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
df_indresvalnit = df_indres[df_indres['train_val'].isin(part_wrk)]
df_indresvalnit =df_indresvalnit[df_indresvalnit['model'].isin(models_in)]

# define empty lists to hold output

temp_sta = []
temp_best = []
temp_clust = []
temp_models = []


for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)]#  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.min(np.abs(df_work['residuals']))
    temp_clust.append(df_work.loc[
        df_work['residuals'].abs() == best_res, 'clust_method'
        ].to_string(index = False))
    temp_models.append(df_work.loc[
        df_work['residuals'].abs() == best_res, 'model'
        ].to_string(index = False))
    if temp_models[-1] == 'Series([], )':
        break
    temp_best.append(df_work.loc[
        df_work['residuals'].abs() == best_res, 'residuals'].values[0])

df_bestvalnit_mannual = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'model': temp_models,
    'temp_best': temp_best
})

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
df_indresvalnit = df_indres[df_indres['train_val'].isin(part_wrk)]
df_indresvalnit = df_indresvalnit[df_indresvalnit['model'].isin(models_in)]

# define empty lists to hold output
temp_sta = []
temp_best = []
temp_clust = []
temp_models = []


for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)]#  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.max(df_work['NSE'])
    temp_clust.append(df_work.loc[
        df_work['NSE'] == best_res, 'clust_method'
        ].to_string(index = False))
    temp_models.append(df_work.loc[
        df_work['NSE'] == best_res, 'model'
        ].to_string(index = False))
    temp_best.append(df_work.loc[
        df_work['NSE'] == best_res, 'NSE'].values[0])
    

df_bestvalnit_annual = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'model': temp_models,
    'temp_best': temp_best
})

# def get_max_score(group):
#     # print(group)
#     return group.loc[group['NSE'].idxmax()]
# test = df_indresvalnit.groupby('STAID').apply(get_max_score)


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
df_indresvalnit = df_indres[df_indres['train_val'].isin(part_wrk)]
df_indresvalnit = df_indresvalnit[df_indresvalnit['model'].isin(models_in)]

# define empty lists to hold output

temp_sta = []
temp_best = []
temp_clust = []
temp_models = []

for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)] #  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.max(df_work['NSE'])
    temp_clust.append(df_work.loc[
        df_work['NSE'] == best_res, 'clust_method'
        ].to_string(index = False))
    temp_models.append(df_work.loc[
        df_work['NSE'] == best_res, 'model'
        ].to_string(index = False))
    temp_best.append(df_work.loc[
        df_work['NSE'] == best_res, 'NSE'].values[0])

df_bestvalnit_monthly = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'model': temp_models,
    'temp_best': temp_best
})

# merge best scores nad models with id dataframe
df_ID_mannual = pd.merge(df_ID, df_bestvalnit_mannual)
# calc inverse of residual to displays good performing better
df_ID_mannual['temp_best'] = df_ID_mannual['temp_best']
df_ID_annual = pd.merge(df_ID, df_bestvalnit_annual)
df_ID_monthly = pd.merge(df_ID, df_bestvalnit_monthly)

## %%
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

## %%
# Map
##################


# make markers legend
downtri = mlines.Line2D([], [], color = 'black', marker = 'v',
                        linestyle = 'None', markersize = 5,
                        label = 'AggEcoregion')

uptri = mlines.Line2D([], [], color = 'black', marker = '^',
                        linestyle = 'None', markersize = 5,
                        label = 'Class')

circle = mlines.Line2D([], [], color = 'black', marker = 'o',
                        linestyle = 'None', markersize = 5,
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


# plot mean annual
# AggEco
geo_df_valnit_mannual[geo_df_valnit_mannual['Region'] == 'AggEcoregion'].plot(
    ax = ax1,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'v',
    legend = False,
    cmap = cmap_in,
    vmin = -5,
    vmax = 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,
    # legend_kwds = {'loc': 'lower center', 
    #                     'ncol': 4})

# None
geo_df_valnit_mannual[geo_df_valnit_mannual['Region'] == 'None'].plot(
    ax = ax1,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'o',
    legend = False,
    cmap = cmap_in,
    vmin = -5,
    vmax = 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,

# Class
geo_df_valnit_mannual[geo_df_valnit_mannual['Region'] == 'Class'].plot(
    ax = ax1,
    column = 'temp_best', 
    markersize = 15, 
    marker = '^',
    legend = False,
    cmap = cmap_in,
    vmin = -5,
    vmax = 5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,


################

# plot  annual
# AggEco
geo_df_valnit_annual[geo_df_valnit_annual['Region'] == 'AggEcoregion'].plot(
    ax = ax2,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'v',
    legend = False,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,
    # legend_kwds = {'loc': 'lower center', 
    #                     'ncol': 4})

# None
geo_df_valnit_annual[geo_df_valnit_annual['Region'] == 'None'].plot(
    ax = ax2,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'o',
    # legend = True,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,

# Class
geo_df_valnit_annual[geo_df_valnit_annual['Region'] == 'Class'].plot(
    ax = ax2,
    column = 'temp_best', 
    markersize = 15, 
    marker = '^',
    # legend = True,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,


################

# plot monthly
# AggEco
geo_df_valnit_monthly[geo_df_valnit_monthly['Region'] == 'AggEcoregion'].plot(
    ax = ax3,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'v',
    legend = False,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,
    # legend_kwds = {'location': 'right'})

# None
geo_df_valnit_monthly[geo_df_valnit_monthly['Region'] == 'None'].plot(
    ax = ax3,
    column = 'temp_best', 
    markersize = 15, 
    marker = 'o',
    # legend = True,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,

# Class
geo_df_valnit_monthly[geo_df_valnit_monthly['Region'] == 'Class'].plot(
    ax = ax3,
    column = 'temp_best', 
    markersize = 15, 
    marker = '^',
    # legend = True,
    cmap = cmap_in,
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,

###############################
# barplots showing how many scores are from which regionalization approach
mannual_count = geo_df_valnit_mannual.groupby('Region')['STAID'].count()
annual_count = geo_df_valnit_annual.groupby('Region')['STAID'].count()
monthly_count = geo_df_valnit_monthly.groupby('Region')['STAID'].count()

axbar1 = ax1.inset_axes([0.11, 0.14, 0.22, 0.17])
order_in = ['AggEcoregion', 'Class', 'None']
hue_order_in = ['regr_precip', 'strd_mlr', 'XGBoost']
xticklabs_in = mannual_count.index.str.replace('AggEcoregion', 'Eco')

axbar1.patch.set_alpha(0)
# turn off frame
# for spine in axbar1.spines.values():
#     spine.set_visible(False)
sns.countplot(
    data = geo_df_valnit_mannual,
    x = 'Region',
    # hue = 'model',
    order = order_in,
    # hue_order = hue_order_in,
    stat = 'percent',
    # palette = palette_in,
    color='gray',
    ax = axbar1
)

# axbar1.legend(loc='center left', fontsize = 9,
#                 labels = ['SLR', 'MLR', 'XGBoost'],
#                 bbox_to_anchor=(2.95, 0.75), ncol = 1,
#                 frameon = False,
#                 title = 'Model')
axbar1.tick_params(axis='x', labelsize=8)
axbar1.set_ylabel('% Best Predicted', rotation = 90)
axbar1.set_xlabel('')
axbar1.set(xticks=[0, 1, 2], 
           xticklabels=xticklabs_in,
           facecolor='none',
           frame_on=False)
# barp = axbar1.bar([1, 2, 3], mannual_count)
# Add count to the top of each bar

# for bar in barp:
#     height = bar.get_height()
#     axbar1.text(bar.get_x() + bar.get_width() / 2, 
#                 height, str(height),
#                 size=8,
#                 ha='center', va='bottom')
# # set attributes
# axbar1.set(xticks=[1, 2, 3], 
#            xticklabels=mannual_count.index,
#            facecolor='none',
#            frame_on=False)
# # adjust fontsize
# axbar1.tick_params(axis='both', which='major', labelsize=8) 
######################

axbar2 = ax2.inset_axes([0.11, 0.14, 0.22, 0.17])
axbar2.patch.set_alpha(0)
# turn off frame
for spine in axbar2.spines.values():
    spine.set_visible(False)
sns.countplot(
    data = geo_df_valnit_annual,
    x = 'Region',
    # hue = 'model',
    order = order_in,
    # hue_order = hue_order_in,
    # palette = palette_in,
    color='gray',
    stat = 'percent',
    ax = axbar2
)

axbar2.legend(loc='center left', bbox_to_anchor=(2, 0.75), ncol = 1)
axbar2.legend().remove()
axbar2.tick_params(axis='x', labelsize=8)
axbar2.set_ylabel('% Best Predicted', rotation = 90)
axbar2.set_xlabel('')
axbar2.set(xticks=[0, 1, 2], 
           xticklabels=xticklabs_in,
           facecolor='none',
           frame_on=False)

# barp = axbar2.bar([1, 2, 3], annual_count)
# # Add count to the top of each bar
# for bar in barp:
#     height = bar.get_height()
#     axbar2.text(bar.get_x() + bar.get_width() / 2, 
#                 height, str(height),
#                 size=8,
#                 ha='center', va='bottom')
# # set attributes
# axbar2.set(xticks=[1, 2, 3], 
#            xticklabels=annual_count.index,
#            facecolor='none',
#            frame_on=False)
# # adjust fontsize
# axbar2.tick_params(axis='both', which='major', labelsize=8) 

################
axbar3 = ax3.inset_axes([0.11, 0.14, 0.22, 0.17])
axbar3.patch.set_alpha(0)
# turn off frame
for spine in axbar3.spines.values():
    spine.set_visible(False)
sns.countplot(
    data = geo_df_valnit_monthly,
    x = 'Region',
    # hue = 'model',
    order = order_in,
    # hue_order = hue_order_in,
    stat = 'percent',
    # palette = palette_in,
    color='gray',
    ax = axbar3
)

axbar3.legend(loc='center left', bbox_to_anchor=(2, 0.75), ncol = 1)
axbar3.legend().remove()
axbar3.tick_params(axis='x', labelsize=8)
axbar3.set_ylabel('% Best Predicted', rotation = 90)
axbar3.set_xlabel('')
axbar3.set(xticks=[0, 1, 2], 
           xticklabels=xticklabs_in,
           facecolor='none',
           frame_on=False)

# barp = axbar3.bar([1, 2, 3], monthly_count)
# # Add count to the top of each bar
# for bar in barp:
#     height = bar.get_height()
#     axbar3.text(bar.get_x() + bar.get_width() / 2, 
#                 height, str(height),
#                 size=8,
#                 ha='center', va='bottom')
# # set attributes
# axbar3.set(xticks=[1, 2, 3], 
#            xticklabels=monthly_count.index,
#            facecolor='none',
#            frame_on=False)
# # adjust fontsize
# axbar3.tick_params(axis='both', which='major', labelsize=8) 


##############
# fig.subplots_adjust(bottom = 0.1)
# cbar_ax = fig.add_axes([0.2, 0.05, 0.2, 0.8])
# fig.colorbar(
#     ax2,  #norm=norm,
#      ax = ax3, location = 'bottom', anchor = (0, 0))
# ax2.legend(label = 'NSE')
ax1.set(
    title = 'Mean Annual', 
    # xlabel = 'Longitude',
    ylabel = 'Latitude')

ax2.set(
    title = 'Annual', 
    # xlabel = 'Longitude',
    ylabel = 'Latitude')

ax3.set(
    title = 'Monthly', 
    xlabel = 'Longitude',
    ylabel = 'Latitude')

# # # legend position
# leg = ax3.get_legend()
# leg.set_bbox_to_anchor((0.37, -0.19))
# leg.get_frame().set_edgecolor('none')

ax2.legend(
    handles = [downtri, uptri, circle],
    labels = ['Eco', 'Class', 'None'],
    loc = 'upper right',
    fontsize=9,
    bbox_to_anchor = (0.95, 0.27),
    frameon=False)
ax2.annotate('Grouping Method', xy = (-79.5, 30.4)) # (-125.5, 30.4))
# ax2.annotate('NSE', xy = (-52, 36.5), rotation = 90, annotation_clip = False)
# ax3.annotate('NSE', xy = (-52, 36.5), rotation = 90, annotation_clip = False)
# ax1.annotate('Residuals [cm]', xy = (-52, 35.1), rotation = 90, annotation_clip = False)



# modify the colorbars
# Create a ScalarMappable object
sm1 = plt.cm.ScalarMappable(cmap=cmap_in, norm=norm1)
sm1._A = [] # ensure scalarmappable is properly initialized

sm2 = plt.cm.ScalarMappable(cmap=cmap_in, norm=norm2)
sm2._A = [] # ensure scalarmappable is properly initialized

# Add colorbars with extended arrow tips to each subplot
cbar1 = fig.colorbar(sm1, ax=ax1, extend='both')
cbar1.set_label('Residuals (cm)')

cbar2 = fig.colorbar(sm2, ax=ax2, extend='min')
cbar2.set_label('NSE')

cbar3 = fig.colorbar(sm2, ax=ax3, extend='min')
cbar3.set_label('NSE')

# ax2.get_legend().set_label('Grouping Method')

# ax2.get_legend().get_frame().set_edgecolor('none')
name_in = 'And'.join(part_wrk)+'_'+'And'.join(models_in)
print(f'saving NSE_Map_{name_in}.png')
# save fig
if save_fig: 
    plt.savefig(
        # f'{dir_figs}/NSE_Map_valnit.png', 
        # f'{dir_figs}/NSE_Map_trainAndValnit.png', 
        f'{dir_figs}/NSE_Map_BestOfGrouping{name_in}.png',
        dpi = 300,
        bbox_inches = 'tight'
        )

# %%
