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

# legend position
leg = ax1.get_legend()
leg.set_bbox_to_anchor((0.37, -0.19))
leg.get_frame().set_edgecolor('none')

ax2.legend(
    handles = [cross, circle],
    loc = 'upper right',
    bbox_to_anchor = (0.95, 1.2))

ax2.get_legend().get_frame().set_edgecolor('none')

# # save fig
plt.savefig(
    f'{dir_figs}/RelativeContributions_WO_Climate.png', 
    dpi = 300,
    bbox_inches = 'tight'
    )






#####################################################
#####################################################














# %%
# create map of NSE results
################

# # read in shap results which hold best score
# df_shaptrain = pd.read_csv(
#     'D:/Projects/GAGESiiANNstuff/Data_Out/SHAP_OUT/MeanShap_mean_annual.csv'
# )

# read in results from independent catchments


# mean annual
# training
df_indres = pd.read_pickle(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/mean_annual/'
    'combined/All_IndResults_mean_annual.pkl'
)

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]
 


# valnit
df_indresvalnit = df_indres[df_indres['train_val'] == 'valnit']

# define empty lists to hold output

temp_sta = []
temp_best = []
temp_clust = []


for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)]#  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.min(np.abs(df_work['residuals']))
    temp_clust.append(df_work.loc[
        df_work['residuals'].abs() == best_res, 'clust_method'
        ].to_string(index = False))
    temp_best.append(df_work.loc[
        df_work['residuals'].abs() == best_res, 'residuals'].values[0])

df_bestvalnit_mannual = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'temp_best': temp_best
})

########################

# annual
# training
df_indres = pd.read_pickle(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/annual/'
    'combined/All_IndResults_annual.pkl'
)

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]
 


# valnit
df_indresvalnit = df_indres[df_indres['train_val'] == 'valnit']

# define empty lists to hold output

temp_sta = []
temp_best = []
temp_clust = []


for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)]#  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.max(df_work['NSE'])
    temp_clust.append(df_work.loc[
        df_work['NSE'] == best_res, 'clust_method'
        ].to_string(index = False))
    temp_best.append(df_work.loc[
        df_work['NSE'] == best_res, 'NSE'].values[0])

df_bestvalnit_annual = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'temp_best': temp_best
})



########################

# monthly
# training
df_indres = pd.read_pickle(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/monthly/'
    'combined/All_IndResults_monthly.pkl'
)

# drop pca columns
df_indres = df_indres[
    ~df_indres['model'].str.contains('PCA')
]
 


# valnit
df_indresvalnit = df_indres[df_indres['train_val'] == 'valnit']

# define empty lists to hold output

temp_sta = []
temp_best = []
temp_clust = []


for i in df_indresvalnit['STAID'].unique():
    temp_sta.append(i)
    df_work = df_indresvalnit[(df_indresvalnit['STAID'] == i)] #  &
        # df_indresvalnit['residual'].abs() == np.min(np.abs(df_bestvalnit))]
    best_res = np.max(df_work['NSE'])
    temp_clust.append(df_work.loc[
        df_work['NSE'] == best_res, 'clust_method'
        ].to_string(index = False))
    temp_best.append(df_work.loc[
        df_work['NSE'] == best_res, 'NSE'].values[0])

df_bestvalnit_monthly = pd.DataFrame({
    'STAID': temp_sta,
    'Region': temp_clust,
    'temp_best': temp_best
})

# merge best scores nad models with id dataframe
df_IDvalnit_mannual = pd.merge(df_IDvalnit, df_bestvalnit_mannual)
# calc inverse of residual to displays good performing better
df_IDvalnit_mannual['temp_best'] = df_IDvalnit_mannual['temp_best']
df_IDvalnit_annual = pd.merge(df_IDvalnit, df_bestvalnit_annual)
df_IDvalnit_monthly = pd.merge(df_IDvalnit, df_bestvalnit_monthly)

# %%
# prep data
##########

# make points from lat long in id files
# valnit
# mean annul
long_valnit = df_IDvalnit_mannual['LNG_GAGE']
lat_valnit = df_IDvalnit_mannual['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_mannual = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_mannual['STAID'] = df_IDvalnit['STAID']
geo_df_valnit_mannual = geo_df_valnit_mannual.merge(df_IDvalnit_mannual, on = 'STAID')

# annual
long_valnit = df_IDvalnit_annual['LNG_GAGE']
lat_valnit = df_IDvalnit_annual['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_annual = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_annual['STAID'] = df_IDvalnit['STAID']
geo_df_valnit_annual = geo_df_valnit_annual.merge(df_IDvalnit_annual, on = 'STAID')

# monthly
long_valnit = df_IDvalnit_monthly['LNG_GAGE']
lat_valnit = df_IDvalnit_monthly['LAT_GAGE']
points_valnit = gpd.points_from_xy(long_valnit, lat_valnit, crs = states.crs)
geo_df_valnit_monthly = gpd.GeoDataFrame(geometry = points_valnit)
geo_df_valnit_monthly['STAID'] = df_IDvalnit['STAID']
geo_df_valnit_monthly = geo_df_valnit_monthly.merge(df_IDvalnit_monthly, on = 'STAID')

# %%
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
    legend = True,
    cmap = 'Spectral',
    vmin = -0.5,
    vmax = 0.5,
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
    # legend = True,
    cmap = 'Spectral',
    vmin = -0.5,
    vmax = 0.5,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,

# Class
geo_df_valnit_mannual[geo_df_valnit_mannual['Region'] == 'Class'].plot(
    ax = ax1,
    column = 'temp_best', 
    markersize = 15, 
    marker = '^',
    # legend = True,
    cmap = 'Spectral',
    vmin = -1,
    vmax = 1,
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
    legend = True,
    cmap = 'Spectral',
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
    cmap = 'Spectral',
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
    cmap = 'Spectral',
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
    legend = True,
    cmap = 'Spectral',
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
    cmap = 'Spectral',
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
    cmap = 'Spectral',
    vmin = -1,
    vmax = 1,
    edgecolor = 'k',
    linewidth = 0.5,
    zorder = 5)# ,




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
    handles = [uptri, downtri, circle],
    loc = 'upper right',
    bbox_to_anchor = (0.3, 0.27))
ax2.annotate('Grouping Method', xy = (-125.5, 30.4))
ax2.annotate('NSE', xy = (-52, 36.5), rotation = 90, annotation_clip = False)
ax3.annotate('NSE', xy = (-52, 36.5), rotation = 90, annotation_clip = False)
ax1.annotate('Residuals', xy = (-52, 35.1), rotation = 90, annotation_clip = False)

# ax2.get_legend().set_label('Grouping Method')

# ax2.get_legend().get_frame().set_edgecolor('none')

# save fig
plt.savefig(
    f'{dir_figs}/NSE_Map_valnit.png', 
    dpi = 300,
    bbox_inches = 'tight'
    )

# %%
