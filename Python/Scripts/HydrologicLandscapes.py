'''
BChoat 2022/11/23

Script to explore data from Winter 2001 - Hydrologic Landscape Concept.

associate catchment outlets with HLC
'''

# %%
# Import libraries
#############

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # for removing axis tick labels


# %%
# define variables and directories
################

# HLC dir
dir_hl = 'D:/DataWorking/hlrshape/hlrshape'

# GAGES dir
dir_gg = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/GAGESiiVariables/'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'


# %% 
# load data
##############

hl_shp = gpd.read_file(
    f'{dir_hl}/hlrus.shp'
)

# read in id data with lat, long, aggecoregion, and reference vs non-reference
# read in ID file (e.g., holds aggecoregions) with gauges available from '98-'12
df_ID = pd.read_csv(
    f'{dir_expl}/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
    )

# df_ID = pd.read_csv(
#     f'{dir_gg}/ID_train.csv',
#     dtype = {'STAID': 'string'}
# )

# df_ID = pd.read_csv(
#     f'{dir_expl}/ID_all_avail98_12.csv',
#     dtype = {'STAID': 'string'}
# ).drop(['CLASS', 'AGGECOREGION'], axis = 1)

# df_IDvalnit = pd.read_csv(
#     f'{dir_gg}/ID_valnit.csv',
#     dtype = {'STAID': 'string'}
# )

# spatial file of us
states = gpd.read_file(
    'D:/DataWorking/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
)
# filter out Alaska, Hawaii, and Puerto Rico
states = states[~states['NAME'].isin(['Alaska', 'Puerto Rico', 'Hawaii'])]

# convert to Albers projection
states = states.to_crs({'init': 'epsg:5070'})

# # directory where to write figs
# dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'

# %%
# prep data
##########

# make points from lat long in id file
long_gg = df_ID['LNG_GAGE']
lat_gg = df_ID['LAT_GAGE']
points_gg = gpd.points_from_xy(long_gg, lat_gg, crs = 'epsg:4326')
geo_df = gpd.GeoDataFrame(geometry = points_gg)
geo_df['STAID'] = df_ID['STAID']
geo_df = geo_df.merge(df_ID, on = 'STAID')
geo_df = geo_df.to_crs('epsg:5070')

# project hl_shp to albers
hl_shp = hl_shp.to_crs('epsg:5070')

# # define bounidng box from states
# xmin, ymin, xmax, ymax = states.total_bounds

# clip hl_shp to contiguous us
hl_shp =  gpd.clip(hl_shp, states) # hl_shp.cx[xmin:xmax, ymin:ymax] #



# %%
# add HLR to id geo_df_train

# commented this out after HLR has been added to df_ID

# geo_df = gpd.sjoin_nearest(
#     geo_df,
#     hl_shp,
#     ).sort_index()




# %%
# plot
################

# main map
fig = plt.figure()
ax_map = fig.add_axes(([0, 0, 1, 1]))
states.boundary.plot(
    ax = ax_map,
    zorder = 0,
    color = 'gray'
    )
geo_df.plot(
    ax = ax_map,
    column = 'HLR',
    zorder = 1
    )

# subplot of histograms
ax_hist = fig.add_axes(
    [0.09, 0.14, 0.2, 0.2]
    )
ax_hist.patch.set_alpha(0)
ax_hist.set_xlabel(
    'Hydrologic Landscape Region',
    fontsize = 9
    )
ax_hist.set_ylabel(
    'Count',
    fontsize = 9
    )
ax_hist.xaxis.set_major_locator(ticker.NullLocator())    
# ax_hist.set_xticklabels('')
# ax_hist.get_xaxis().set_visible(False)
geo_df.HLR.hist(
    bins = 20,
    ax = ax_hist,
    zorder = 2
    )

# xlab = p1.get_l

# p1.xlabel(
#     'test'
# )


# hl_shp['HLR'] = str(hl_shp['HLR'])
# define base axes as state boundaries
# fig, ax = plt.subplots() # figsize = (10, 12)
# hl_shp.plot( # boundary.plot(
#     ax = ax,
#     # figsize = (12, 9), 
#     # color = 'Gray',
#     column = 'HLR',
#     vmin = 0,
#     vmax = 21,
#     # linewidth = 0.01,
#     zorder = 0,
#     legend = True,
#     # legend_kwds = {
#     #     'loc': 'lower center',
#     #     'bbox_to_anchor': (0.5, 0.5)
#     # }
# )

# # plot catchment polygons
# geo_df_train.plot(
#     ax = ax,
#     # column = 'AggEcoregion',
#     column = 'HLR',
#     markersize = 1, 
#     # marker = 'x',
#     legend = True,
#     # legend_kwds = {# 'loc': 'lower center', 
#     #                 'ncol': 4,
#     #                 'bbox_to_anchor': (1.1, 0)},
#     zorder = 1)

# # plot histogram of HLR representation
# geo_df_train.HLR.hist()

# geo_df_valnit.HLR.hist()



# %%
# add HLR column to id dfs and write them out
#####################

# add HLR column

# df_ID = pd.merge(
#     df_ID,
#     pd.DataFrame({
#         'STAID': geo_df['STAID'],
#         'HLR': geo_df['HLR']
#     }),
#     on = 'STAID')


# # write to csv

# df_ID.to_csv(
#     f'{dir_expl}/ID_all_avail98_12.csv',
#     index = False
# )

