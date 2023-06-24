'''
BChoat 11/14/2022

Script to associate WWTP with catchments.
Note that all HydroSHEDS data (including WWTP data) comes
projected to WGS 1984.

All lines that write out files have been commented out.
'''

# %%
# Import libraries
#################

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import glob # for unix like interface to directories

# %%
# define variables, directories, and such
##############

# directory with data to work with
dir_work = 'D:/DataWorking/HydroWASTE_v10'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# %%
# read in data
#############

df_wwtp = pd.read_csv(
    f'{dir_work}/HydroWASTE_v10.csv',
    encoding = 'latin-1'
)

df_wwtp = df_wwtp[df_wwtp['COUNTRY'] == 'United States']

# spatial file of us
states = gpd.read_file(
    'D:/DataWorking/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
)
# filter out Alaska, Hawaii, and Puerto Rico
states = states[~states['NAME'].isin(['Alaska', 'Puerto Rico', 'Hawaii'])]

# convert to Albers projection
states = states.to_crs({'init': 'epsg:5070'})

df_ID = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
).drop(['CLASS', 'AGGECOREGION'], axis = 1)

# read in all static vars
df_static_all = pd.read_csv(
    f'{dir_expl}/GAGES_Static_Filtered.csv',
    dtype = {'STAID': 'string'}
)

# filter expl. vars to only those catchments being used in this study
df_static_all = df_static_all[
    df_static_all['STAID'].isin(df_ID['STAID'])
    ]


# combine all catchment polygons from all 'regions'

name_list = ['nonref_CntlPlains', 'nonref_EastHghlnds', 'nonref_MxWdShld', 
                'nonref_NorthEast', 'nonref_SECstPlain', 'nonref_SEPlains', 
                'nonref_WestMnts', 'nonref_WestPlains', 'nonref_WestXeric', 'ref_all']

# read and merge shape files of catchments
geo_gages = gpd.pd.concat([
    gpd.read_file(
        f'D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco/bas_{x}.shp'
        ) for
            x in name_list
    ])

# define GAGE_ID to be the same as STAID 
geo_gages = geo_gages[
    geo_gages['GAGE_ID'].isin(df_ID['STAID'].to_numpy())
]

# ensure gages polygons are in same projection as wwtp
geo_gages.to_crs(states.crs)

# sort by area so small catchments plot on top of bigger catchments
geo_gages = geo_gages.sort_values(
    by = 'AREA',
    ascending = False
    )

geo_gages.plot(column = 'AREA')



# %%
# convert wwtp to spatial file
#############

temp_long = df_wwtp['LON_OUT']
temp_lat = df_wwtp['LAT_OUT']
temp_points = gpd.points_from_xy(temp_long, temp_lat, crs = 'epsg:4326') 
geo_wwtp = gpd.GeoDataFrame(geometry = temp_points)
geo_wwtp['WASTE_ID'] = df_wwtp['WASTE_ID'].to_numpy()
geo_wwtp = geo_wwtp.merge(df_wwtp, on = 'WASTE_ID')
geo_wwtp = geo_wwtp.to_crs(states.crs)

# clip wwtps to study catchment boundaries
# geo_wwtp = gpd.clip(geo_wwtp, geo_gages)
geo_wwtp = gpd.sjoin(geo_wwtp, geo_gages)

# # write wwtp dataframe to a csv
# geo_wwtp.to_csv(
#     f'{dir_work}/WWTP_GAGESii.csv',
#     index = False
# )

# # write wwtp dataframe to a feather file
# geo_wwtp.to_feather(
#     f'{dir_work}/WWTP_GAGESii.feather',
#     index = False
# )


# %%
# plot
################

# define base axes as state boundaries
fig, ax = plt.subplots() # figsize = (10, 12)
states.boundary.plot(
    ax = ax,
    figsize = (12, 9), 
    color = 'Gray',
    linewidth = 1,
    zorder = 0
)

# # plot catchment polygons
# geo_gages.plot(
#     ax = ax,
#     column = 'AREA', # 'RIVER_DIS', 
#     # markersize = 1, 
#     # marker = 'x',
#     legend = True,
#     # legend_kwds = {'loc': 'lower center', 
#     #                     'ncol': 4},
#     zorder = 1)

# plot wwtp points
geo_wwtp.plot(
    ax = ax,
    column = 'WASTE_DIS', # 'RIVER_DIS', 
    markersize = 1, 
    alpha = 0.5,
    # marker = 'x',
    # legend = True,
    # legend_kwds = {'loc': 'lower center', 
    #                     'ncol': 4},
    zorder = 2)



# %%
# sum wwtp discharge in each catchment and 
# combine wwtp data with gagesii vars

# sum by catchment
temp_df = pd.DataFrame({
    'WWTP_Effluent': geo_wwtp.groupby('GAGE_ID').sum('WASTE_DIS')['WASTE_DIS']
})

# add effluent to expl vars
df_expl = pd.merge(
    df_static_all,
    temp_df,
    left_on = 'STAID',
    right_on = 'GAGE_ID',
    how = 'left'
)

# replace nans in effluent with 0
df_expl['WWTP_Effluent'].fillna(0, inplace = True)

# plot distribution of effluent in training and valnit data
df_expl['WWTP_Effluent'].hist(bins = 100)



# %%
# write new static expl vars back to csv
##############

df_expl.to_csv(
    f'{dir_expl}/GAGES_Static_FilteredWWTP.csv',
    index = False
)


