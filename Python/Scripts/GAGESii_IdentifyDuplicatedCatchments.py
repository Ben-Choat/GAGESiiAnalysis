'''
BChoat 2022/12/13

Script written to identify catchment polygons that are very similar to
each other.

First, Identify catchments where the area is within 5% of each other.
Then, if the overlap area / union area is > 0.95 then remove one catchment.
'''

# %% 
# import libraries
##################

import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import seaborn as sns



# %% 
# define directories and variables
###############

# dir_spatial = 'D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco'



# %%
# read in shape files and other data
#####################

# read in id variables for labelling output and subsetting shape
# files to study catchments
df_ID = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
    )

list_shp=glob.glob(r'D:/DataWorking/GAGESii/boundaries-shapefiles-by-aggeco/*.shp')

# read in reference catchments to a variable and remove them from the file list
shp_ref = gpd.read_file(
             list_shp[len(list_shp)-1]
            ).to_crs("EPSG:5070")

# subset to study catchments
shp_ref = shp_ref[shp_ref['GAGE_ID'].isin(df_ID['STAID'])]

# remove shape file of AK, HI, and PR
list_shp.remove(list_shp[0])
list_shp.remove(list_shp[len(list_shp)-1])


# read in states shape file
states = gpd.read_file(
    'D:/DataWorking/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'
    )

# print head of states shape file df
states.head()

# print shapes coordinate system
states.crs

# convert to WGSS84 coordinate system
shape = states.to_crs('EPSG:5070') # "EPSG:4326")

# plot states
shape.boundary.plot()        

# exclude Alaska, Hawaii, and Puerto Rico
shape = shape.loc[~shape['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]


# read in annual water yield data
df_wy_annual = pd.read_csv(
    'D:/DataWorking/USGS_discharge/annual_WY/Annual_WY_1976_2013.csv',
    dtype = {'site_no': 'string'}
    )

# filter to years of interest
df_wy_annual = df_wy_annual[
    df_wy_annual['yr'].isin(range(1998, 2013))
    ].reset_index(drop = True)



# read in explanatory variables
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# read in all static vars
df_static_all = pd.read_csv(
    f'{dir_expl}/GAGES_Static_FilteredWWTP.csv',
    dtype = {'STAID': 'string'}
)

# read in time-series vars
df_ts_all = pd.read_csv(
    f'{dir_expl}/gagesii_ts/GAGESts_InterpYrs_Wide.csv',
    dtype = {'STAID': 'string'}
)

# subset to years of interest 1998 - 2012
df_ts_all = df_ts_all[df_ts_all['year'].isin(np.arange(1998, 2013, 1))]

# merge static and ts vars
df_expl_all = pd.merge(
    df_ts_all,
    df_static_all,
    on = 'STAID'
)



# DAYMET

dir_DMT = 'D:/DataWorking/Daymet/train_val_test'

# read in annual daymet data
df_dmt_annual = pd.read_csv(
    'D:/DataWorking/Daymet/Daymet_Annual_GAGES.csv',
    dtype = {'site_no': 'string'}
    )
# %%
# loop through polygons and return number of overlapping polygons for each
###############

# define threshold for proportion overlap to remove
thresh_ovlp = 0

# define lists to append results to
STAID_out1 = []
STAID_out2 = []
Prop_Ovlp = []
AggEco_out = []

# loop through shape files
for shp in list_shp:
    shp_in = gpd.read_file(
                shp 
                ).to_crs('EPSG:5070')
    print(shp)

    # subset to study catchments
    shp_in = shp_in[shp_in['GAGE_ID'].isin(df_ID['STAID'])]

    # join reference catchments with non-reference catchemnts
    shp_in = pd.concat([shp_ref, shp_in], ignore_index = True)

    for i in shp_in.index:

        shp_in2 = shp_in.drop(i, axis = 0)

        print(np.round(i/shp_in.index.max(), 2))

        # calculate area of intercept between polygons
        int_area = shp_in2[
        'geometry'
        ].intersection(
            shp_in.loc[i, 'geometry']
            ).area

        # calculate area of union between polygons
        int_union = shp_in2[
            'geometry'
            ].union(
                shp_in.loc[i, 'geometry']
                ).area

        # calculate proportion of total area that intersects
        prop_ovlp = int_area/int_union

        # identify where proportion overlap is greater than specified proportion
        ind_temp = prop_ovlp.index[np.where(prop_ovlp > thresh_ovlp)]
        prop_ovlp = prop_ovlp.loc[ind_temp]
        ovlp_out = shp_in.loc[ind_temp]


        # update df_out with new results
        STAID_out1.extend(ovlp_out['GAGE_ID'].values)
        STAID_out2.extend(
            [shp_in.loc[i, 'GAGE_ID']] * len(ovlp_out['GAGE_ID'].values)
            )
        Prop_Ovlp.extend(prop_ovlp.values)
        AggEco_out.extend(
            df_ID.loc[
                df_ID['STAID'].isin(ovlp_out['GAGE_ID'].values), 'AggEcoregion'
                ]
            )


df_out = pd.DataFrame({
    'STAID1': STAID_out1,
    'STAID2': STAID_out2,
    'Prop_Ovlp': Prop_Ovlp,
    'AggEcoregion': AggEco_out
    })

# the next three lines of code were produced by openAI's chatGPT.
# create a new column that concatenates the values of the two columns
# in a way that the order of the values doesn't matter
df_out['combined'] = df_out[['STAID1', 'STAID2']].apply(
    lambda x: '_'.join(sorted(x)), axis=1
    )

# drop duplicates based on the new column
df_out = df_out.drop_duplicates(subset=['combined'])

# drop the combined column
df_out = df_out.drop(columns=['combined'])

# keep only catchments with overlap > 0
# df_out = df_out[df_out['Prop_Ovlp'] > 0]

# df_out['Count1'] = [STAID_out1.count(i) for i in STAID_out1]
# df_out['Count2'] = [STAID_out2.count(i) for i in STAID_out2]

# write results to csv
df_out.to_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/OverlappingCatchments.csv',
    index = False
    )


# %%
# some more plotting
################

# make points from lat long in id files
long = df_ID['LNG_GAGE']
lat = df_ID['LAT_GAGE']
points = gpd.points_from_xy(long, lat, crs = states.crs)
geo_df = gpd.GeoDataFrame(geometry = points)
geo_df['STAID'] = df_ID['STAID']
geo_df = geo_df.merge(df_ID, on = 'STAID')
geo_df = geo_df.to_crs('EPSG:5070')

# plot catchments of interest
# input_catchs = ['03341500', '03342000', '03340500']
input_catchs = ['07241550', '07238000', '07239300',
                '07239450', '07239500', '07241000',
                '07241520', '07242000', '07237500']

# input_catchs = list(df_out.loc[df_out['STAID1'] == '07241520', 'STAID2'])
# input_catchs.extend(['07241520'])
# input_catchs = ['10109000', '10109001']
# input_catchs = ['01646502', '01646500']
# input_catchs = ['06862700', '06862850']
# input_catchs = ['08384500', '08383500']
# input_catchs = ['11062001', '11062000']


geo_work = geo_df[geo_df['STAID'].isin(input_catchs)]

#########
# investigate water yield
df_wy = df_wy_annual[df_wy_annual['site_no'].isin(input_catchs)]

# fine mean water yield for each input catchment
wy_mn = df_wy.groupby('site_no').mean('Ann_WY_acft')

geo_wy = pd.merge(geo_work, wy_mn, left_on = 'STAID', right_index = True)
#########

# identify which shape file to read in based on aggecoregion the
# catchments are located in
agg_work = df_ID.loc[
    df_ID['STAID'].isin(input_catchs), 'AggEcoregion'
    ].iloc[0]

for i in list_shp:
    if agg_work in i:
        shp = i

shp_in = gpd.read_file(shp).to_crs('EPSG:5070')

# subset to study catchments
shp_in = shp_in[shp_in['GAGE_ID'].isin(df_ID['STAID'])]

fig, ax = plt.subplots(figsize = (20, 10))
# shape.boundary.plot(ax = ax)
# polygons
shp_in.loc[
    shp_in['GAGE_ID'].isin(input_catchs), 'geometry'
    ].boundary.plot(
        ax = ax, 
        color = sns.color_palette("Set2", len(input_catchs))) # 'blue') # 'red')
# catchment outlets
geo_work.plot(ax = ax, color = 'red')
# add labels to catchment outlets
for x, y, label in zip(
    geo_work.geometry.x, 
    geo_work.geometry.y,
    # geo_work.STAID
    round(geo_wy.Ann_WY_acft).astype(str)
    ):
    ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")


plt.title(f'{input_catchs}')


# %%
# Investigate explanatory variables
#####################

df_expl = df_expl_all[
    df_expl_all['STAID'].isin(input_catchs)
    ]

# calculate mean values for easier comparisons
df_expl_mn = df_expl.groupby('STAID', as_index = False).mean()

df_expl_mn[['STAID', 'DRAIN_SQKM']]

# DAYMET
df_dmt = df_dmt_annual[
    df_dmt_annual['site_no'].isin(input_catchs)
    ]

# calculate mean values for easier comparisons
df_dmt_mn = df_dmt.groupby('site_no', as_index = False).mean()


# %%
# read results to csv
###########

df_out = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/OverlappingCatchments.csv',
    dtype = {'STAID1': 'string',
            'STAID2': 'string'}
    )


# %%
# remove catchments with greater than 99% (or specfied) overlap
# Want to investigate nested catchments to understand spatial scale
# but also want to eliminate 'replicated' catchments that model could
# not differentiate between
######################################


# overlap threshold; overlaps > ov_thresh are eliminated
ov_thresh = 0.99

# get unique STAIDs with overlap >= 0.99

# combine STAID1 and STAID2 into single list
# staid_ov = pd.concat([
#     df_out.loc[df_out['Prop_Ovlp'] > ov_thresh, 'STAID1'],
#     df_out.loc[df_out['Prop_Ovlp'] > ov_thresh, 'STAID2']    
#     ]).unique()

# staid_ov

# identify stations in STAID1 to remove
staid_rm = df_out.loc[
    df_out['STAID1'].isin(
        df_out.loc[df_out['Prop_Ovlp'] >= ov_thresh, 'STAID1']
        ), 'STAID1'].unique()

df_out = df_out[
    ~df_out['STAID1'].isin(staid_rm)
    ].reset_index(drop = True)


# write stations to be removed to a text file
pd.Series(staid_rm).to_csv(
    f'{dir_expl}/STAID_ToRemove.csv',
    index = False
)



# %%
