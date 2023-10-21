'''
BChoat 2022/11/11
Script to investigate trees from XGBoost models
'''


# %% 
# Import Libraries
###########

import pandas as pd
import numpy as np
from Load_Data import load_data_fun
import glob
import xgboost as xgb
import matplotlib.pyplot as plt




# from xgboost import plot_tree # can't get to work on 
# windows (try on linux)




# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth =  'AggEcoregion' # 'Class' # 'None' # 'AggEcoregion', 'None', 

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
region =  'SECstPlain' # 'WestMnts' # 'All' # 'CntlPlains' # 'Non-ref' # 'All'
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly', 'daily'

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/' 
# dir_work = '/media/bchoat/Local Disk/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'



# %%
# XGBOOST
##########
# first define xgbreg object
model = xgb.XGBRegressor()

# define temp time_scale var since model names do not have '_' in them
temp_time = time_scale.replace('_', '')

# reload model into object
model.load_model(
    f'{dir_work}/data_out/{time_scale}'
    f'/Models/XGBoost_{temp_time}_{clust_meth}_{region}_model.json'
    )

# %% 
# Write tree to dataframe and Return branches with specified variable
###################

# define variable/feature of interest
# voi = 'DRAIN_SQKM'
# voi = 'WTDEPAVE'
# voi = 'prcp'
# voi = 'prcp_1'
voi = 'TS_NLCD_21'
# voi = 'TS_NLCD_24'
# voi = 'TS_NLCD_71'
# voi = 'TS_NLCD_31'
# voi = 'TS_NLCD_52'
# voi = 'TS_NLCD_11'
# voi = 'TS_ag_irrig'
# voi = 'TS_WaterUse_wu'
# voi = 'STOR_NID_2009'
# voi = 'swe'
# voi = 'SLOPE_PCT'
# voi = 'BDAVE'
# voi = 'TS_Housing_HDEN'
# voi = 'CANALS_PCT'
# voi = 'TOPWET'


# print trees to dataframe
df_tree = model.get_booster().trees_to_dataframe()

# remove '_x' from DRAIN_SQKM feature label
df_tree['Feature'] = df_tree['Feature'].str.replace('_x', '')

# subset to shallower branches
# df_tree = df_tree[
#     (df_tree['ID'].str.split('-').str[1] == '0') | \
#     (df_tree['ID'].str.split('-').str[1] == '1')   
#     # df_tree['ID'].str.contains('|'.join(['-0', '-1'])).tolist()
# ]

df_treeGain = df_tree.groupby(['Feature', 'Split'])['Gain'].sum().reset_index()
df_treeGain['Feat_Split'] = \
    df_treeGain['Feature'] + '-' + df_treeGain['Split'].round(2).astype(str)
df_treeGain.sort_values(by = 'Gain', ascending = False).head(20)
# drop 'leaaf' feature
df_tree = df_tree[df_tree['Feature'] != 'Leaf']


################################
# gains of each split of each var
# Calculate the count of values in the 'Category' column
# value_counts = df_tree['Feature'].value_counts()[20:40]

df_plot = df_treeGain.sort_values(
    by = 'Gain', ascending = False
    ).reset_index(drop = True)[120:130]
df_plot.loc[0:125, 'Feature'].unique()
# Create a bar plot
plt.bar(df_plot['Feat_Split'], df_plot['Gain'])

# Adding labels and title
plt.xlabel('Category')
plt.ylabel('Gain')
plt.title('Total gain at split value')
# Rotate x-axis tick labels and anchor them to the end
plt.xticks(rotation=45, horizontalalignment='right')
# Show the plot
plt.show()

#####################

################################
# gain of each split
# Calculate the count of values in the 'Category' column
# value_counts = df_tree['Feature'].value_counts()[20:40]

# Create a bar plot
plt.bar(
    df_tree.loc[df_tree['Feature'] == voi, 'Split'], 
    df_tree.loc[df_tree['Feature'] == voi, 'Gain'],
    width = 1)

# Adding labels and title
plt.xlabel('Split')
plt.ylabel('Gain')
plt.title(f'Gains at Specific Split Values - {voi} - {region}')
# Rotate x-axis tick labels and anchor them to the end
plt.xticks(rotation=45, horizontalalignment='right')
# Show the plot
plt.show()

#####################


################################
# count of each var
# Calculate the count of values in the 'Category' column
value_counts = df_tree['Feature'].value_counts()[20:40]

# Create a bar plot
plt.bar(value_counts.index, value_counts.values)

# Adding labels and title
plt.xlabel('Category')
plt.ylabel('Count')
plt.title('Count of Splits Using Feature')
# Rotate x-axis tick labels and anchor them to the end
plt.xticks(rotation=45, horizontalalignment='right')
# Show the plot
plt.show()

#####################

print(df_tree.sort_values(
    by = 'Gain', ascending = False
    ).reset_index(drop = True)['Feature'].unique())


# subset trees to nodes where voi appears
df_tree_voi = df_tree[df_tree['Feature'] == voi]
# df_tree_voi
# df_tree_voi.sort_values(by = 'Gain', ascending = False)
# df_tree_voi.sort_values(by = 'Tree')

# print number of times each unique value of the voi was uesd for splitting
# df_tree_voi.loc[df_tree_voi['Feature'] == 'DRAIN_SQKM', 'Split'].value_counts().head(20)

# sum total gain by each value of area used for splitting nodes
df_splitGain = df_tree_voi.groupby('Split').sum('Gain').sort_values(
    by = 'Gain', ascending = False
    ).head(20)[['Gain']]
df_splitGain = df_splitGain.reset_index()
fig, ax = plt.subplots(figsize = [8,8])
ax.plot(df_splitGain['Split'], df_splitGain['Gain'])
ax.scatter(df_splitGain['Split'], df_splitGain['Gain'], color = 'orange')
ax.set(xlabel = 'Split Values', ylabel = 'Gain at split', title = voi)
plt.show()

import math
# print the number of times each split occurs.
nbins = math.ceil(math.log(df_tree_voi.shape[0], 2) + 1) * 2
# df_tree_voi.value_counts('Split').hist(bins = nbins)
ax = df_tree_voi.Split.hist(bins = nbins)
ax.set_title(f'Counts of splits in {voi}')
plt.show()
df_tree_voi.value_counts('Split')


# %% find peaks and plot
#############################

# define data
# Compute the histogram

from scipy.signal import find_peaks

df_plot = df_tree[
    df_tree['Feature'] == voi
    ].sort_values(by = 'Split').reset_index(drop = True)

df_plot = df_plot.groupby(['Split'])['Gain'].sum().reset_index()

x = df_plot['Split']
y = df_plot['Gain']

# calc distance threshold and prominence_threshold
data_range = np.max(y) - np.min(y)
distance_fraction = 0.0005  # 10% of data range
prominence_fraction = 0.3  # 20% of data range
distance_threshold = data_range * distance_fraction
if distance_threshold < 20: distance_threshold = 20
prominence_threshold = data_range * prominence_fraction
print(f'\n\ndistance_threshold: {distance_threshold}')
print(f'\nprominence_threshold: {prominence_threshold}')

peaks, _ = find_peaks(
    df_plot['Gain'], 
    distance = distance_threshold, prominence = prominence_threshold
    )

# plot
plt.figure(figsize=(8, 4))

# Plot the continuous data
plt.plot(
    x, y,
    label='Continuous Data', color='b')
plt.scatter(x.iloc[peaks], 
            y.iloc[peaks], 
            c='r', marker='o', label='Peaks')

plt.title(f"Identifying Peaks in Split Values of {voi} Based on Gain")
plt.xlabel("Split Value")
plt.ylabel("Gain")
plt.legend()
plt.grid(True)
plt.show()


# %%
# plot tree
##############

fig, ax = plt.subplots(figsize = (100, 100))
# xgb.plot_tree(model, num_trees = 89, ax = ax)
xgb.plot_tree(model, num_trees = 749, ax = ax)
# %%
