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
clust_meth = 'None' # 'Class' # 'None' # 'AggEcoregion', 'None', 

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
region =  'All' # 'CntlPlains' # 'Non-ref' # 'All'
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
time_scale = 'mean_annual' # 'mean_annual', 'annual', 'monthly', 'daily'

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
voi = 'DRAIN_SQKM'

# print trees to dataframe
df_tree = model.get_booster().trees_to_dataframe()

# remove '_x' from DRAIN_SQKM feature label
df_tree['Feature'] = df_tree['Feature'].str.replace('_x', '')

# subset trees to nodes where voi appears
df_tree_voi = df_tree[df_tree['Feature'] == voi]
df_tree_voi
df_tree_voi.sort_values(by = 'Gain', ascending = False)
df_tree_voi.sort_values(by = 'Tree')

# print number of times each unique value of the voi was uesd for splitting
df_tree_voi.loc[df_tree_voi['Feature'] == 'DRAIN_SQKM', 'Split'].value_counts().head(20)

# sum total gain by each value of area used for splitting nodes
df_tree_voi.groupby('Split').sum('Gain').sort_values(by = 'Gain', ascending = False).head(20)[['Gain']]
df_tree_voi.value_counts('Split')

# %%
# plot tree
##############

fig, ax = plt.subplots(figsize = (100, 100))
# xgb.plot_tree(model, num_trees = 89, ax = ax)
xgb.plot_tree(model, num_trees = 7, ax = ax)
# %%
