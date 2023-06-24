'''
BChoat 2022/12/08

This script is written to explore the clusters resulting from
UMAP-HDBSCAN.
'''


# %%
# import libraries
###############

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from Load_Data import load_data_fun


# %%
# load data
##################

# load data (explanatory, water yield, ID)
# training
df_trainexpl, df_trainWY, df_trainID = load_data_fun(
    dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned', 
    time_scale = 'mean_annual',
    train_val = 'train',
    clust_meth = 'AggEcoregion',
    region = 'NorthEast',
    standardize = False # whether or not to standardize data
    )


df_IDtrain = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_train.csv',
    dtype = {'STAID': 'string'}
    ).drop('DRAIN_SQKM', axis = 1)

# df_IDvalnit = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_valnit.csv',
#     dtype = {'STAID': 'string'}
#     )

# # expl vars
# df_expltrain = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/Expl_train.csv',
#     dtype = {'STAID': 'string'}
# )

# %%
# 
####################