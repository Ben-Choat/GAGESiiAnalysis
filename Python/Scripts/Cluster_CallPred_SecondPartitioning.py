# Ben Choat 7/28/2022

# Script:
# perform clustering. For the top two performing clustering results,
# call GAGESii_MeanAnnual_Callable.py which goes through all regression models of interset 
# and asks for input where needed.


# %% import libraries and classes

from GAGESii_Class import Clusterer
from GAGESii_MeanAnnual_Callable import *
# from GAGESii_Class import Regressor
# from Regression_PerformanceMetrics_Functs import *
import pandas as pd
import os
# import plotnine as p9
# import plotly.express as px # easier interactive plots
# from scipy import stats
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error

# %% load data

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned'

# directory to write csv holding removed columns (due to high VIF)
dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'

# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/Expl_train.csv',
    dtype = {'STAID': 'string'}
)
# test_in
df_testin_expl = pd.read_csv(
    f'{dir_expl}/Expl_testin.csv',
    dtype = {'STAID': 'string'}
)
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/Expl_valnit.csv',
    dtype = {'STAID': 'string'}
)


# Water yield variables
# Annual Water yield
# training
df_train_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    dtype = {"site_no":"string"}
    )

# val_in
df_testin_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_testin.csv',
    dtype = {"site_no":"string"}
    )

# val_nit
df_valnit_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    dtype = {"site_no":"string"}
    )

# mean annual water yield
# training
df_train_mnanWY = df_train_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])
# val_in
df_testin_mnanWY = df_testin_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])
# val_nit
df_valnit_mnanWY = df_valnit_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])

# mean GAGESii explanatory vars
# training
df_train_mnexpl = df_train_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])
# val_in
df_testin_mnexpl = df_testin_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])
#val_nit
df_valnit_mnexpl = df_valnit_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])

# ID vars (e.g., ecoregion)

# training ID
df_train_ID = pd.read_csv(f'{dir_expl}/ID_train.csv',
    dtype = {'STAID': 'string'})
# val_in ID
df_testin_ID = df_train_ID
# val_nit ID
df_valnit_ID = pd.read_csv(f'{dir_expl}/ID_valnit.csv',
    dtype = {'STAID': 'string'})

del(df_train_anWY, df_train_expl, df_testin_anWY, df_testin_expl, df_valnit_anWY, df_valnit_expl)




# %% 
# Define input variables for modeling

# Here are the function inputs
# def regress_fun(df_train_expl, # training data explanatory variables. Expects STAID to be a column
#                 df_testin_expl, # validation data explanatory variables using same catchments that were trained on
#                 df_valnit_expl, # validation data explanatory variables using different catchments than were trained on
#                 train_resp, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_mnanWY['Ann_WY_ft'])
#                 testin_resp, # validation data response variables using same catchments that were trained on
#                 valnit_resp, # validation data response variables using different catchments than were trained on
#                 train_ID, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
#                 testin_ID, # validation data id's from catchments used in training (e.g., clusters or ecoregions)
#                 valnit_ID, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
#                 clust_meth, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
#                 reg_in, # region label, i.e., 'NorthEast'
#                 grid_in) # dict with XGBoost parameters

########
# subset data to catchment IDs that match the cluster or region being predicted
########

# define clustering method used
# this variable is only used for keeping track fo results
clust_meth_in = 'AggEcoregion'

# list of possible AggEcoregions:
# 'All
# 'NorthEast', 'SECstPlain', 'SEPlains', 'EastHghlnds', 'CntlPlains',
#       'MxWdShld', 'WestMnts', 'WestPlains', 'WestXeric'
# set region_in = 'All' to include all data
region_in = 'NorthEast'

if region_in == 'All':
    cidtrain_in = df_train_ID
    cidtestin_in = df_testin_ID
    cidvalnit_in = df_valnit_ID
else:
    cidtrain_in = df_train_ID[df_train_ID['AggEcoregion'] == region_in]
    cidtestin_in = df_testin_ID[df_testin_ID['AggEcoregion'] == region_in]
    cidvalnit_in = df_valnit_ID[df_valnit_ID['AggEcoregion'] == region_in]

# Water yield
train_resp_in = pd.merge(
    df_train_mnanWY, cidtrain_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
testin_resp_in = pd.merge(
    df_testin_mnanWY, cidtestin_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
valnit_resp_in = pd.merge(
    df_valnit_mnanWY, cidvalnit_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
# explanatory variables
train_expl_in = pd.merge(df_train_mnexpl, cidtrain_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
testin_expl_in = pd.merge(df_testin_mnexpl, cidtestin_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
valnit_expl_in = pd.merge(df_valnit_mnexpl, cidvalnit_in, on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
# ID dataframes
train_ID_in = pd.merge(
    df_train_ID, cidtrain_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
testin_ID_in = pd.merge(
    df_testin_ID, cidtestin_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
valnit_ID_in = pd.merge(
    df_valnit_ID, cidvalnit_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
##########

# %%
#####
# Remove variables with a VIF > defined threshold (e.g., 10)
#####
 
X_in = train_expl_in.drop(
    ['STAID'], axis = 1
)

vif_th = 10 # 20

# calculate all vifs and store in dataframe
df_vif = VIF(X_in)

# initiate array to hold varibles that have been removed
df_removed = []

while any(df_vif > vif_th):
    # find max vifs and remove. If > 1 max vif, then remove only 
    # the first one
    maxvif = np.where(df_vif == df_vif.max())[0][0]

    # append inices of max vifs to removed dataframe
    df_removed.append(df_vif.index[maxvif])

    # drop max vif feature
    # df_vif.drop(df_vif.index[maxvif], inplace = True)
    
    # calculate new vifs
    df_vif = VIF(X_in.drop(df_removed, axis = 1))

# redefine mean explanatory var df by dropping 'df_removed' vars and year column
# drop columns from mean and timeseries explanatory vars
# training data
train_expl_in.drop(
    df_removed, axis = 1, inplace = True
)
# testin data
testin_expl_in.drop(
    df_removed, axis = 1, inplace = True
)
# valnit data
valnit_expl_in.drop(
    df_removed, axis = 1, inplace = True
)

# print columns removed
print(df_removed)

# write csv with removed columns
import os
if not os.path.exists(dir_VIF):
    os.mkdir(dir_VIF)

df_vif_write = pd.DataFrame({
    'Columns_Removed': df_removed
})

df_vif_write.to_csv(f'{dir_VIF}/VIF_ClmnsRemoved_{clust_meth_in}_{region_in}.csv')



# %% ###################
# UMAP followed by HDBSCAN
########################

# Standardize data

# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']

# define clusterer object
cl_obj = Clusterer(clust_vars = df_train_mnexpl.drop(columns = ['STAID']),
    id_vars = df_train_mnexpl['STAID'])

# note that once input data is transformed, the transformed
# version will be used automatically in all functions related to
# Clusterer object
cl_obj.stand_norm(method = 'standardize', # 'normalize'
    not_tr = not_tr_in)

# see UMAP_HDBSCAN.py for code

# %% 
# Call function to perform modeling

regress_fun(df_train_expl = train_expl_in, # training data explanatory variables. Expects STAID to be a column
            df_testin_expl = testin_expl_in, # validation data explanatory variables using same catchments that were trained on
            df_valnit_expl = valnit_expl_in, # validation data explanatory variables using different catchments than were trained on
            train_resp = train_resp_in, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_mnanWY['Ann_WY_ft'])
            testin_resp = testin_resp_in, # validation data response variables using same catchments that were trained on
            valnit_resp = valnit_resp_in, # validation data response variables using different catchments than were trained on
            train_ID = train_ID_in, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
            testin_ID = testin_ID_in, # validation data id's from catchments used in training (e.g., clusters or ecoregions)
            valnit_ID = valnit_ID_in, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
            clust_meth = clust_meth_in, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
            reg_in = region_in, # region label, i.e., 'NorthEast'
            grid_in = { # dict with XGBoost parameters
                'n_estimators': 500, #  [100, 500], # [100, 250, 500], # [10], # 
                'colsample_bytree': 1, # [1], # [0.7, 1], 
                'max_depth': 6, # [6], # [4, 6, 8],
                'gamma': 0, # [0], # [0, 1], 
                'reg_lambda': 0, # [0], # [0, 1, 2]
                'learning_rate': 0.3, # [0.3] # [0.02, 0.1, 0.3]
                }
            )


# %%
