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
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_VIF10_Filtered'

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

# %% read or create a results dataframe to append new results to
# explantory var (and other data) directory

 # DELETE THIS CODE UNLESS YOU FIND IT IS NEED (2022/08/17)
# dir_res = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results'

# try:
#     df_results = pd.read_csv(f'{dir_expl}/Results_AllRegionsNonTimeSeries.csv')
# except:
#     df_results = pd.DataFrame({
#         'model': [], # specify which model, e.g., 'raw_lasso'
#         'train_val': [], # specify if results from training, validation (i.e., testin or testint)
#         'parameters': [], # any hyperparameters (e.g., alpha penalty in lasso regression)
#         'n_features': [], # number of explanatory variables (i.e., featuers)
#         'ssr': [], # sum of squared residuals
#         'r2': [], # unadjusted R2
#         'r2adj': [], # adjusted R2
#         'mae': [], # mean absolute error
#         'rmse': [], # root mean square error
#         'VIF': [], # variance inflation factor (vector of all included featuers and related VIFs)
#         'percBias': [] # percent bias
#     })











# %% k-means clustering

# not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
#             'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
#             'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']

# test = Clusterer(clust_vars = df_train_mnexpl.drop(columns = ['STAID']),
#     id_vars = df_train_mnexpl['STAID'])

# # note that once input data is transformed, the transformed
# # version will be used automatically in all functions related to
# # Clusterer object
# test.stand_norm(method = 'standardize', # 'normalize'
#     not_tr = not_tr_in)

# test.k_clust(
#     ki = 2, kf = 20, 
#     method = 'kmeans', 
#     plot_mean_sil = True, 
#     plot_distortion = True)

# #####
# # Based on results from previous chunk, chose k = 10
# # Here defining the test object to have a k-means model of k = 10 for projecting new data
# test.k_clust(
#     ki = 10, kf = 10, 
#     method = 'kmeans', 
#     plot_mean_sil = False, 
#     plot_distortion = False)

# #######
# # data to project into k-means (or k-medoids) space
# df_valnit_trnsfrmd = pd.DataFrame(test.scaler_.transform(df_valnit_mnexpl.drop(
#     columns = ['STAID'])
#     ))
# # give column names to transformed dataframe
# df_valnit_trnsfrmd.columns = df_valnit_mnexpl.drop(
#     columns = ['STAID']
# ).columns

# # replace ohe columns with untransformed data
# df_valnit_trnsfrmd[not_tr_in] = df_valnit_mnexpl[not_tr_in]
# #########
# # get predicted k's for each catchment           
# km_valnit_pred = pd.DataFrame(
#     {'STAID': df_valnit_mnexpl['STAID'],
#     'K': test.km_clusterer_.predict(
#             df_valnit_trnsfrmd
#             )
#     }
# )
# ##########
# # K-medoids

# test.k_clust(
#     ki = 2, kf = 30, 
#     method = 'kmedoids', 
#     plot_mean_sil = True, 
#     plot_distortion = True,
#     kmed_method = 'alternate')

# for i in range(19, 23):
#     test.plot_silhouette_vals(k = i)
# #####
# # Based on results from previous chunk, chose k = 8
# # Here defining the test object to have a k-means model of k = 8 for projecting new data
# test.k_clust(
#     ki = 8, kf = 8, 
#     method = 'kmedoids', 
#     plot_mean_sil = False, 
#     plot_distortion = False)

# # get predicted k's for each catchment           
# km_valnit_pred = test.km_clusterer_.predict(
#     df_valnit_trnsfrmd
#     )
# # predicted clusters dataframe
# df_km_valnit_pred = pd.DataFrame({
#     'STAID': df_valnit_mnexpl['STAID'],
#     'K_predicted': km_valnit_pred
# })




# %% AggEcoregions as clusters

# for i in df_train_ID['AggEcoregion'].unique():
#     print(i)









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
#                 reg_in) # region label, i.e., 'NorthEast'

########
# subset data to catchment IDs that match the cluster or region being predicted
########


# list of possible AggEcoregions:
# 'NorthEast', 'SECstPlain', 'SEPlains', 'EastHghlnds', 'CntlPlains',
#       'MxWdShld', 'WestMnts', 'WestPlains', 'WestXeric'
# set region_in = 'All' to include all data
region_in = 'All'

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
            clust_meth = 'AggEcoregions', # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
            reg_in = region_in # region label, i.e., 'NorthEast'
            )       


# %%
