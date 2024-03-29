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
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_Train_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])
# val_in
df_valin_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_ValIn_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE', 'GEOL_REEDBUSH_DOM_anorthositic'])
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_ValNit_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])


# Explanatory variables
# Annual Water yield
# training
df_train_anWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_train.csv',
    dtype = {"site_no":"string"}
    )
# drop stations not in explantory vars
df_train_anWY = df_train_anWY[
    df_train_anWY['site_no'].isin(df_train_expl['STAID'])
    ].reset_index(drop = True)
# create annual water yield in ft
df_train_anWY['Ann_WY_ft'] = df_train_anWY['Ann_WY_ft3']/(
    df_train_expl['DRAIN_SQKM']*(3280.84**2)
    )

# val_in
df_valin_anWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_val_in.csv',
    dtype = {"site_no":"string"}
    )
# drop stations not in explantory vars    
df_valin_anWY = df_valin_anWY[
    df_valin_anWY['site_no'].isin(df_valin_expl['STAID'])
    ].reset_index(drop = True)
# create annual water yield in ft
df_valin_anWY['Ann_WY_ft'] = df_valin_anWY['Ann_WY_ft3']/(
    df_valin_expl['DRAIN_SQKM']*(3280.84**2)
    )

# val_nit
df_valnit_anWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_val_nit.csv',
    dtype = {"site_no":"string"}
    )
# drop stations not in explantory vars
df_valnit_anWY = df_valnit_anWY[
    df_valnit_anWY['site_no'].isin(df_valnit_expl['STAID'])
    ].reset_index(drop = True)
# subset valint expl and response vars to common years of interest
df_valnit_expl = pd.merge(
    df_valnit_expl, 
    df_valnit_anWY, 
    how = 'inner', 
    left_on = ['STAID', 'year'], 
    right_on = ['site_no', 'yr']).drop(
    labels = df_valnit_anWY.columns, axis = 1
)
df_valnit_anWY = pd.merge(df_valnit_expl, 
    df_valnit_anWY, 
    how = 'inner', 
    left_on = ['STAID', 'year'], 
    right_on = ['site_no', 'yr']).drop(
    labels = df_valnit_expl.columns, axis = 1
)
df_valnit_anWY['Ann_WY_ft'] = df_valnit_anWY['Ann_WY_ft3']/(
    df_valnit_expl['DRAIN_SQKM']*(3280.84**2)
    )

# mean annual water yield
# training
df_train_mnanWY = df_train_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])
# val_in
df_valin_mnanWY = df_valin_anWY.groupby(
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
df_valin_mnexpl = df_valin_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])
#val_nit
df_valnit_mnexpl = df_valnit_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])

# ID vars (e.g., ecoregion)
# vars to color plots with (e.g., ecoregion)
df_ID = pd.read_csv(
    f'{dir_expl}/GAGES_idVars.csv',
    dtype = {'STAID': 'string'}
)

# training ID
df_train_ID = df_ID[df_ID.STAID.isin(df_train_expl.STAID)].reset_index(drop = True)
# val_in ID
df_valin_ID = df_train_ID
# val_nit ID
df_valnit_ID = df_ID[df_ID.STAID.isin(df_valnit_expl.STAID)].reset_index(drop = True)

del(df_train_anWY, df_train_expl, df_valin_anWY, df_valin_expl, df_valnit_anWY, df_valnit_expl)

# %% read or create a results dataframe to append new results to
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

try:
    df_results = pd.read_csv(f'{dir_expl}/Results/Results_NonTimeSeries.csv')
except:
    df_results = pd.DataFrame({
        'model': [], # specify which model, e.g., 'raw_lasso'
        'train_val': [], # specify if results from training, validation (i.e., valin or valint)
        'parameters': [], # any hyperparameters (e.g., alpha penalty in lasso regression)
        'n_features': [], # number of explanatory variables (i.e., featuers)
        'ssr': [], # sum of squared residuals
        'r2': [], # unadjusted R2
        'r2adj': [], # adjusted R2
        'mae': [], # mean absolute error
        'rmse': [], # root mean square error
        'VIF': [], # variance inflation factor (vector of all included featuers and related VIFs)
        'percBias': [] # percent bias
    })











# %% k-means clustering

not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']

test = Clusterer(clust_vars = df_train_mnexpl.drop(columns = ['STAID']),
    id_vars = df_train_mnexpl['STAID'])

# note that once input data is transformed, the transformed
# version will be used automatically in all functions related to
# Clusterer object
test.stand_norm(method = 'standardize', # 'normalize'
    not_tr = not_tr_in)

test.k_clust(
    ki = 2, kf = 20, 
    method = 'kmeans', 
    plot_mean_sil = True, 
    plot_distortion = True)

#####
# Based on results from previous chunk, chose k = 10
# Here defining the test object to have a k-means model of k = 10 for projecting new data
test.k_clust(
    ki = 10, kf = 10, 
    method = 'kmeans', 
    plot_mean_sil = False, 
    plot_distortion = False)

#######
# data to project into k-means (or k-medoids) space
df_valnit_trnsfrmd = pd.DataFrame(test.scaler_.transform(df_valnit_mnexpl.drop(
    columns = ['STAID'])
    ))
# give column names to transformed dataframe
df_valnit_trnsfrmd.columns = df_valnit_mnexpl.drop(
    columns = ['STAID']
).columns

# replace ohe columns with untransformed data
df_valnit_trnsfrmd[not_tr_in] = df_valnit_mnexpl[not_tr_in]
#########
# get predicted k's for each catchment           
km_valnit_pred = pd.DataFrame(
    {'STAID': df_valnit_mnexpl['STAID'],
    'K': test.km_clusterer_.predict(
            df_valnit_trnsfrmd
            )
    }
)
##########
# K-medoids

test.k_clust(
    ki = 2, kf = 30, 
    method = 'kmedoids', 
    plot_mean_sil = True, 
    plot_distortion = True,
    kmed_method = 'alternate')

for i in range(19, 23):
    test.plot_silhouette_vals(k = i)
#####
# Based on results from previous chunk, chose k = 8
# Here defining the test object to have a k-means model of k = 8 for projecting new data
test.k_clust(
    ki = 8, kf = 8, 
    method = 'kmedoids', 
    plot_mean_sil = False, 
    plot_distortion = False)

# get predicted k's for each catchment           
km_valnit_pred = test.km_clusterer_.predict(
    df_valnit_trnsfrmd
    )
# predicted clusters dataframe
df_km_valnit_pred = pd.DataFrame({
    'STAID': df_valnit_mnexpl['STAID'],
    'K_predicted': km_valnit_pred
})




# %% AggEcoregions as clusters

for i in df_train_ID['AggEcoregion'].unique():
    print(i)








# %% Call function to perform modeling

# Here are the function inputs
# def regress_fun(df_train_expl, # training data explanatory variables. Expects STAID to be a column
#                 df_valin_expl, # validation data explanatory variables using same catchments that were trained on
#                 df_valnit_expl, # validation data explanatory variables using different catchments than were trained on
#                 train_resp, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_mnanWY['Ann_WY_ft'])
#                 valin_resp, # validation data response variables using same catchments that were trained on
#                 valnit_resp, # validation data response variables using different catchments than were trained on
#                 train_ID, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
#                 valin_ID, # validation data id's from catchments used in training (e.g., clusters or ecoregions)
#                 valnit_ID, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
#                 clust_meth, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
#                 reg_in) # region label, i.e., 'NorthEast'

# subset data to catchment IDs that match the cluster or region being predicted
region_in = 'SECstPlain'
cidtrain_in = df_train_ID[df_train_ID['AggEcoregion'] == region_in]
cidvalin_in = df_valin_ID[df_valin_ID['AggEcoregion'] == region_in]
cidvalnit_in = df_valnit_ID[df_valnit_ID['AggEcoregion'] == region_in]

# Water yield
train_resp_in = pd.merge(
    df_train_mnanWY, cidtrain_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
valin_resp_in = pd.merge(
    df_valin_mnanWY, cidvalin_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
valnit_resp_in = pd.merge(
    df_valnit_mnanWY, cidvalnit_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
# explanatory variables
train_expl_in = pd.merge(df_train_mnexpl, cidtrain_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
)
valin_expl_in = pd.merge(df_valin_mnexpl, cidvalin_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
)
valnit_expl_in = pd.merge(df_valnit_mnexpl, cidvalnit_in, on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
)
# ID dataframes
train_ID_in = pd.merge(
    df_train_ID, cidtrain_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['ECO3_Site']
valin_ID_in = pd.merge(
    df_valin_ID, cidvalin_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['ECO3_Site']
valnit_ID_in = pd.merge(
    df_valnit_ID, cidvalnit_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['ECO3_Site']

regress_fun(df_train_expl = train_expl_in, # training data explanatory variables. Expects STAID to be a column
            df_valin_expl = valin_expl_in, # validation data explanatory variables using same catchments that were trained on
            df_valnit_expl = valnit_expl_in, # validation data explanatory variables using different catchments than were trained on
            train_resp = train_resp_in, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_mnanWY['Ann_WY_ft'])
            valin_resp = valin_resp_in, # validation data response variables using same catchments that were trained on
            valnit_resp = valnit_resp_in, # validation data response variables using different catchments than were trained on
            train_ID = train_ID_in, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
            valin_ID = valin_ID_in, # validation data id's from catchments used in training (e.g., clusters or ecoregions)
            valnit_ID = valnit_ID_in, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
            clust_meth = 'Ecoregion', # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
            reg_in = region_in, # region label, i.e., 'NorthEast'
            grid_in = { # dict with XGBoost parameters
                'n_estimators': [100, 500], #  [100, 500], # [100, 250, 500], # [10], # 
                'colsample_bytree': [1], # [1], # [0.7, 1], 
                'max_depth': [6], # [6], # [4, 6, 8],
                'gamma': [0], # [0], # [0, 1], 
                'reg_lambda': [0], # [0], # [0, 1, 2]
                'learning_rate': [0.3], # [0.3] # [0.02, 0.1, 0.3]
                }
            )       


# %%
