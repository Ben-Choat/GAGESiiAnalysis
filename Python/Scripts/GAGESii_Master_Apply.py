# Ben Choat, 2022/07/06

# Script for executing analysis for dissertation on mean water yield and explanatory vars
# Ch4: Understanding drivers of water yield and how they vary across temporal scales

# Raw: Explanatory variables directly
# Rescale: Standardization or Normalization
# Reduce: PCA or UMAP
# Cluster: k-means, k-medoids, HDBSCAN, ecoregions
    # cluster on 1. all expl. vars, 2. physcial catchments vars, 3. anthropogenic vars,
    # 4. climate vars, 5. stack (2) and (3), 6. stack (2) and (4), 7. stack (2), (4), (3)

# Predict: OLS-MLR, Lasso, XGBOOST

# Raw -> Predict

# Raw -> Rescale -> Predict

# Raw -> Reduce -> Predict

# Raw -> Cluster -> Raw -> Predict

# Raw -> Cluster -> Cluster -> Predict

# Raw -> Rescale -> Reduce -> Predict

# Raw -> Rescale -> Cluster -> Predict

# Raw -> Reduce -> Cluster -> Predict

# Raw -> Rescale -> Reduce -> Cluster -> Predict

# %% Import classes and libraries
# from statistics import LinearRegression
from GAGESii_Class import Clusterer
from GAGESii_Class import Regressor
from Regression_PerformanceMetrics_Functs import *
import pandas as pd
import plotnine as p9
import plotly.express as px # easier interactive plots
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression

# %% load data

# water yield directory
dir_WY = 'E:/DataWorking/USGS_discharge/train_val_test'

# explantory var (and other data) directory
dir_expl = 'E:/Projects/GAGESii_ANNstuff/Data_Out'

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

# %% Define dataframe to append results to as models are generated
# try:

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
    'VIF': [] # variance inflation factor (vector of all included featuers and related VIFs)
})
# %% Raw -> Predict

# Define features not to transform
not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']
            # ,
            # 'TS_NLCD_11', 'TS_NLCD_12', 'TS_NLCD_21', 'TS_NLCD_22', 'TS_NLCD_23',
            # 'TS_NLCD_24', 'TS_NLCD_31', 'TS_NLCD_41', 'TS_NLCD_42', 'TS_NLCD_43',
            # 'TS_NLCD_52', 'TS_NLCD_71', 'TS_NLCD_81', 'TS_NLCD_82', 'TS_NLCD_90',
            # 'TS_NLCD_95', 'TS_NLCD_AG_SUM', 'TS_NLCD_DEV_SUM', 'TS_NLCD_imperv',
            # 'TS_ag_hcrop', 'TS_ag_irrig']

# apply transform to response var
# apply box-cox transformation
# train_mnanWY_tr, best_lambda = stats.boxcox(df_train_mnanWY['Ann_WY_ft'])

# regression on untransformed explanatory variables
# Instantiate a Regressor object 
regr = Regressor(expl_vars = df_train_mnexpl.drop(columns = ['STAID']),
    resp_var = df_train_mnanWY['Ann_WY_ft']) # train_mnanWY_tr)# 

# %%
##### Lasso regression
# search via cross validation
regr.lasso_regression(
    alpha_in = list(np.arange(0.01, 1.01, 0.01)), # must be single interger or list
    max_iter_in = 1000,
    n_splits_in = 10,
    n_repeats_in = 1, # 3,
    random_state_in = 100
)
# print all results from CV
pd.DataFrame(regr.lassoCV_results.cv_results_)

regr.df_lassoCV_mae[0:10, 70:80]
regr.df_lassoCV_rmse[0:10]
regr.df_lassoCV_r2[0:10]

# %%

##### apply best alpha identified in cross validation
# Lasso with alpha = 0.01
regr.lasso_regression(
    alpha_in = 0.01, # must be single interger or list
    max_iter_in = 1000,
    n_splits_in = 10,
    n_repeats_in = 3,
    random_state_in = 100
)

# print features kept by Lasso regression and the associated coefficients
# regr.df_lasso_features_coef_.sort_values(by = 'coefficients')

# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_train_mnexpl.drop(columns = 'STAID')
resp_in = df_train_mnanWY['Ann_WY_ft']
# resp_in = train_mnanWY_tr
id_in = df_train_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
# regr.df_pred_performance_
# print features and VIF
#  dict(regr.df_pred_performance_['VIF'])

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 0.01')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'train'

df_results = pd.concat([df_results, to_append], ignore_index = True)

# # Check linear regression assumptions
# # define ypred_in
# ypred_in = mdl_in.predict(expl_in)
# # plot diagnostic plots
# test_reg_assumpts(residuals = (ypred_in - resp_in), y_pred = ypred_in)

#####
# return prediction metrics for validation data from stations that were trained on
# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_valin_mnexpl.drop(columns = 'STAID')
resp_in = df_valin_mnanWY['Ann_WY_ft']
# resp_in = train_mnanWY_tr
id_in = df_valin_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
# regr.df_pred_performance_
# print features and VIF
#  dict(regr.df_pred_performance_['VIF'])

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 0.01')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'valin'

df_results = pd.concat([df_results, to_append], ignore_index = True)

#####
# return prediction metrics for validation data from stations that were NOT trained on
# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_valnit_mnexpl.drop(columns = 'STAID')
resp_in = df_valnit_mnanWY['Ann_WY_ft']
# resp_in = train_mnanWY_tr
id_in = df_valnit_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
# regr.df_pred_performance_
# print features and VIF
#  dict(regr.df_pred_performance_['VIF'])

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 0.01')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'valnit'

df_results = pd.concat([df_results, to_append], ignore_index = True)

# Write results to csv
df_results.to_excel(f'{dir_expl}/Results/Results_NonTimeSeries.xlsx', index = False)
# %%
##### Lasso with alpha = 1

# apply best alpha identified in cross validation
regr.lasso_regression(
    alpha_in = 1, # 1 # must be single interger or list
    max_iter_in = 1000,
    n_splits_in = 10, # only applicable when alpha_in is a list of values
    n_repeats_in = 3, # only applicable when alpha_in is a list of values
    random_state_in = 100
)

# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_train_mnexpl.drop(columns = 'STAID')
resp_in = df_train_mnanWY['Ann_WY_ft']
id_in = df_train_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 1')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'train'

df_results = pd.concat([df_results, to_append], ignore_index = True)

# # Check linear regression assumptions
# # define ypred_in
# ypred_in = mdl_in.predict(expl_in)
# # plot diagnostic plots
# test_reg_assumpts(residuals = (ypred_in - resp_in), y_pred = ypred_in)

#####
# return prediction metrics for validation data from stations that were trained on
# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_valin_mnexpl.drop(columns = 'STAID')
resp_in = df_valin_mnanWY['Ann_WY_ft']
# resp_in = train_mnanWY_tr
id_in = df_valin_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
# regr.df_pred_performance_
# print features and VIF
#  dict(regr.df_pred_performance_['VIF'])

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 1')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'valin'

df_results = pd.concat([df_results, to_append], ignore_index = True)

#####
# return prediction metrics for validation data from stations that were NOT trained on
# plot regression
mdl_in = regr.lasso_reg_
expl_in = df_valnit_mnexpl.drop(columns = 'STAID')
resp_in = df_valnit_mnanWY['Ann_WY_ft']
# resp_in = train_mnanWY_tr
id_in = df_valnit_ID['AggEcoregion']

regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
# regr.df_pred_performance_
# print features and VIF
#  dict(regr.df_pred_performance_['VIF'])

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_lasso'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('alpha = 1')]
# specify if results are from training data or validation (in) or validation (not it)
to_append['train_val'] = 'valnit'

df_results = pd.concat([df_results, to_append], ignore_index = True)


##### 
# %% feature selection - 'forward'
regr.lin_regression_select(
    sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
    float_opt = 'True', # 'True' or 'False'
    min_k = 1, # only active for 'exhaustive' option
    klim_in = 81, # controls max/min number of features for forward/backward selection
    timeseries = False) # if timeseries = True then NSE and KGE are also calculated

# print performance metric dataframe subset to n_features of desired number
# regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == 20,]
# define variable holding the selected features and vifs.
vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == 20, 'VIF']

# Extract feature names for selecting features
features_in = pd.DataFrame(dict((vif_in))).index

# Subset appropriate explanatory variables to columns of interest
# validation data from catchments used in training
expl_in = df_train_mnexpl[features_in]

# define response variable
resp_in = df_train_mnanWY['Ann_WY_ft']

# define id vars
id_in = df_train_ID['AggEcoregion']
# id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

#
# OLS regression predict
# specifiy input model
mdl_in = LinearRegression().fit(
            # df_train_mnexpl[features_in], df_train_mnanWY['Ann_WY_ft']
            expl_in, resp_in
            )

# Create predict-plot object
regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_mlr'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('forward', 'klim_in = 81')]

df_results = pd.concat([df_results, to_append], ignore_index = True)

# %%
#####
# feature selection - backward
regr.lin_regression_select(
    sel_meth = 'backward', # 'forward', 'backward', or 'exhaustive'
    float_opt = 'True', # 'True' or 'False'
    min_k = 18, # only active for 'exhaustive' option
    klim_in = 1, # controls max/min number of features for forward/backward selection
    timeseries = False) # if timeseries = True then NSE and KGE are also calculated

# define variable holding the selected features and vifs.
vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == 16, 'VIF']

# Extract feature names for selecting features
features_in = pd.DataFrame(dict((vif_in))).index

# Subset appropriate explanatory variables to columns of interest
# validation data from catchments used in training
expl_in = df_train_mnexpl[features_in]

# define response variable
resp_in = df_train_mnanWY['Ann_WY_ft']

# define id vars
id_in = df_train_ID['AggEcoregion']
# id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

#
# OLS regression predict
# specifiy input model
mdl_in = LinearRegression().fit(
            # df_train_mnexpl[features_in], df_train_mnanWY['Ann_WY_ft']
            expl_in, resp_in
            )

# Create predict-plot object
regr.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# append results to df_results
to_append = regr.df_pred_performance_
# include model
to_append['model'] = 'raw_mlr'
# include hyperparameters (tuning parameters)
to_append['parameters'] = [('backward', 'klim_in = 1')]

df_results = pd.concat([df_results, to_append], ignore_index = True)

# %%
##### NOTE: exhaustive search with small k (n choose k) take a very a long, so commented out
# only run if you are sure you want to.
# feature selection - exhaustive 
# regr.lin_regression_select(
#     sel_meth = 'exhaustive', # 'forward', 'backward', or 'exhaustive'
#     float_opt = 'True', # 'True' or 'False'
#     min_k = 10, # only active for 'exhaustive' option
#     klim_in = 25, # controls max/min number of features for forward/backward selection, or max for exhaustive
#     timeseries = False) # if timeseries = True then NSE and KGE are also calculated

# # define variable holding the selected features and vifs.
# vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == 16, 'VIF']

# # Extract feature names for selecting features
# features_in = pd.DataFrame(dict((vif_in))).index

# # Subset appropriate explanatory variables to columns of interest
# # validation data from catchments used in training
# expl_in = df_train_mnexpl[features_in]

# # define response variable
# resp_in = df_train_mnanWY['Ann_WY_ft']

# # define id vars
# id_in = df_train_ID['AggEcoregion']
# # id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

# #
# # OLS regression predict
# # specifiy input model
# mdl_in = LinearRegression().fit(
#             # df_train_mnexpl[features_in], df_train_mnanWY['Ann_WY_ft']
#             expl_in, resp_in
#             )

# # Create predict-plot object
# regr.pred_plot(
#     model_in = mdl_in,
#     X_pred =  expl_in,
#     y_obs = resp_in,
#     id_vars = id_in
# )

# # append results to df_results
# to_append = regr.df_pred_performance_
# # include model
# to_append['model'] = 'raw_mlr'
# # include hyperparameters (tuning parameters)
# to_append['parameters'] = [('backward', 'klim_in = 1')]

# df_results = pd.concat([df_results, to_append], ignore_index = True)
# %% Check assumptions of linear regerssion and explore transformations
# plot histogram of flow data
dist_flow = (
    p9.ggplot(data = df_train_mnanWY) +
    p9.geom_histogram(p9.aes(x = 'Ann_WY_ft'))
)

dist_flow

# apply box-cox transformation
df_train_mnanWY_tr, best_lambda = stats.boxcox(df_train_mnanWY['Ann_WY_ft'])

dist_flow_tr = (
    p9.ggplot() +
    p9.geom_histogram(p9.aes(x = df_train_mnanWY_tr))
)

dist_flow_tr

