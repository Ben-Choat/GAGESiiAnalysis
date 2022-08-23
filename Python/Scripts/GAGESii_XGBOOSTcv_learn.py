# 2022/07/29 B Choat, 
# script to apply xgboost with cross validation
# testing methods and learning what will work best


# %% Load Libraries

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import umap
import plotnine as p9
from Regression_PerformanceMetrics_Functs import *

# %% Info about XGBoost parameters
# XGBOOST parameters and hyperparameters 
#   (https://www.datacamp.com/tutorial/xgboost-in-python)
# learning_rate (aka eta): step size shrinkage used to prevent overfitting. 
#   Range is [0,1] # based on kaggle, common values are 0.01 - 0.2
# max_depth: determines how deeply each tree is allowed to grow during 
#   any boosting round.# based on kaggle, common values are 3 - 10
# subsample: percentage of samples used per tree. Low value can lead 
#   to underfitting.
# colsample_bytree: percentage of features used per tree. High value can 
#   lead to overfitting
# n_estimators: number of trees you want to build
# objective: determines the loss function to be used like reg:linear 
#   for regression problems, reg:logistic for classification problems 
#   with only decision, binary:logistic for classification problems with 
#   probability.
# tree_method: tree construction algorithm used by XGBoost. 'approx', 'hist' 
#   or 'gpu_hist' for distributed trianing.
# REGULARIZATION PARAMETERS
# gamma: controls whether a given node will split based on the expected 
#   reduction in loss after the split. A higher value leads to fewer splits. 
#   Supported only for tree-based learners. Larger gamma => more conservative [0, INF]
# alpha: L1 regularization on leaf weights. A large value leads to 
#   more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 
#   regularization; [0, INF]

# more helpful info about hyperparameters can be found here:
# https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook

# %% Load Data


# # water yield directory
# dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# # explantory var (and other data) directory
# dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# # GAGESii explanatory vars
# # training
# df_train_expl = pd.read_csv(
#     f'{dir_expl}/ExplVars_Model_In/All_ExplVars_Train_Interp_98_12.csv',
#     dtype = {'STAID': 'string'}
# ).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])
# # val_in
# df_testin_expl = pd.read_csv(
#     f'{dir_expl}/ExplVars_Model_In/All_ExplVars_testin_Interp_98_12.csv',
#     dtype = {'STAID': 'string'}
# ).drop(columns = ['LAT_GAGE', 'LNG_GAGE', 'GEOL_REEDBUSH_DOM_anorthositic'])
# # val_nit
# df_valnit_expl = pd.read_csv(
#     f'{dir_expl}/ExplVars_Model_In/All_ExplVars_ValNit_Interp_98_12.csv',
#     dtype = {'STAID': 'string'}
# ).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])


# # Explanatory variables
# # Annual Water yield
# # training
# df_train_anWY = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_train.csv',
#     dtype = {"site_no":"string"}
#     )
# # drop stations not in explantory vars
# df_train_anWY = df_train_anWY[
#     df_train_anWY['site_no'].isin(df_train_expl['STAID'])
#     ].reset_index(drop = True)
# # create annual water yield in ft
# df_train_anWY['Ann_WY_ft'] = df_train_anWY['Ann_WY_ft3']/(
#     df_train_expl['DRAIN_SQKM']*(3280.84**2)
#     )

# # val_in
# df_testin_anWY = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_val_in.csv',
#     dtype = {"site_no":"string"}
#     )
# # drop stations not in explantory vars    
# df_testin_anWY = df_testin_anWY[
#     df_testin_anWY['site_no'].isin(df_testin_expl['STAID'])
#     ].reset_index(drop = True)
# # create annual water yield in ft
# df_testin_anWY['Ann_WY_ft'] = df_testin_anWY['Ann_WY_ft3']/(
#     df_testin_expl['DRAIN_SQKM']*(3280.84**2)
#     )

# # val_nit
# df_valnit_anWY = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_val_nit.csv',
#     dtype = {"site_no":"string"}
#     )
# # drop stations not in explantory vars
# df_valnit_anWY = df_valnit_anWY[
#     df_valnit_anWY['site_no'].isin(df_valnit_expl['STAID'])
#     ].reset_index(drop = True)
# # subset testint expl and response vars to common years of interest
# df_valnit_expl = pd.merge(
#     df_valnit_expl, 
#     df_valnit_anWY, 
#     how = 'inner', 
#     left_on = ['STAID', 'year'], 
#     right_on = ['site_no', 'yr']).drop(
#     labels = df_valnit_anWY.columns, axis = 1
# )
# df_valnit_anWY = pd.merge(df_valnit_expl, 
#     df_valnit_anWY, 
#     how = 'inner', 
#     left_on = ['STAID', 'year'], 
#     right_on = ['site_no', 'yr']).drop(
#     labels = df_valnit_expl.columns, axis = 1
# )
# df_valnit_anWY['Ann_WY_ft'] = df_valnit_anWY['Ann_WY_ft3']/(
#     df_valnit_expl['DRAIN_SQKM']*(3280.84**2)
#     )

# # mean annual water yield
# # training
# df_train_mnanWY = df_train_anWY.groupby(
#     'site_no', as_index = False
# ).mean().drop(columns = ["yr"])
# # val_in
# df_testin_mnanWY = df_testin_anWY.groupby(
#     'site_no', as_index = False
# ).mean().drop(columns = ["yr"])
# # val_nit
# df_valnit_mnanWY = df_valnit_anWY.groupby(
#     'site_no', as_index = False
# ).mean().drop(columns = ["yr"])

# # mean GAGESii explanatory vars
# # training
# df_train_mnexpl = df_train_expl.groupby(
#     'STAID', as_index = False
# ).mean().drop(columns = ['year'])
# # val_in
# df_testin_mnexpl = df_testin_expl.groupby(
#     'STAID', as_index = False
# ).mean().drop(columns = ['year'])
# #val_nit
# df_valnit_mnexpl = df_valnit_expl.groupby(
#     'STAID', as_index = False
# ).mean().drop(columns = ['year'])

# # ID vars (e.g., ecoregion)
# # vars to color plots with (e.g., ecoregion)
# df_ID = pd.read_csv(
#     f'{dir_expl}/GAGES_idVars.csv',
#     dtype = {'STAID': 'string'}
# )

# # training ID
# df_train_ID = df_ID[df_ID.STAID.isin(df_train_expl.STAID)].reset_index(drop = True)
# # val_in ID
# df_testin_ID = df_train_ID
# # val_nit ID
# df_valnit_ID = df_ID[df_ID.STAID.isin(df_valnit_expl.STAID)].reset_index(drop = True)

# del(df_train_anWY, df_train_expl, df_testin_anWY, df_testin_expl, df_valnit_anWY, df_valnit_expl)


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


# %% Define data to use with xgboost
# define input data matrices
# training
Xtrain = df_train_mnexpl.drop(columns = 'STAID')
# Remove variables with a pearsons r > 0.95

# using algorith found here:
# https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# define working data
# Xtrain = df_train_mnexpl.drop(columns = ['STAID'])

# # calculate correlation
# df_cor = Xtrain.corr().abs()

# # Select upper triangle of correlation matrix
# upper = df_cor.where(np.triu(np.ones(df_cor.shape), k=1).astype(bool))

# # sort upper columns so largest variables with largest max correlation are 
# # seen first in the loop below
# # upper = upper.loc[:, upper.max().sort_values(ascending=False).index]

# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# # Drop features 
# Xtrain_dr = Xtrain.drop(to_drop, axis=1) #, inplace=True)
ytrain = df_train_mnanWY['Ann_WY_ft']
Xtestin = df_testin_mnexpl.drop(columns = 'STAID')
ytestin = df_testin_mnanWY['Ann_WY_ft']
Xvalnit = df_valnit_mnexpl.drop(columns = 'STAID')
yvalnit = df_valnit_mnanWY['Ann_WY_ft']

# %% k-fold Cross Validation using XGBoost


# define model

# define xgboost model
xgb_reg = xgb.XGBRegressor(
    objective = 'reg:squarederror',
    tree_method = 'hist', # 'gpu_hist',
    # colsample_bytree = 0.7, # 0.3, # default = 1
    # learning_rate = 0.1, # default = 0.3
    # max_depth = 5, # default = 6
    # gamma = 0, # default = 0
    # reg_alpha = 10, # default = 0
    # reg_lambda = 1, # default = 1
    # n_estimators = 100,
    verbosity = 1, # 0 = silent, 1 = warning (default), 2 = info, 3 = debug
    sampling_method = 'uniform', # 'gradient_based', # default is 'uniform'
    # nthread = 4 defaults to maximum number available, only use for less threads
)


# define search grid of params


# the following commented out grid took 11m 8.4s on campus pc
# it has a total of 108 parameters to consider

# grid = {
#     'n_estimators': [50, 100, 150], # [10], # 
#     'colsample_bytree': [0.7, 1], # [0.3, 0.7, 1],
#     'max_depth': [4, 6, 8], # [6], # 
#     # 'reg_alpha': [1.1, 1.2, 1.3],
#     'reg_lambda': [1, 2], # [0, 1, 2]
#     'learning_rate': [0.1, 0.3, 0.5]
# }


grid = {
    'n_estimators': [100, 250, 500], # [10], # 
    'colsample_bytree': [0.7, 1], # [0.3, 0.7, 1],
    'max_depth': [4, 6, 8], # [6], # 
    'gamma': [0, 1], 
    # 'reg_alpha': [1.1, 1.2, 1.3],
    'reg_lambda': [1, 2], # [0, 1, 2]
    'learning_rate': [0.02, 0.1, 0.2]
}

gsCV = GridSearchCV(
    estimator = xgb_reg,
    param_grid = grid,
    scoring = 'neg_root_mean_squared_error',
    cv = 5, # default = 5
    return_train_score = False, # True, # default = False
    verbose = 2,
    n_jobs = -1
)

cv_fitted = gsCV.fit(Xtrain, ytrain)

# print results
df_cv_results = pd.DataFrame(cv_fitted.cv_results_).sort_values(by = 'rank_test_score')

# print best parameters
print(cv_fitted.best_params_)

# %% apply XGBOOST using best parameters from cross-validation

# define xgboost model
xgb_reg = xgb.XGBRegressor(
    objective = 'reg:squarederror',
    tree_method = 'hist', # 'gpu_hist',
    colsample_bytree = 1, # 0.7, # 0.3, # default = 1
    learning_rate = 0.1, # default = 0.3
    max_depth = 4, # 4, #5, # default = 6
    gamma = 0, # default = 0
    # reg_alpha = 10, # default = 0
    reg_lambda = 1, # 2, # default = 1
    n_estimators = 250,
    verbosity = 1, # 0 = silent, 1 = warning (default), 2 = info, 3 = debug
    sampling_method = 'uniform', # 'gradient_based', # default is 'uniform'
    # nthread = 4 defaults to maximum number available, only use for less threads
)

# train model
xgb_reg.fit(Xtrain, ytrain)
# xgb_reg.fit(Xvalnit, yvalnit)

# return fit from training
train_pred = xgb_reg.predict(Xtrain)
rmse = mean_squared_error(ytrain, train_pred)
mae = mean_absolute_error(ytrain, train_pred)
r2 = r2_score(ytrain, train_pred)

print(
    '---TRAINING--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )
###
# return fit from testin
testin_pred = xgb_reg.predict(Xtestin)
rmse = mean_squared_error(ytestin, testin_pred)
mae = mean_absolute_error(ytestin, testin_pred)
r2 = r2_score(ytestin, testin_pred)

print(
    '---testin--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )
###
# return fit from valnit
valnit_pred = xgb_reg.predict(Xvalnit)
rmse = mean_squared_error(yvalnit, valnit_pred)
mae = mean_absolute_error(yvalnit, valnit_pred)
r2 = r2_score(yvalnit, valnit_pred)

print(
    '---VALNIT--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )


# %% ##################
# Save Model and reload model
########################

# save model
xgb_reg.save_model('D:/Projects/GAGESii_ANNstuff/Python/Scripts/Learning_Results/xgbreg_learn_model.json')

# load model

# first define xgbreg object
xgb_reg_loaded = xgb.XGBRegressor()
# reload model into object
xgb_reg_loaded.load_model('D:/Projects/GAGESii_ANNstuff/Python/Scripts/Learning_Results/xgbreg_learn_model.json')

# test reloaded correctly by repredicting data as above

# return fit from training
train_pred = xgb_reg_loaded.predict(Xtrain)
rmse = mean_squared_error(ytrain, train_pred)
mae = mean_absolute_error(ytrain, train_pred)
r2 = r2_score(ytrain, train_pred)

print(
    '---TRAINING--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )
###
# return fit from testin
testin_pred = xgb_reg_loaded.predict(Xtestin)
rmse = mean_squared_error(ytestin, testin_pred)
mae = mean_absolute_error(ytestin, testin_pred)
r2 = r2_score(ytestin, testin_pred)

print(
    '---testin--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )
###
# return fit from valnit
valnit_pred = xgb_reg_loaded.predict(Xvalnit)
rmse = mean_squared_error(yvalnit, valnit_pred)
mae = mean_absolute_error(yvalnit, valnit_pred)
r2 = r2_score(yvalnit, valnit_pred)

print(
    '---VALNIT--- \n'
    f'RMSE: {rmse: .2f} \n'
    f'mae: {mae: .2f} \n'
    f'R2: {r2: .2f}'
    )


# %% Permutation Importance
############################

# this method replaces each feature with a shuffled version the
# specified number of times. It subtracts the average score with
# shuffled data from the score with unshuffled data to arrive at 
# importance.

# a negative value essentially means that the model performed better
# when the feature with a negative score was shuffled.

X_in = Xvalnit
y_in = yvalnit

mdl_score = xgb_reg.score(X_in,
                        y_in)

print(np.round(mdl_score, 3))

scores = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']

perm_imp = permutation_importance(xgb_reg,
    X = X_in, 
    y = y_in, 
    scoring = scores,
    n_repeats = 10, # 30
    n_jobs = -1,
    random_state = 100)

# assign results to pandas dataframe
df_perm_imp = pd.DataFrame({
    'Features': X_in.columns,
    'r2_MeanImp': perm_imp['r2'].importances_mean,
    'r2_StDev': perm_imp['r2'].importances_std,
    'RMSE_MeanImp': perm_imp['neg_root_mean_squared_error'].importances_mean,
    'RMSE_StDev': perm_imp['neg_root_mean_squared_error'].importances_std,
    'MAE_MeanImp': perm_imp['neg_mean_absolute_error'].importances_mean,
    'MAE_StDev': perm_imp['neg_mean_absolute_error'].importances_std
}).sort_values(by = 'r2_MeanImp', 
                ascending = False, 
                ignore_index = True)

df_perm_imp.head(25)

# df_perm_imp.tail(35)


# %% DELETE OR MOVE LATER
# investigate 6 poorly predicted catchments

# create dataframe with predicted, observed and STAID
df_valnit_inv = pd.DataFrame({
    'STAID': df_valnit_mnexpl['STAID'],
    'predicted': valnit_pred,
    'observed': yvalnit
})

df_valnit_inv['diff'] = df_valnit_inv['predicted'] - df_valnit_inv['observed']


# add binary target column based on a defined residual (diff) threshold
diff_th = 1

df_valnit_inv['of_int'] = np.where(df_valnit_inv['diff'] > diff_th, 1, 0)

df_valnit_inv['of_int'].hist()

# define input parameters
# X_in = df_valnit_inv.drop(columns = ['STAID', 'of_int'])
# y_in = df_valnit_inv['of_int']

X_in = df_valnit_mnexpl.drop(columns = 'STAID')
y_in = df_valnit_inv['of_int']

# adversarial validation identifying only those catchments that perform
# poorly within the valnit data

xgbClass = xgb.XGBClassifier(
    objective = 'binary:logistic',
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 3,
    random_state = 100,
    use_label_encoder = False
)

# cross validate
class_cv = cross_validate(
    estimator = xgbClass,
    X = X_in,
    y = y_in,
    scoring = 'roc_auc',
    cv = 10,
    n_jobs = -1 
)

pd.DataFrame(class_cv)

# mean of test scores
ts_cv = np.round(class_cv['test_score'].mean(), 3)

# fit xgbClass obj to X_in and y_in
xgbClass.fit(X_in, y_in)

xgb_pred = xgbClass.predict(X_in)

pred_score = np.round(roc_auc_score(y_in, xgb_pred), 3)

perm_imp = permutation_importance(
    xgbClass,
    X = X_in, 
    y = y_in, 
    scoring = 'roc_auc', #scores,
    n_repeats =  30, # 10, #
    n_jobs = 8, #-1,
    random_state = 100)

df_perm_imp = pd.DataFrame({
    'Features': X_in.columns,
    'AUC_imp': perm_imp['importances_mean']
}).sort_values(by = 'AUC_imp', 
                ascending = False, 
                ignore_index = True)

print(f'\n ROC-AUC from fitted model: {pred_score} \n')
print(f'\n Mean Test ROC-AUC: {ts_cv} \n')
print(f'\n Number of catchments meeting threshold ({diff_th} ft): {sum(y_in)} \n')

df_perm_imp.head(10)
# df_perm_imp.tail(25)



#%% Extra code from exploring poorly performing catchments

# df_in = df_valnit_mnexpl
# df_in['of_int'] = np.where(df_in['STAID'].isin(df_poor['STAID']), 1, 0)

# 1 = yes, of interest
# 0 = not of interest

# # plot impervious against population density (or other vars)
# p = (
#         p9.ggplot(data = df_in) +
#         p9.geom_point(p9.aes(x = 'TS_WaterUse_wu', # 'TS_NLCD_imperv', 
#                                 y = 'TS_Population_PDEN',
#                                 color = 'of_int',
#                                 alpha = 'of_int')) +
#         p9.scale_alpha_discrete(range = [0.1, 1])
# )

# p

# # other exploratory plots
# p = (
#         p9.ggplot(data = df_in) +
#         p9.geom_point(p9.aes(x = 'WD_BASIN', # 'TS_NLCD_imperv', 
#                                 y = 'PPTAVG_BASIN',
#                                 color = 'of_int',
#                                 alpha = 'of_int')) +
#         p9.scale_alpha_discrete(range = [0.1, 1])
# )

# p

# boxplots of catchments of interest vs others

# p = (
#     p9.ggplot(data = df_in) +
#     p9.geom_boxplot(p9.aes(x = 1, y = 'TS_Population_PDEN'))
# )

# p