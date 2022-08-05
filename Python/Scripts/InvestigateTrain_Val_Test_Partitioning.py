# %% 8/5/2022

# After some initial modeling, it appears there are at least some catchments in
# the valnit data that are poorly represented by the training data.
# That is to say, the models perform poorly on those catchments.

# Here I am exploring some moethodologies for comparing the different partitions
# of data.

# %% import libraries

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
import plotnine as p9


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

# %% Info about XGBoost parameters for regression
# XGBOOST parameters and hyperparameters 
#   (https://www.datacamp.com/tutorial/xgboost-in-python)
# learning_rate (aka eta): step size shrinkage used to prevent overfitting. 
#   Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during 
#   any boosting round.
# subsample: percentage of samples used per tree. Low value can lead 
#   to underfitting.
# colsample_bytree: percentage of features used per tree. High value can 
#   lead to overfitting
# n_estimators: number of trees you want to build
# objective: determines the loss function to be used like reg:linear 
#   for regression problems, reg:logistic for classification problems 
#   with only decision, binary:logistic for classification problems with 
#   probability.
# tree_method: tree construction algorithm used by XGBoost. 'approse', 'hist' 
#   or 'gpu_hist' for distributed trianing.
# REGULARIZATION PARAMETERS
# gamma: controls whether a given node will split based on the expected 
#   reduction in loss after the split. A higher value leads to fewer splits. 
#   Supported only for tree-based learners. Larger gamma => more conservative
# alpha: L1 regularization on leaf weights. A large value leads to 
#   more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 
#   regularization

# %% Load Data

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


# %% advaserial validation on training and valnit data

# define input parameters
# training data
X_train = df_train_mnexpl.drop(columns = 'STAID')
X_train['AV_label'] = 0
# y_train = df_train_mnanWY['Ann_WY_ft']
# testing data
X_valnit = df_valnit_mnexpl.drop(columns = 'STAID')
X_valnit['AV_label'] = 1
# y_valin = df_valnit_mnanWY['Ann_WY_ft']

# combine into one dataset
all_data = pd.concat([X_train, X_valnit], axis = 0, ignore_index = True)

# shuffle
all_data_shuffled = all_data.sample(frac = 1)

# create DMatrix
X_all = all_data_shuffled.drop(columns = 'AV_label')
y_all = all_data_shuffled['AV_label']


# Instantiate classifier object

xgbClassifier = xgb.XGBClassifier(
    objective = 'binary:logistic',
    n_estimators = 100,
    learning_rate = 0.1,
    max_depth = 3,
    random_state = 100,
    use_label_encoder = False
)

# apply cross validation

classify_cv = cross_validate(
    estimator = xgbClassifier,
    X = X_all,
    y = y_all,
    scoring = 'roc_auc',
    cv = 10,
    n_jobs = -1
)


# calculate mean roc-auc test score
ts_cv = np.round(classify_cv['test_score'].mean(), 3)


# investigate importance features
# fit classifier
xgbClassifier.fit(X_all, y_all)

# return roc auc score for prediction of training vs valnit
ts_pred = np.round(roc_auc_score(y_all, xgbClassifier.predict(X_all)), 3)


# xgb.plot_importance(xgbClassifier, max_num_features = 10)

perm_imp = permutation_importance(
    xgbClassifier,
    X = X_all, 
    y = y_all, 
    scoring = 'roc_auc', #scores,
    n_repeats = 10, # 30
    n_jobs = -1,
    random_state = 100)


df_perm_imp = pd.DataFrame({
    'Features': X_all.columns,
    'AUC_imp': perm_imp['importances_mean']
}).sort_values(by = 'AUC_imp', 
                ascending = False, 
                ignore_index = True)



print(f'Mean Test ROC-AUC: {ts_cv}')
# auc-roc of 0.79 suggests values are not from the same distribution
# a value of 1 means train and valnit can be perfectly identified (this is bad)
# a value of 0.5 is desirable and suggests both datasets are from the same
# distribution
print(f'ROC-AUC from fitted model: {ts_pred}')
df_perm_imp.head(10)
# df_perm_imp.tail(25)