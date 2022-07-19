# File specifically for finding best hyperparemters for the model pipeline of
# raw -> stdrd -> umap -> lasso
# same umap paramters identified here will be applied for the pipeline of
# raw -> stdrd -> umap -> ols-mlr



# %% import libraries

from xmlrpc.server import SimpleXMLRPCDispatcher
import pandas as pd
import numpy as np
import scipy.stats as spst
from sklearn.linear_model import Lasso
import umap
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


# mean annual water yield
# training
df_train_mnanWY = df_train_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])

# mean GAGESii explanatory vars
# training
df_train_mnexpl = df_train_expl.groupby(
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


del(df_train_anWY, df_train_expl, df_ID)


# %% Define models and pipeline
# define standard scaler
scaler = StandardScaler()

# define umap object
umap_reducer = umap.UMAP(
                n_components = 40,
                spread = 1, 
                random_state = 100, 
                n_epochs = 200, 
                n_jobs = -1)

# define lasso object
lasso_mod = Lasso(max_iter = 1000)

# define pipline
pipe = Pipeline(steps = [('scaler', scaler), ('umap_reducer', umap_reducer), ('lasso_mod', lasso_mod)])

# # %% Define parameter grid and gridsearch (exhaustive)

# parameters for the pipeline search
param_grid = {
    'umap_reducer__n_neighbors': [2, 20, 40, 60, 80, 100], # for initial run do 6 for each
    'umap_reducer__min_dist': np.arange(0.01, 1.1, 0.2),
    'lasso_mod__alpha': [0.01] # np.arange(0.01, 1.01, 0.01)
}

search = GridSearchCV(pipe, param_grid, n_jobs = -1)
# %%
"""
# %% Define parameter grid and gridsearch (random)

# # parameters for the pipeline search
# param_grid = {
#     'umap_reducer__n_neighbors': spst.uniform(0, 100).rvs(5), # [2, 10], # , 20, 30, 40, 50, 60, 70, 80, 90, 100],
#     'umap_reducer__min_dist': spst.uniform(0, 1).rvs(5),# [0.01, 0.99], # np.arange(0.01, 1.1, 0.1)
# }

# search = GridSearchCV(pipe, param_grid, n_jobs = -1)
"""
# %% Apply gridsearch to data and return best fit parameters

search.fit(df_train_mnexpl.drop(columns = 'STAID'), df_train_mnanWY['Ann_WY_ft'])
search.best_params_
# results are here:
# {'lasso_mod__alpha': 0.01,
#  'umap_reducer__min_dist': 0.81,
#  'umap_reducer__n_neighbors': 80}

# print results
pd.DataFrame(search.cv_results_)
# %%
