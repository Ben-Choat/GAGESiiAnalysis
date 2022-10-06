'''
Bchoat 2022/10/03
Script to calculate SHapley Additive exPlanations (SHAP) values
'''


# %% 
# Import Libraries
###########

import pandas as pd
import numpy as np
import shap
from Load_Data import load_data_fun
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import xgboost as xgb
import matplotlib.pyplot as plt
# from xgboost import plot_tree # can't get to work on windows (try on linux)





# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'AggEcoregion' # 'AggEcoregion', 'None', 

# define which region to work with
region =  'WestXeric' # 'NorthEast' # 'MxWdShld' #

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# directory where to place outputs
dir_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'

# Define features not to transform
not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 
            'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 
            'GEOL_REEDBUSH_DOM_volcanic',
            'year',
            'month',
            'day',
            'date']
            # 'GEOL_REEDBUSH_DOM_gneiss', 



# %%
# Load data
###########
# load data (explanatory, water yield, ID)
df_expl, df_WY, df_ID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'train',
    clust_meth = clust_meth,
    region = region,
    standardize = True # whether or not to standardize data
)

# read in columns that were previously removed due to high VIF
file = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_Removed/*{clust_meth}_{region}.csv')[0]
try:
    vif_removed = pd.read_csv(
        file
    )['columns_Removed']
except:
    vif_removed = pd.read_csv(
    file
    )['Columns_Removed']

# drop columns that were removed due to high VIF
df_expl.drop(vif_removed, axis = 1, inplace = True)


# remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
# subset WY to version desired (ft)
# store staid's and date/year/month
if(time_scale == 'mean_annual'):
    STAID = df_expl['STAID']  
    df_expl.drop('STAID', axis = 1, inplace = True)
    df_WY = df_WY['Ann_WY_ft']
if(time_scale == 'annual'):
    STAID = df_expl[['STAID', 'year']]  
    df_expl.drop(['STAID', 'year'], axis = 1, inplace = True)
    df_WY = df_WY['Ann_WY_ft']
if(time_scale == 'monthly'):
    STAID = df_expl[['STAID', 'year', 'month']]   
    df_expl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)
    df_WY = df_WY['Mnth_WY_ft']
if(time_scale == 'daily'):
    STAID = df_expl[['STAID', 'date']] 
    df_expl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace =True)
    df_WY = df_WY['dlyWY_ft']


# read in names of VIF_df csvs to see what models were used
###########
models_list = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_*')
print(models_list)

# read in results for the time_scale being worked with
results_summ = pd.read_pickle(
    f'{dir_work}/data_out/{time_scale}/combined/All_SummaryResults_{time_scale}.pkl'
)
# subset to clust_meth and region of interest
results_summ = results_summ[
    (results_summ['clust_method'] == clust_meth) & (results_summ['region'] == region)
    ]

results_summ

# %%
# Define, fit model, and return coefficients or other relevant info
###########


#######
# LASSO

# define model and parameters
model = Lasso(
    alpha = 0.05,
    max_iter = 1000
)

# fit model
model.fit(df_expl, df_WY)

# create dataframe with regression coefficients and variables
vars_temp = df_expl.columns[np.abs(model.coef_) > 10e-20]
coef_temp = [x for x in model.coef_ if np.abs(x) > 10e-20]
df_coef = pd.DataFrame({
    'features': vars_temp,
    'coef': coef_temp
})

# define X_in so you can call it below without needing to edit the text
X_in = df_expl





# %%
#######
# MLR
file = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_strd_mlr*.csv')[0]
# get variables appearing in final model
vars_keep = pd.read_csv(
    file
)['feature']

# subset data to variables used in model
X_in = df_expl[vars_keep]

# define model and parameters
reg = LinearRegression()

# apply linear regression using all explanatory variables
# this object is used in feature selection just below and
# results from this will be used to calculate Mallows' Cp
model = reg.fit(X_in, df_WY)



# create dataframe with regression coefficients and variables
vars_temp = vars_keep
coef_temp = model.coef_
df_coef = pd.DataFrame({
    'features': vars_temp,
    'coef': coef_temp
})




# %%
# XGBOOST
##########

# first define xgbreg object
model = xgb.XGBRegressor()

# define temp time_scale var since model names do not have '_' in them
temp_time = time_scale.replace('_', '')

# reload model into object
model.load_model(
    f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
    f'/Models/xgbreg_{temp_time}_XGBoost_{clust_meth}_{region}_model.json'
    )

X_in = df_expl

# xgb.plot_tree(model)

# %%
# Shapley Values
###########

# take subsample of specified % to use as background distribution
# % to subsample
ratio_sample = 0.50
Xsubset = shap.utils.sample(X_in, int(np.floor(X_in.shape[0] * ratio_sample)))

# define explainer
explainer = shap.Explainer(model.predict, Xsubset)
# calc shap values
shap_values = explainer(X_in) # df_expl)

# define the catchment for which to create a partial dependence plot and/or
# a waterfall plot
sample_ind = 50

# %%
# make a standard partial dependence plot
shap.partial_dependence_plot(
    'PPTAVG_BASIN', 
    model.predict, 
    Xsubset, 
    model_expected_value = True,
    feature_expected_value = True, 
    ice = False,
    shap_values = shap_values[sample_ind:sample_ind + 1, :]
)
# scatter
shap.plots.scatter(shap_values[:, 'PPTAVG_BASIN'])


# %%
# waterfall plot shows how we get from shap_values.base to model.predict(x)[sample_ind]
# for a given catchment (sample_ind)
shap.plots.waterfall(shap_values[sample_ind], 
    max_display = len(X_in.columns) + 1) # vars_temp


# %%
# beeswarm plot plots same values as waterfall, except for all samples
shap.plots.beeswarm(shap_values, 
    max_display = 20,#14, # len(vars_temp) + 1,
    show = False)

plt.title(f'{clust_meth}: {region}')

plt.show()


# %%
