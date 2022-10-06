'''
BChoat 2022/10/05
Script to plot timeseries results
'''


# %% 
# Import Libraries
###########

import pandas as pd
import numpy as np
from Load_Data import load_data_fun
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import xgboost as xgb
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sns
from Regression_PerformanceMetrics_Functs import *
from NSE_KGE_timeseries import NSE_KGE_Apply
from statsmodels.distributions.empirical_distribution import ECDF


# from xgboost import plot_tree # can't get to work on windows (try on linux)





# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'AggEcoregion' # 'AggEcoregion', 'None', 

# define which region to work with
region =  'SECstPlain' # 'WestXeric' # 'NorthEast' # 'MxWdShld' #

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# define which model you want to work with
model_work = 'xgboost' # ['XGBoost', 'strd_mlr', 'strd_lasso]

# # directory where to place outputs
# dir_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'




# %%
# Load data
###########
# load data (explanatory, water yield, ID)
df_trainexpl, df_trainWY, df_trainID = load_data_fun(
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
df_trainexpl.drop(vif_removed, axis = 1, inplace = True)


# remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
# subset WY to version desired (ft)
# store staid's and date/year/month
if(time_scale == 'mean_annual'):
    STAID = df_trainexpl['STAID']  
    df_trainexpl.drop('STAID', axis = 1, inplace = True)
    df_trainWY = df_trainWY['Ann_WY_ft']
if(time_scale == 'annual'):
    STAID = df_trainexpl[['STAID', 'year']]  
    df_trainexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
    df_trainWY = df_trainWY['Ann_WY_ft']
if(time_scale == 'monthly'):
    STAID = df_trainexpl[['STAID', 'year', 'month']]   
    df_trainexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)
    df_trainWY = df_trainWY['Mnth_WY_ft']
if(time_scale == 'daily'):
    STAID = df_trainexpl[['STAID', 'date']] 
    df_trainexpl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace =True)
    df_trainWY = df_trainWY['dlyWY_ft']


# # read in names of VIF_df csvs to see what models were used
# ###########
# models_list = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_*')
# print(models_list)

# read in results for the time_scale being worked with
results_summ = pd.read_pickle(
    f'{dir_work}/data_out/{time_scale}/combined/All_SummaryResults_{time_scale}.pkl'
)
# subset to clust_meth and region of interest
results_summ = results_summ[
    (results_summ['clust_method'] == clust_meth) & (results_summ['region'] == region)
    ]

# repeat for individual catchments
results_ind = pd.read_pickle(
    f'{dir_work}/data_out/{time_scale}/combined/All_IndResults_{time_scale}.pkl'
)
# subset to clust_meth and region of interest
results_ind = results_ind[
    (results_ind['clust_method'] == clust_meth) & (results_ind['region'] == region)
    ]

results_summ

results_ind
# %%
# Define, fit model, and return coefficients or other relevant info
###########


#######
# LASSO
if model_work == 'strd_lasso':
    # define model and parameters
    model = Lasso(
        alpha = 0.17,
        max_iter = 1000
    )

    # fit model
    model.fit(df_trainexpl, df_trainWY)

    # create dataframe with regression coefficients and variables
    vars_temp = df_trainexpl.columns[np.abs(model.coef_) > 10e-20]
    coef_temp = [x for x in model.coef_ if np.abs(x) > 10e-20]
    df_coef = pd.DataFrame({
        'features': vars_temp,
        'coef': coef_temp
    })

    # define X_in so you can call it below without needing to edit the text
    X_in = df_trainexpl




# %%
#######
# MLR
if model_work == 'strd_mlr':
    file = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_strd_mlr*.csv')[0]
    # get variables appearing in final model
    vars_keep = pd.read_csv(
        file
    )['feature']

    # DRAIN_SQKM stored as DRAIN_SQKM_x in some output by accident,
    # so replace it
    vars_keep = vars_keep.str.replace('SQKM_x', 'SQKM')

    # subset data to variables used in model
    X_in = df_trainexpl[vars_keep]


    # define model and parameters
    reg = LinearRegression()

    # apply linear regression using all explanatory variables
    # this object is used in feature selection just below and
    # results from this will be used to calculate Mallows' Cp
    model = reg.fit(X_in, df_trainWY)



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
if model_work == 'XGBoost':
    # first define xgbreg object
    model = xgb.XGBRegressor()

    # define temp time_scale var since model names do not have '_' in them
    temp_time = time_scale.replace('_', '')

    # reload model into object
    model.load_model(
        f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
        f'/Models/xgbreg_{temp_time}_XGBoost_{clust_meth}_{region}_model.json'
        )

    X_in = df_trainexpl




# %%
# Plot

# predict WY
y_pred = model.predict(X_in)

# define array to use as label of observed or predicted

lab_in = np.hstack([
    np.repeat('Obs', len(y_pred)),
    np.repeat('Pred', len(y_pred))
])

# create dataframe for plotting
plot_data = pd.DataFrame({
    'STAID': np.hstack([STAID['STAID'], STAID['STAID']]),
    'year': np.hstack([STAID['year'], STAID['year']]),
    'WY_ft': np.hstack([df_trainWY, y_pred]),
    'label': lab_in
})

# define empty array to append NSE to
nse_out = []

# subset data to data of interest
# plot_in = plot_data
STAID_work = STAID['STAID'].unique() # ['01305500', '02309848', '08211520']
for STAID_work in STAID_work:
    plot_in = plot_data[plot_data['STAID'] == STAID_work]

    # NSE_in = results_ind.loc[
    #     (results_ind['STAID'] == STAID_work) &
    #     (results_ind['model'] == model_work) &
    #     (results_ind['train_val'] == 'train'),
    #     'NSE'
    #     ].values[0]

    NSE_in = NSE(
        plot_in.loc[
            plot_in['label'] == 'Pred', 'WY_ft'
            ], plot_in.loc[
                plot_in['label'] == 'Obs', 'WY_ft'
                ]
    )

    nse_out.extend([NSE_in])

    # plot
    p = (
        p9.ggplot(plot_in) +
        p9.geom_line(
            p9.aes(x = 'year', 
                    y = 'WY_ft', 
                    color = 'label')) +
        p9.theme_light() +
        p9.scales.scale_x_continuous(
            breaks = plot_in['year'].unique()) +
        p9.ggtitle(
            f'{model_work}: {STAID_work} (NSE = {NSE_in})') +
        p9.ylab('Water Yield [ft]')
        )


    print(p)

# %%
# calc NSE and KGE from all stations
df_nse_kge = NSE_KGE_Apply(
    pd.DataFrame({
        'y_obs': df_trainWY,
        'y_pred': y_pred,
        'ID_in': STAID['STAID']
        })
    )


# %%
# ECDF of NSE and KGE

# define which metric to use (e.g., NSE, or KGE)
metr_in = 'KGE'

cdf = ECDF(df_nse_kge[metr_in])

p = (
    p9.ggplot() +
    p9.geom_line(p9.aes(
        x = cdf.x,
        y = cdf.y)) +
        p9.xlim([-1, 1]) +
        p9.theme_light() +
        p9.ggtitle(
            f'eCDF of {metr_in} in {region}-train from {model_work}'
        )
    
    )


print(p)
# %%
