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
from sklearn.preprocessing import StandardScaler
from GAGESii_Class import *



# from xgboost import plot_tree # can't get to work on windows (try on linux)





# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'AggEcoregion' # 'Class' # 'None' # 'AggEcoregion', 'None', 

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
region =  'SECstPlain' # 'CntlPlains' # 'Non-ref' # 'All'
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly', 'daily'

# define which model you want to work with
model_work = 'strd_mlr' # ['XGBoost', 'strd_mlr', 'strd_lasso]

# if lasso, define alpha
alpha_in = 0.04

# which data to plot/work with
train_val = 'valnit'

# specifiy whether or not you want the explanatory vars to be standardized
strd_in = True #False #

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# # directory where to place outputs
dir_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/TEST_RESULTS'




# %%
# Load data
###########
# load train data (explanatory, water yield, ID)
df_trainexpl, df_trainWY, df_trainID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'train',
    clust_meth = clust_meth,
    region = region,
    standardize = strd_in# True # whether or not to standardize data
)

# load testin data (explanatory, water yield, ID)
df_testinexpl, df_testinWY, df_testinID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'testin',
    clust_meth = clust_meth,
    region = region,
    standardize = strd_in # whether or not to standardize data
)

# load valnit data (explanatory, water yield, ID)
df_valnitexpl, df_valnitWY, df_valnitID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'valnit',
    clust_meth = clust_meth,
    region = region,
    standardize = strd_in # whether or not to standardize data
)

# read in columns that were previously removed due to high VIF
file = glob.glob(
    f'{dir_work}/data_out/{time_scale}/VIF_Removed/*{clust_meth}_{region}.csv'
    )[0]
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
df_testinexpl.drop(vif_removed, axis = 1, inplace = True)
df_valnitexpl.drop(vif_removed, axis = 1, inplace = True)


# store staid's and date/year/month
# define columns to keep if present in STAID
col_keep = ['STAID', 'year', 'month', 'day', 'date']
STAIDtrain = df_trainexpl[df_trainexpl.columns.intersection(col_keep)]
STAIDtestin = df_testinexpl[df_testinexpl.columns.intersection(col_keep)]
STAIDvalnit = df_valnitexpl[df_valnitexpl.columns.intersection(col_keep)] 

# remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
# subset WY to version desired (ft)

if(time_scale == 'mean_annual'):
     
    df_trainexpl.drop('STAID', axis = 1, inplace = True)
    df_trainWY = df_trainWY['Ann_WY_ft']
    df_testinexpl.drop('STAID', axis = 1, inplace = True)
    df_testinWY = df_testinWY['Ann_WY_ft']
    df_valnitexpl.drop('STAID', axis = 1, inplace = True)
    df_valnitnWY = df_valnitWY['Ann_WY_ft']

if(time_scale == 'annual'):

    df_trainexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
    df_trainWY = df_trainWY['Ann_WY_ft']
    df_testinexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
    df_testinWY = df_testinWY['Ann_WY_ft']
    df_valnitexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
    df_valnitWY = df_valnitWY['Ann_WY_ft']

if(time_scale == 'monthly'):

    df_trainexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)
    df_trainWY = df_trainWY['Mnth_WY_ft']
    df_testinexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace = True)
    df_testinWY = df_testinWY['Mnth_WY_ft']
    df_valnitexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace = True)
    df_valnitWY = df_valnitWY['Mnth_WY_ft']

if(time_scale == 'daily'):

    STAIDtrain = df_trainexpl[['STAID', 'date']] 
    df_trainexpl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace =True)
    df_trainWY = df_trainWY['dlyWY_ft']
    df_testinexpl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace = True)
    df_testinWY = df_testinWY['dlyWY_ft']
    df_valnitexpl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace = True)
    df_valnitnWY = df_valnitWY['dlyWY_ft']


# %%
#  # read in names of VIF_df csvs to see what models were used
# ###########
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

# repeat for individual catchments
results_ind = pd.read_pickle(
    f'{dir_work}/data_out/{time_scale}/combined/All_IndResults_{time_scale}.pkl'
)
# subset to clust_meth and region of interest
results_ind = results_ind[
    (results_ind['clust_method'] == clust_meth) & (results_ind['region'] == region)
    ]

results_summ

# results_ind
# %%
# Define, fit model, and return coefficients or other relevant info
###########


#######
# LASSO
if model_work == 'strd_lasso':
    # define model and parameters
    model = Lasso(
        alpha = alpha_in,
        # max_iter = 1000,
        # selection = 'random',
        random_state = 100,
        max_iter = 10000,
        tol = 1e-8,
        fit_intercept = True
    )

    # fit model
    model.fit(df_trainexpl, df_trainWY)

    # create dataframe with regression coefficients and variables
    vars_temp = df_trainexpl.columns[np.abs(model.coef_) > 10e-20]
    vars_temp = pd.concat([pd.Series(vars_temp), pd.Series('intercept')])
    coef_temp = [x for x in model.coef_ if np.abs(x) > 10e-20]
    coef_temp = pd.concat([pd.Series(coef_temp), pd.Series(model.intercept_)])
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
    vars_temp = vars_keep.append(pd.Series('intercept'))
    coef_temp = np.append(model.coef_, model.intercept_)
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
    # try:
    model.load_model(
        f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
        f'/Models/XGBoost_{temp_time}_{clust_meth}_{region}_model.json'
        # f'/Models/xgbreg_meanannual_XGBoost_{clust_meth}_{region}_model.json'
        )
    # except:
    #     model.load_model(
    #         f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
    #         # f'/Models/XGBoost_{temp_time}_{clust_meth}_{region}_model.json'
    #         f'/Models/xgbreg_{temp_time}_XGBoost_{clust_meth}_{region}_model.json'
    #         )

    X_in = df_trainexpl



# %% 
# Plot
#############
if train_val == 'train':
    X_in = df_trainexpl[vars_keep] if model_work == 'strd_mlr' else df_trainexpl
    STAID_in = STAIDtrain
    df_WY_in = df_trainWY    
if train_val == 'testin':
    X_in = df_testinexpl[vars_keep] if model_work == 'strd_mlr' else df_testinexpl
    STAID_in = STAIDtestin
    df_WY_in = df_testinWY
if train_val == 'valnit':
    X_in = df_valnitexpl[vars_keep] if model_work == 'strd_mlr' else df_valnitexpl
    STAID_in = STAIDvalnit
    df_WY_in = df_valnitWY


# predict WY
y_pred = model.predict(X_in)

# define array to use as label of observed or predicted

lab_in = np.hstack([
    np.repeat('Obs', len(y_pred)),
    np.repeat('Pred', len(y_pred))
])

if time_scale == 'annual':
    # create dataframe for plotting
    plot_data = pd.DataFrame({
        'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
        'year': np.hstack([STAID_in['year'], STAID_in['year']]),
        'WY_ft': np.hstack([df_WY_in, y_pred]),
        'label': lab_in
    })

if time_scale == 'monthly':
    # create dataframe for plotting
    plot_data = pd.DataFrame({
        'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
        'year': np.hstack([STAID_in['year'], STAID_in['year']]),
        'month': np.hstack([STAID_in['month'], STAID_in['month']]),
        'WY_ft': np.hstack([df_WY_in, y_pred]),
        'label': lab_in
    })

if time_scale == 'daily':
    # create dataframe for plotting
    plot_data = pd.DataFrame({
        'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
        'year': np.hstack([STAID_in['year'], STAID_in['year']]),
        'month': np.hstack([STAID_in['month'], STAID_in['month']]),
        'date': np.hstack([STAID_in['date'], STAID_in['date']]),
        'WY_ft': np.hstack([df_WY_in, y_pred]),
        'label': lab_in
    })



##
# plots of individual catchment time-seires obs vs pred

# define empty array to append NSE to
nse_out = []

STAID_work = results_ind[
    (results_ind['train_val'] == train_val) & 
    (results_ind['model'] == model_work)
    ].sort_values(by = 'NSE', ascending = False)['STAID']

# STAID_in = STAID_in['STAID'].unique() # ['01305500', '02309848', '08211520']

# subset data to data of interest
# plot_in = plot_data
for STAID in STAID_work[0:20]:
    plot_in = plot_data[plot_data['STAID'] == STAID]
    plot_in['month_yr'] = plot_in['year'].astype(str) + \
        '-' + \
        plot_in['month'].astype(str)

    plot_in['month_count'] = np.concatenate(
        [np.arange(0, len(plot_in)/2, 1),
        np.arange(0, len(plot_in)/2, 1)])

    
    NSE_in = results_ind.loc[
        (results_ind['STAID'] == STAID) &
        (results_ind['model'] == model_work) &
        (results_ind['train_val'] == train_val),
        'NSE'
        ].values[0]

    NSE_in = NSE(
        plot_in.loc[
            plot_in['label'] == 'Pred', 'WY_ft'
            ], plot_in.loc[
                plot_in['label'] == 'Obs', 'WY_ft'
                ]
    )

    nse_out.extend([NSE_in])
    print(NSE_in)
    
    # plot
    p = (
        p9.ggplot(plot_in) +
        p9.geom_line(
            p9.aes(x = 'month_count', 
                    y = 'WY_ft', 
                    color = 'label')) +
        p9.theme_light() +
        # p9.scales.scale_x_continuous(
        #     breaks =  plot_in['month'].unique()) +
        p9.ggtitle(
            f'{model_work}: {STAID} (NSE = {NSE_in})') +
        p9.ylab('Water Yield [ft]')
        )


    print(p)


##
# all pred vs obs
plot_dataw = pd.DataFrame({
    'STAID': plot_data.loc[plot_data['label'] == 'Obs', 'STAID'],
    'year': plot_data.loc[plot_data['label'] == 'Obs', 'year'],
    'Obs': plot_data.loc[plot_data['label'] == 'Obs', 'WY_ft'],
    'Pred': plot_data.loc[plot_data['label'] == 'Pred', 'WY_ft'].reset_index(
        drop = True)
    })

p = (
    p9.ggplot(plot_dataw) + 
    p9.geom_point(p9.aes(
        x = 'Obs',
        y = 'Pred',
        color = plot_data.loc[plot_data['label'] == 'Obs', 'year']
            )
        ) +
        p9.geom_abline(color = 'red', slope = 1) 
    )

print(p)


# %%
# calc NSE and KGE from all stations
df_nse_kge = NSE_KGE_Apply(
    pd.DataFrame({
        'y_obs': df_WY_in,
        'y_pred': y_pred,
        'ID_in': STAID_in['STAID']
        })
    )


# %%
# ECDF of NSE and KGE

# define which metric to use (e.g., NSE, or KGE)
metr_in = 'NSE'

cdf = ECDF(df_nse_kge[metr_in])

# define lower limit for x (NSE)
xlow = -1
# remove vals from cdf lower than xlow
cdf.y = cdf.y[cdf.x > xlow]
cdf.x = cdf.x[cdf.x > xlow]


p = (
    p9.ggplot() +
    p9.geom_line(p9.aes(
        x = cdf.x,
        y = cdf.y)) +
        p9.xlim([-1, 1]) +
        p9.ylim([0,1]) +
        p9.theme_bw() +
        p9.ggtitle(
            f'eCDF of {metr_in} in {region}-train from {model_work}'
        )
    
    )


print(p)

print(f'median NSE: {np.median(df_nse_kge[metr_in])}')







# %% 
# manually applied regression with numpy

# read in coefs. and intercept from summit
coef_int = pd.read_csv(
    f'{dir_work}/data_out/{time_scale}/Models/{model_work}_{time_scale}'
    f'_{clust_meth}_{region}_model.csv'
    )

# if lasso, keep only non-zero features
if model_work == 'strd_lasso':
    coef_int = coef_int[np.abs(coef_int['coef']) > 1e-6]

# coef_int = df_coef.reset_index(drop = True)
# coef_int = df_coef.append(
#     pd.DataFrame({
#         'features': 'intercept',
#         'coef': model.intercept_},
#         index = [0]
#     )
# ).reset_index(
#     drop = True
# )

if 'DRAIN_SQKM_x' in coef_int['features'].values:
    coef_int['features'] = coef_int[
        'features'
        ].str.replace('DRAIN_SQKM_x', 'DRAIN_SQKM')


df_work = np.array(
    df_trainexpl[
        df_trainexpl.columns.intersection(coef_int['features'])
        ]
    )

# df_work = df_work.reshape(len(df_work), df_work.shape[1])

# define coefs var
beta_vars = np.array(coef_int['coef'].drop(
    (len(coef_int['coef']) - 1), axis = 0))

# define intercept
b0 = np.array(coef_int.loc[coef_int['features'] == 'intercept', 'coef'])

y = np.dot(df_work, beta_vars) + b0[0]
# np.linalg.inv(df_work.T.dot(df_expl)).dot(df_expl.T)dot()

df_out = pd.DataFrame({
    'STAID': STAIDtrain['STAID'],
    'y_pred': y
})

print(NSE(df_trainWY, y))


# calc NSE and KGE from all stations
df_nse_kge2 = NSE_KGE_Apply(
    pd.DataFrame({
        'y_obs': df_WY_in,
        'y_pred': y,
        'ID_in': STAID_in['STAID']
        })
    )

# %%


# reg = LinearRegression()

# X_in = np.array(df_trainexpl['prcp']).reshape(-1,1)

# # define standardizer
# stdsc = StandardScaler()
# # fit scaler
# scaler = stdsc.fit(X_in)
# X_in = scaler.transform(X_in)

# reg.fit(X_in, df_trainWY)

# print(reg.coef_)
# print(reg.intercept_)








# %%
# SOME TROUBLE SHOOTING STUFF BELOW HERE
#########################






# # %%
# df_trainexpl['prcp'].hist()
# plt.show()


# # %% 

# #  # Define features not to transform
# # not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
# #             'GEOL_REEDBUSH_DOM_quarternary', 
# #             'GEOL_REEDBUSH_DOM_sedimentary', 
# #             'GEOL_REEDBUSH_DOM_ultramafic', 
# #             'GEOL_REEDBUSH_DOM_volcanic',
# #             'year',
# #             'month',
# #             'day',
# #             'date',
# #             'STAID']

# # not_tr_in = df_trainexpl.columns.intersection(not_tr_in)


# # regr = Regressor(
# #     expl_vars = df_trainexpl, 
# #     resp_var = df_trainWY
# # )

# # regr.stand_norm(method = 'standardize',
# #     not_tr = not_tr_in)

# # test_tr = pd.DataFrame(regr.scaler_.transform(df_testinexpl))


# # %% trouble shoot lasso model reporting incorrect coeficients


# from GAGESii_Class import Regressor
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import HalvingGridSearchCV # for hyperparameter tuning
# # from sklearn.model_selection import RepeatedKFold # for repeated k-fold validation
# from sklearn.model_selection import GroupKFold # for splitting catchments without splitting individual catchments




# # %%
# cv = GroupKFold(
#     n_splits = 10
# )

# model = Lasso(
#     max_iter = 10000,
#     random_state = 100,
#     tol = 1e-8
# )

# grid = {'alpha': np.arange(0.01, 1.01, 0.1)}


# search = HalvingGridSearchCV(
#     model,
#     grid,
#     scoring = 'neg_mean_absolute_error',
#     refit = False,
#     cv = cv
# )

# results = search.fit(df_trainexpl, df_trainWY, groups = STAIDtrain['STAID'])

# cv_results = pd.DataFrame(results.cv_results_).sort_values(
#     by = 'rank_test_score'
#     )[['param_alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']]