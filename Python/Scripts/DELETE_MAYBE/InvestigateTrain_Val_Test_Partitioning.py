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
from sklearn.model_selection import train_test_split
import plotnine as p9

# %% Load Data

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# read in all static vars
df_static_all = pd.read_csv(
    f'{dir_expl}/GAGES_Static_Filtered.csv',
    dtype = {'STAID': 'string'}
)

# read in time-series vars
df_ts_all = pd.read_csv(
    f'{dir_expl}/gagesii_ts/GAGESts_InterpYrs_Wide.csv',
    dtype = {'STAID': 'string'}
)

# merge static and ts vars
df_expl_all = pd.merge(
    df_ts_all,
    df_static_all,
    on = 'STAID'
)

# remove 'anorthositic' and 'intermediate' GEOL_REEDUBSH_DOM types
# there are only 4 anorthositic and 1 intermediate
df_expl_all = df_expl_all[-df_expl_all['GEOL_REEDBUSH_DOM'].isin(
                ['anorthositic', 'intermediate']
                )]

# drop 'GEOL_HUNT_DOM_CODE
df_expl_all = df_expl_all.drop('GEOL_HUNT_DOM_CODE', axis = 1)

# drop NWALT columns
df_expl_all = df_expl_all.iloc[
    :, ~df_expl_all.columns.str.contains('NWALT')
]

# drop lat, long, class, hydro_dist_indx, and baseflow index columns
df_expl_all = df_expl_all.drop(
    ['LAT_GAGE', 'LNG_GAGE', 'CLASS', 'HYDRO_DISTURB_INDX', 'BFI_AVE'], 
    axis = 1
)

# one hot encode GEOL_REDDBUSH_DOM columns
df_expl_all = pd.get_dummies(df_expl_all)

# # Read in training ID vars and rewrite as ID_all_avail98_12.csv
# df_ID_all_avail98_12 = pd.read_csv(
#     f'{dir_expl}/ID_train.csv',
#     dtype = {'STAID': 'string'}).drop(columns = 'Unnamed: 0')

# df_ID_all_avail98_12.to_csv(
#     f'{dir_expl}/ID_all_avail98_12.csv',
#     index = False
# )

# read in ID file (e.g., holds aggecoregions) with gauges available from '98-'12
df_ID = pd.read_csv(
    f'{dir_expl}/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
)



##################

# %% Investigate partitinionings with various random seeds

# NOTE:
# changing how I am partitioning compared to original partitioning
# using only the 3211 catchments with continuous data from 1998-2012 (15 years)
# Of the ~ 3211 60% goes to training
# of the remaining 40%, ~70% goes to valnit and ~30% goes to testnit

# define empty dataframe to append results to
cv_results = pd.DataFrame({
    'random_seed': [],
    'cv_mean_valnit': [],
    'cv_stdev_valnit': [],
    'cv_mean_testnit': [],
    'cv_stdev_testnit': []
})

# first split train and validation data
for rs in np.arange(100, 1100, 100):

    random_state = rs

    X_tr, X_other = train_test_split(
        df_ID,
        train_size = 0.60,
        random_state = random_state
    )

    X_vnit, X_tnit = train_test_split(
        X_other,
        train_size = 0.70,
        random_state = random_state
    )

    # calculate mean of explanatory variables through time for each catchment
    X_train_mn = df_expl_all[(df_expl_all['STAID'].isin(
        X_tr['STAID'])) & 
        (df_expl_all['year'].isin(
            np.arange(1998, 2008, 1)
        ))].groupby('STAID').mean().drop('year', axis = 1).reset_index()

    X_valnit_mn = df_expl_all[(df_expl_all['STAID'].isin(
        X_vnit['STAID'])) & (df_expl_all['year'].isin(
            np.arange(1998, 2008, 1)
        ))].groupby('STAID').mean().drop('year', axis = 1).reset_index()

    X_testnit_mn = df_expl_all[df_expl_all['STAID'].isin(
        X_tnit['STAID']) & (df_expl_all['year'].isin(
            np.arange(1998, 2008, 1)
        ))].groupby('STAID').mean().drop('year', axis = 1).reset_index()

    #####
    # advaserial validation on training and valnit data
    #####

    # Instantiate classifier object
    xgbClassifier = xgb.XGBClassifier(
        objective = 'binary:logistic',
        n_estimators = 100,
        learning_rate = 0.1,
        max_depth = 3,
        random_state = 100,
        use_label_encoder = False
    )

    ######

    # define input parameters
    # training data
    X_train = X_train_mn.drop(columns = 'STAID')
    X_train['AV_label'] = 0

    # valnit data
    X_valnit = X_valnit_mn.drop(columns = 'STAID')
    X_valnit['AV_label'] = 1

    # combine into one dataset
    all_data = pd.concat([X_train, X_valnit], axis = 0, ignore_index = True)

    # shuffle
    all_data_shuffled = all_data.sample(frac = 1)

    # define X and y for use in classification
    X_all = all_data_shuffled.drop(columns = 'AV_label')
    y_all = all_data_shuffled['AV_label']

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
    valnit_cv_mn = np.round(classify_cv['test_score'].mean(), 3)
    valnit_cv_stdev = np.round(np.std(classify_cv['test_score']), 3)

    #######
    # repeat for testnit

    # valnit data
    X_testnit = X_testnit_mn.drop(columns = 'STAID')
    X_testnit['AV_label'] = 1

    # combine into one dataset
    all_data = pd.concat([X_train, X_testnit], axis = 0, ignore_index = True)

    # shuffle
    all_data_shuffled = all_data.sample(frac = 1)

    # define X and y for use in classification
    X_all = all_data_shuffled.drop(columns = 'AV_label')
    y_all = all_data_shuffled['AV_label']

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
    testnit_cv_mn = np.round(classify_cv['test_score'].mean(), 3)
    testnit_cv_stdev = np.round(np.std(classify_cv['test_score']), 3)


    cv_results = cv_results.append(
                    pd.DataFrame({
                    'random_seed': [random_state],
                    'cv_mean_valnit': [valnit_cv_mn],
                    'cv_stdev_valnit': [valnit_cv_stdev],
                    'cv_mean_testnit': [testnit_cv_mn],
                    'cv_stdev_testnit': [testnit_cv_stdev]
                }), ignore_index = True)
    


cv_results

# %% perminantly parition data
# 900 performed the best

random_state = 900 # rs

X_tr, X_other = train_test_split(
    df_ID,
    train_size = 0.60,
    random_state = random_state
)

X_vnit, X_tnit = train_test_split(
    X_other,
    train_size = 0.70,
    random_state = random_state
)

# Write partitions to csvs

# training
X_tr.to_csv(f'{dir_expl}/ID_train.csv',
    index = False)
# valnit
X_vnit.to_csv(f'{dir_expl}/ID_valnit.csv',
    index = False)
# testnit
X_tnit.to_csv(f'{dir_expl}/ID_testnit.csv',
    index = False)






# # %% If desired, investigate the important parameters allowing prediction
# # of valnit, and testnit
# # investigate importance features
# # fit classifier
# xgbClassifier.fit(X_all, y_all)

# # return roc auc score for prediction of training vs valnit
# ts_pred = np.round(roc_auc_score(y_all, xgbClassifier.predict(X_all)), 3)


# # xgb.plot_importance(xgbClassifier, max_num_features = 10)

# perm_imp = permutation_importance(
#     xgbClassifier,
#     X = X_all, 
#     y = y_all, 
#     scoring = 'roc_auc', #scores,
#     n_repeats = 10, # 30
#     n_jobs = -1,
#     random_state = 100)


# df_perm_imp = pd.DataFrame({
#     'Features': X_all.columns,
#     'AUC_imp': perm_imp['importances_mean']
# }).sort_values(by = 'AUC_imp', 
#                 ascending = False, 
#                 ignore_index = True)



# print(f'Mean Test ROC-AUC: {ts_cv}')
# # auc-roc of 0.79 suggests values are not from the same distribution
# # a value of 1 means train and valnit can be perfectly identified (this is bad)
# # a value of 0.5 is desirable and suggests both datasets are from the same
# # distribution
# print(f'ROC-AUC from fitted model: {ts_pred}')
# df_perm_imp.head(10)
# # df_perm_imp.tail(25)















# Explanatory variables
# Annual Water yield
# training
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
# df_valin_anWY = pd.read_csv(
#     f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_val_in.csv',
#     dtype = {"site_no":"string"}
#     )
# # drop stations not in explantory vars    
# df_valin_anWY = df_valin_anWY[
#     df_valin_anWY['site_no'].isin(df_valin_expl['STAID'])
#     ].reset_index(drop = True)
# # create annual water yield in ft
# df_valin_anWY['Ann_WY_ft'] = df_valin_anWY['Ann_WY_ft3']/(
#     df_valin_expl['DRAIN_SQKM']*(3280.84**2)
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
# # subset valint expl and response vars to common years of interest
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
# df_valin_mnanWY = df_valin_anWY.groupby(
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
# df_valin_mnexpl = df_valin_expl.groupby(
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
# df_valin_ID = df_train_ID
# # val_nit ID
# df_valnit_ID = df_ID[df_ID.STAID.isin(df_valnit_expl.STAID)].reset_index(drop = True)

# del(df_train_anWY, df_train_expl, df_valin_anWY, df_valin_expl, df_valnit_anWY, df_valnit_expl)

# %%
