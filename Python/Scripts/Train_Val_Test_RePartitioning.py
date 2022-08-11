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
from Regression_PerformanceMetrics_Functs import VIF
# import plotnine as p9

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

# subset to years of interest 1998 - 2012
df_ts_all = df_ts_all[df_ts_all['year'].isin(np.arange(1998, 2013, 1))]

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

# drop lat, long, class, hydro_dist_indx, baseflow index, relief mean
# # and fragmentation columns
df_expl_all = df_expl_all.drop(
    ['LAT_GAGE', 
    'LNG_GAGE', 
    'CLASS', 
    'HYDRO_DISTURB_INDX', 
    'BFI_AVE', 
    'RRMEAN'],# 'FRAGUN_BASIN'], 
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

###################

# %% Investigate correlation between variables

# calculate mean variables

df_expl_all_mn = df_expl_all.groupby('STAID').mean().reset_index()

#####
# Remove variables with a pearsons r > defined threshold
#####

# # define threshold
# th_in = 0.95

# # using algorith found here:
# # https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
# # define working data
# Xtrain = df_expl_all_mn.drop(
#     ['STAID', 'year'], axis = 1
# )
# # Xtrain = Xtrain_dr

# # calculate correlation
# df_cor = Xtrain.corr(method = 'pearson').abs()

# # Select upper triangle of correlation matrix
# upper = df_cor.where(np.triu(np.ones(df_cor.shape), k = 1).astype(bool))

# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > th_in)]

# # Drop features 
# Xtrain_dr = Xtrain.drop(to_drop, axis=1) #, inplace=True)
##################

#####
# Remove variables with a VIF > defined threshold
#####

X_in = df_expl_all_mn.drop(
    ['STAID', 'year'], axis = 1
)

vif_th = 10 # 20

# calculate all vifs and store in dataframe
df_vif = VIF(X_in)

# initiate array to hold varibles that have been removed
df_removed = []

while any(df_vif > vif_th):
    # find max vifs and remove. If > 1 max vif, then remove only 
    # the first one
    maxvif = np.where(df_vif == df_vif.max())[0][0]

    # append inices of max vifs to removed dataframe
    df_removed.append(df_vif.index[maxvif])

    # drop max vif feature
    # df_vif.drop(df_vif.index[maxvif], inplace = True)
    
    # calculate new vifs
    df_vif = VIF(X_in.drop(df_removed, axis = 1))

# redefine mean explanatory var df by dropping 'df_removed' vars and year column
df_removed.append('year')

df_expl_all_mn = df_expl_all_mn.drop(
    df_removed, axis = 1
)


# %% Investigate partitinionings with various random seeds

# NOTE:
# changing how I am partitioning compared to original partitioning
# using only the 3214 catchments with continuous data from 1998-2012 (15 years)
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
        random_state = random_state,
        stratify = df_ID['AGGECOREGION']
    )

    X_vnit, X_tnit = train_test_split(
        X_other,
        train_size = 0.70,
        random_state = random_state,
        stratify = X_other['AGGECOREGION']
    )

    # calculate mean of explanatory variables through time for each catchment
    X_train_mn = df_expl_all_mn[(df_expl_all_mn['STAID'].isin(
        X_tr['STAID']))
        ].reset_index(drop = True)

    X_valnit_mn = df_expl_all_mn[(df_expl_all_mn['STAID'].isin(
        X_vnit['STAID']))
        ].reset_index(drop = True)

    X_testnit_mn = df_expl_all_mn[df_expl_all_mn['STAID'].isin(
        X_tnit['STAID'])
        ].reset_index(drop = True)

    #####
    # advaserial validation on training and valnit data
    #####

    # Instantiate classifier object
    xgbClassifier = xgb.XGBClassifier(
        objective = 'binary:logistic',
        n_estimators = 1000,
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


    cv_results = pd.concat(
                    [cv_results,
                    pd.DataFrame({
                    'random_seed': [random_state],
                    'cv_mean_valnit': [valnit_cv_mn],
                    'cv_stdev_valnit': [valnit_cv_stdev],
                    'cv_mean_testnit': [testnit_cv_mn],
                    'cv_stdev_testnit': [testnit_cv_stdev]
                    })]
                    , ignore_index = True)
    


cv_results


# %% define final split using best performing seed

random_state = 500 # rs

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
# ID files
# training
id_tr = X_tr.sort_values(by = 'STAID')
id_tr.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/ID_train.csv',
    index = False)
# valnit
id_vnit = X_vnit.sort_values(by = 'STAID')
id_vnit.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/ID_valnit.csv',
    index = False)
# testnit
id_tnit = X_tnit.sort_values(by = 'STAID')
id_tnit.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/ID_testnit.csv',
    index = False)

# Explanatory variables
# ID files
# training
X_train = df_expl_all[(df_expl_all['STAID'].isin(X_tr['STAID'])) &
    df_expl_all['year'].isin(np.arange(1998, 2008, 1))]
X_train.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/Expl_train.csv',
    index = False)
# valin
X_valin = df_expl_all[(df_expl_all['STAID'].isin(X_tr['STAID'])) &
    df_expl_all['year'].isin(np.arange(2008, 2012, 1))]
X_valin.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/Expl_train.csv',
    index = False)
#testing


# valnit
X_valnit = df_expl_all[df_expl_all['STAID'].isin(X_vnit['STAID'])]
X_valnit.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/Expl_valnit.csv',
    index = False)
# testnit
X_testnit = df_expl_all[df_expl_all['STAID'].isin(X_tnit['STAID'])]
X_testnit.to_csv(f'{dir_expl}/AllVars_VIF10_Filtered/Expl_testnit.csv',
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
