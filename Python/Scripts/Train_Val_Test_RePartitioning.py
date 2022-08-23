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
import os # for checking if directories exist and/or making directories
import shutil # for copying files
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
).drop(['CLASS', 'AGGECOREGION'], axis = 1)

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

# #####
# # Remove variables with a VIF > defined threshold
# #####

# X_in = df_expl_all_mn.drop(
#     ['STAID', 'year'], axis = 1
# )

# vif_th = 10 # 20

# # calculate all vifs and store in dataframe
# df_vif = VIF(X_in)

# # initiate array to hold varibles that have been removed
# df_removed = []

# while any(df_vif > vif_th):
#     # find max vifs and remove. If > 1 max vif, then remove only 
#     # the first one
#     maxvif = np.where(df_vif == df_vif.max())[0][0]

#     # append inices of max vifs to removed dataframe
#     df_removed.append(df_vif.index[maxvif])

#     # drop max vif feature
#     # df_vif.drop(df_vif.index[maxvif], inplace = True)
    
#     # calculate new vifs
#     df_vif = VIF(X_in.drop(df_removed, axis = 1))

# # redefine mean explanatory var df by dropping 'df_removed' vars and year column
# # drop columns from mean and timeseries explanatory vars
# df_expl_all_mn = df_expl_all_mn.drop(
#     df_removed + ['year'], axis = 1
# )

# df_expl_all = df_expl_all.drop(
#     df_removed, axis = 1
# )


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
        stratify = df_ID['AggEcoregion']
    )

    X_vnit, X_tnit = train_test_split(
        X_other,
        train_size = 0.70,
        random_state = random_state,
        stratify = X_other['AggEcoregion']
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

# set seed
random_state = 200

X_tr, X_other = train_test_split(
    df_ID,
    train_size = 0.60,
    random_state = random_state,
    stratify = df_ID['AggEcoregion']
)

X_vnit, X_tnit = train_test_split(
    X_other,
    train_size = 0.70,
    random_state = random_state,
    stratify = X_other['AggEcoregion']
)

# Write partitions to csvs
# ID files
# training
id_tr = X_tr.sort_values(by = 'STAID')
id_tr.to_csv(f'{dir_expl}/AllVars_Partitioned/ID_train.csv',
    index = False)
# valnit
id_vnit = X_vnit.sort_values(by = 'STAID')
id_vnit.to_csv(f'{dir_expl}/AllVars_Partitioned/ID_valnit.csv',
    index = False)
# testnit
id_tnit = X_tnit.sort_values(by = 'STAID')
id_tnit.to_csv(f'{dir_expl}/AllVars_Partitioned/ID_testnit.csv',
    index = False)

# Explanatory variables
# ID files
# training
X_train = df_expl_all[(df_expl_all['STAID'].isin(X_tr['STAID'])) &
    df_expl_all['year'].isin(np.arange(1998, 2008, 1))]
X_train.to_csv(f'{dir_expl}/AllVars_Partitioned/Expl_train.csv',
    index = False)
# testin
X_testin = df_expl_all[(df_expl_all['STAID'].isin(X_tr['STAID'])) &
    df_expl_all['year'].isin(np.arange(2008, 2013, 1))]
X_testin.to_csv(f'{dir_expl}/AllVars_Partitioned/Expl_testin.csv',
    index = False)


# valnit
X_valnit = df_expl_all[df_expl_all['STAID'].isin(X_vnit['STAID'])]
X_valnit.to_csv(f'{dir_expl}/AllVars_Partitioned/Expl_valnit.csv',
    index = False)
# testnit
X_testnit = df_expl_all[df_expl_all['STAID'].isin(X_tnit['STAID'])]
X_testnit.to_csv(f'{dir_expl}/AllVars_Partitioned/Expl_testnit.csv',
    index = False)


# %%
##########################
# Annual Water yield
##########################
# NOTE: this chunk of code partitions data into training, testing, valnit, and 
# testnit partitions. It also adds a column of water yield with 'ft' as the unit

# define conversion factor for going from acre-ft to ft using sqkm area value
# 247.105 acres in 1 km2
vol_conv = 1/247.105

# read in annual water yield vars and partition them
df_wy_annual = pd.read_csv(
    'D:/DataWorking/USGS_discharge/annual_WY/Annual_WY_1976_2013.csv',
    dtype = {'site_no': 'string'}
)

# subset to years '98 - '12
df_wy_annual = df_wy_annual[
    df_wy_annual['yr'].isin(np.arange(1998, 2013, 1))
]

# training
df_wy_annual_train = pd.merge(
    id_tr,
    df_wy_annual,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_annual_train['Ann_WY_ft'] = df_wy_annual_train['Ann_WY_acft']/df_wy_annual_train['DRAIN_SQKM'] * vol_conv

df_wy_annual_train = df_wy_annual_train.drop(id_tr.columns, axis = 1)

# split to train and testing
# testin
df_wy_annual_testin = df_wy_annual_train[
    df_wy_annual_train['yr'].isin(np.arange(2008, 2013))
    ]
# train
df_wy_annual_train = df_wy_annual_train[
    df_wy_annual_train['yr'].isin(np.arange(1998, 2008))
    ]

# valnit
df_wy_annual_valnit = pd.merge(
    id_vnit,
    df_wy_annual,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_annual_valnit['Ann_WY_ft'] = df_wy_annual_valnit['Ann_WY_acft']/df_wy_annual_valnit['DRAIN_SQKM'] * vol_conv

df_wy_annual_valnit = df_wy_annual_valnit.drop(id_tr.columns, axis = 1)

# testnit
df_wy_annual_testnit = pd.merge(
    id_tnit,
    df_wy_annual,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_annual_testnit['Ann_WY_ft'] = df_wy_annual_testnit['Ann_WY_acft']/df_wy_annual_testnit['DRAIN_SQKM'] * vol_conv

df_wy_annual_testnit = df_wy_annual_testnit.drop(id_tr.columns, axis = 1)


# Write to csv's
# see if directory exists - if not - make it
if not os.path.exists(f'{dir_WY}/annual'):
    os.mkdir(f'{dir_WY}/annual')

# training
df_wy_annual_train.to_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    index = False
)

# testin
df_wy_annual_testin.to_csv(
    f'{dir_WY}/annual/WY_Ann_testin.csv',
    index = False
)

# valnit
df_wy_annual_valnit.to_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    index = False
)

# testnit
df_wy_annual_testnit.to_csv(
    f'{dir_WY}/annual/WY_Ann_testnit.csv',
    index = False
)


#############


# %%
##########################
# Monthly Water yield
##########################
# NOTE: this chunk of code partitions data into training, testing, valnit, and 
# testnit partitions. It also adds a column of water yield with 'ft' as the unit

# read in annual water yield vars and partition them
df_wy_monthly = pd.read_csv(
    'D:/DataWorking/USGS_discharge/monthly_WY/Monthly_WY_1976_2013.csv',
    dtype = {'site_no': 'string'}
)

# subset to years '98 - '12
df_wy_monthly = df_wy_monthly[
    df_wy_monthly['yr'].isin(np.arange(1998, 2013, 1))
]

# training
df_wy_monthly_train = pd.merge(
    id_tr,
    df_wy_monthly,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_monthly_train['Mnth_WY_ft'] = df_wy_monthly_train['Mnth_WY_acft']/df_wy_monthly_train['DRAIN_SQKM'] * vol_conv

df_wy_monthly_train = df_wy_monthly_train.drop(id_tr.columns, axis = 1)

# split to train and testing
# testin
df_wy_monthly_testin = df_wy_monthly_train[
    df_wy_monthly_train['yr'].isin(np.arange(2008, 2013))
    ]
# train
df_wy_monthly_train = df_wy_monthly_train[
    df_wy_monthly_train['yr'].isin(np.arange(1998, 2008))
    ]

# valnit
df_wy_monthly_valnit = pd.merge(
    id_vnit,
    df_wy_monthly,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_monthly_valnit['Mnth_WY_ft'] = df_wy_monthly_valnit['Mnth_WY_acft']/df_wy_monthly_valnit['DRAIN_SQKM'] * vol_conv

df_wy_monthly_valnit = df_wy_monthly_valnit.drop(id_tr.columns, axis = 1)

# testnit
df_wy_monthly_testnit = pd.merge(
    id_tnit,
    df_wy_monthly,
    left_on = 'STAID',
    right_on = 'site_no'
)

df_wy_monthly_testnit['Mnth_WY_ft'] = df_wy_monthly_testnit['Mnth_WY_acft']/df_wy_monthly_testnit['DRAIN_SQKM'] * vol_conv

df_wy_monthly_testnit = df_wy_monthly_testnit.drop(id_tr.columns, axis = 1)


# Write to csv's
# see if directory exists - if not - make it
if not os.path.exists(f'{dir_WY}/monthly'):
    os.mkdir(f'{dir_WY}/monthly')

# training
df_wy_monthly_train.to_csv(
    f'{dir_WY}/monthly/WY_Mnth_train.csv',
    index = False
)

# testin
df_wy_monthly_testin.to_csv(
    f'{dir_WY}/monthly/WY_Mnth_testin.csv',
    index = False
)

# valnit
df_wy_monthly_valnit.to_csv(
    f'{dir_WY}/monthly/WY_Mnth_valnit.csv',
    index = False
)

# testnit
df_wy_monthly_testnit.to_csv(
    f'{dir_WY}/monthly/WY_Mnth_testnit.csv',
    index = False
)

######################

# %%
##########################
# Daily Water yield
##########################
# NOTE: this chunk of code partitions data into training, testing, valnit, and 
# testnit partitions. 

#  make daily_WY folder if it does not exist
# see if directory exists - if not - make it
if not os.path.exists(f'{dir_WY}/daily'):
    os.mkdir(f'{dir_WY}/daily')
if not os.path.exists(f'{dir_WY}/daily/train'):
    os.mkdir(f'{dir_WY}/daily/train')
if not os.path.exists(f'{dir_WY}/daily/testin'):
    os.mkdir(f'{dir_WY}/daily/testin')
if not os.path.exists(f'{dir_WY}/daily/valnit'):
    os.mkdir(f'{dir_WY}/daily/valnit')    
if not os.path.exists(f'{dir_WY}/daily/testnit'):
    os.mkdir(f'{dir_WY}/daily/testnit')

#define conversion factor for converting from sqkm to ft2
# 1 km2 = (1000 m)^2 * (3.28084 ft)^2 = 10,763,911.1056 ft2
conv_fct = ((1000 ** 2) * (3.28084 ** 2))


# Training and testin

# loop through training ID's and copy years '98-'07 from training catchments
# to training folder and years '08-'12 to testin folder.

for gg_id in id_tr['STAID']:
    
    # read in csv
    temp_df = pd.read_csv(
        f'D:/DataWorking/USGS_discharge/daily_WY/Daily_WY_1976_2013_{gg_id}.csv',
        dtype = {'site_no': 'string'}
    )

    # keep only columns of interestdrop columns not to keep 
    temp_df = temp_df[['site_no', 'yr', 'mnth', 'day', 'wtryr', 'dlyWY_cfd']]

    
    # calculate daily water yield as ft using cfd column and area from id vars
    # round to 6 sig figs
    temp_df['dlyWY_ft'] = np.round(temp_df['dlyWY_cfd']/(
        float(id_tr.loc[id_tr['STAID'] == gg_id, 'DRAIN_SQKM']) * conv_fct
        ), 6)

    # write training years from temp_df to daily/training directory
    temp_df[temp_df['yr'].isin(np.arange(1998, 2008, 1))].to_csv(
        f'{dir_WY}/daily/train/WY_daily_train_{gg_id}.csv',
        index = False
    )

    # write testnit years from temp_df to daily/testnit directory
    temp_df[temp_df['yr'].isin(np.arange(2008, 2013, 1))].to_csv(
        f'{dir_WY}/daily/testin/WY_daily_testin_{gg_id}.csv',
        index = False
    )


# valnit

for gg_id in id_vnit['STAID']:
    
    # read in csv
    temp_df = pd.read_csv(
        f'D:/DataWorking/USGS_discharge/daily_WY/Daily_WY_1976_2013_{gg_id}.csv',
        dtype = {'site_no': 'string'}
    )

    # keep only columns of interestdrop columns not to keep 
    temp_df = temp_df[['site_no', 'yr', 'mnth', 'day', 'wtryr', 'dlyWY_cfd']]
    
    # calculate daily water yield as ft using cfd column and area from id vars
    # round to 6 sig figs
    temp_df['dlyWY_ft'] = np.round(temp_df['dlyWY_cfd']/(
        float(id_vnit.loc[id_vnit['STAID'] == gg_id, 'DRAIN_SQKM']) * conv_fct
        ), 6)

    # write training years from temp_df to daily/training directory
    temp_df[temp_df['yr'].isin(np.arange(1998, 2013, 1))].to_csv(
        f'{dir_WY}/daily/valnit/WY_daily_valnit_{gg_id}.csv',
        index = False
    )

# testnit

for gg_id in id_tnit['STAID']:
    
    # read in csv
    temp_df = pd.read_csv(
        f'D:/DataWorking/USGS_discharge/daily_WY/Daily_WY_1976_2013_{gg_id}.csv',
        dtype = {'site_no': 'string'}
    )

    # keep only columns of interestdrop columns not to keep 
    temp_df = temp_df[['site_no', 'yr', 'mnth', 'day', 'wtryr', 'dlyWY_cfd']]
    
    # calculate daily water yield as ft using cfd column and area from id vars
    # round to 6 sig figs
    temp_df['dlyWY_ft'] = np.round(temp_df['dlyWY_cfd']/(
        float(id_tnit.loc[id_tnit['STAID'] == gg_id, 'DRAIN_SQKM']) * conv_fct
        ), 6)

    # write training years from temp_df to daily/training directory
    temp_df[temp_df['yr'].isin(np.arange(1998, 2013, 1))].to_csv(
        f'{dir_WY}/daily/testnit/WY_daily_testnit_{gg_id}.csv',
        index = False
    )

##################


# %%
##########################
# Annual DAYMET data
##########################

dir_DMT = 'D:/DataWorking/Daymet/train_val_test'

# make annual_DAYMET folder if it does not exist
# see if directory exists - if not - make it
if not os.path.exists(f'{dir_DMT}/annual'):
    os.mkdir(f'{dir_DMT}/annual')

# read in annual daymet data
df_dmt_annual = pd.read_csv(
    'D:/DataWorking/Daymet/Daymet_Annual.csv',
    dtype = {'site_no': 'string'}
)

df_dmt_annual['year'] = df_dmt_annual['year'].astype(int)

# write annual water yield
# training
temp_df = df_dmt_annual[
    (df_dmt_annual['site_no'].isin(id_tr['STAID'])) &
    (df_dmt_annual['year'].isin(np.arange(1998, 2008, 1)))
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
    index = False
)

# testin
temp_df = df_dmt_annual[
    (df_dmt_annual['site_no'].isin(id_tr['STAID'])) &
    (df_dmt_annual['year'].isin(np.arange(2008, 2013, 1)))
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_testin.csv',
    index = False
)

# valnit
temp_df = df_dmt_annual[
    df_dmt_annual['site_no'].isin(id_vnit['STAID'])
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_valnit.csv',
    index = False
)

# testnit
temp_df = df_dmt_annual[
    df_dmt_annual['site_no'].isin(id_tnit['STAID'])
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_testnit.csv',
    index = False
)

# %%
##########################
# Monthly DAYMET data
##########################

# make monthly_DAYMET folder if it does not exist
# see if directory exists - if not - make it
if not os.path.exists(f'{dir_DMT}/monthly'):
    os.mkdir(f'{dir_DMT}/monthly')

# read in monthly daymet data
df_dmt_monthly = pd.read_csv(
    'D:/DataWorking/Daymet/Daymet_monthly.csv',
    dtype = {'site_no': 'string'}
)

df_dmt_monthly['year'] = df_dmt_monthly['year'].astype(int)

# write monthly water yield
# training
temp_df = df_dmt_monthly[
    (df_dmt_monthly['site_no'].isin(id_tr['STAID'])) &
    (df_dmt_monthly['year'].isin(np.arange(1998, 2008, 1)))
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_train.csv',
    index = False
)

# testin
temp_df = df_dmt_monthly[
    (df_dmt_monthly['site_no'].isin(id_tr['STAID'])) &
    (df_dmt_monthly['year'].isin(np.arange(2008, 2013, 1)))
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_testin.csv',
    index = False
)

# valnit
temp_df = df_dmt_monthly[
    df_dmt_monthly['site_no'].isin(id_vnit['STAID'])
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_valnit.csv',
    index = False
)

# testnit
temp_df = df_dmt_monthly[
    df_dmt_monthly['site_no'].isin(id_tnit['STAID'])
]

# write to csv
temp_df.to_csv(
    f'{dir_DMT}/monthly/DAYMET_monthly_testnit.csv',
    index = False
)
# %%
