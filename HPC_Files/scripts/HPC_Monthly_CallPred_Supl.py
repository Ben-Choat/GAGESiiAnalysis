# Ben Choat 2023/06/07

# For an unknown reason, the strd_PCA_mlr model for Anth_0 -1 monthly
# data was either not
# executed or not saved on Summit HPC. Here I run the model.
# The same code is used to ensure consistency with other models.


# Ben Choat 7/28/2022

# Script:
# perform clustering. For the top two performing clustering results,
# call GAGESii_Monthly_Callable.py which goes through all regression models of interset 
# and asks for input where needed.


# %% import libraries and classes

# from GAGESii_Class import Clusterer
from HPC_Monthly_Callable import *
import pandas as pd
import os
import sys


# %%


# define clustering method used
# this variable is only used for keeping track fo results
# clust_meth_in =  sys.argv[1] # 'HLR' # 'AggEcoregion' #  'Class' #'None' # 'AggEcoregion'
clust_meth_in = 'Anth_0'

# list of possible AggEcoregions:
# 'All
# 'NorthEast', 'SECstPlain', 'SEPlains', 'EastHghlnds', 'CntlPlains',
#       'MxWdShld', 'WestMnts', 'WestPlains', 'WestXeric'
# set region_in = 'All' to include all data
# region_in = sys.argv[2] # 'MxWdShld' # 'Ref' # 'All' # '1' # 
region_in = '-1'


# define number of cores to be used for relevant processes
ncores = 2 # int(sys.argv[3]) # 1 # 


# %% load data

# main working directory
# NOTE: may need to change '8' below to a different value
# dir_Work = '/media/bchoat/2706253089/GAGES_Work' 
# dir_Work = os.getcwd()[0:(len(os.getcwd()) - 8)]
# dir_Work = '/scratch/bchoat'
# dir_Work = '/scratch/summit/bchoat@colostate.edu/GAGES'
dir_Work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work'


# water yield directory
# dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
dir_WY = f'{dir_Work}/data_work/USGS_discharge'

# DAYMET directory
# dir_DMT = 'D:/DataWorking/Daymet/train_val_test'
dir_DMT = f'{dir_Work}/data_work/Daymet'

# explantory var (and other data) directory
# dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned'
dir_expl = f'{dir_Work}/data_work/GAGESiiVariables'

# directory to write csv holding removed columns (due to high VIF)
# dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'
dir_VIF = f'{dir_Work}/data_out/monthly/VIF_Removed'


# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/Expl_train.csv',
    dtype = {'STAID': 'string'}
)
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/Expl_valnit.csv',
    dtype = {'STAID': 'string'}
)


# Water yield variables
# Monthly Water yield
# training
df_train_mnthWY = pd.read_csv(
    f'{dir_WY}/monthly/WY_Mnth_train.csv',
    dtype = {"site_no":"string"}
    )


# val_nit
df_valnit_mnthWY = pd.read_csv(
    f'{dir_WY}/monthly/WY_Mnth_valnit.csv',
    dtype = {"site_no":"string"}
    )


# DAYMET
# training
df_train_mnthDMT = pd.read_parquet(
    f'{dir_DMT}/monthly/DAYMET_monthly_train.parquet'
    )


# val_nit
df_valnit_mnthDMT = pd.read_parquet(
    f'{dir_DMT}/monthly/DAYMET_monthly_valnit.parquet'
    )
# rename DAYMET columns 
#colnames = [
#    'year',
#    'month',
#    'site_no',
#    'tmin',
#    'tmax',
#    'prcp',
#    'vp',
#    'swe'
#]
#df_train_mnthDMT.columns = colnames
#df_testin_mnthDMT.columns = colnames
#df_valnit_mnthDMT.columns = colnames
#
# ID vars (e.g., ecoregion)

# training ID
df_train_ID = pd.read_csv(f'{dir_expl}/ID_train.csv',
    dtype = {'STAID': 'string'})
# val_nit ID
df_valnit_ID = pd.read_csv(f'{dir_expl}/ID_valnit.csv',
    dtype = {'STAID': 'string'})

# convert all columns to strings
df_train_ID = df_train_ID.applymap(str)
df_valnit_ID = df_valnit_ID.applymap(str)

# %% 
# Define input variables for modeling

########
# subset data to catchment IDs that match the cluster or region being predicted
########


if region_in == 'All':
    cidtrain_in = df_train_ID
    cidvalnit_in = df_valnit_ID
else:
    cidtrain_in = df_train_ID[df_train_ID[clust_meth_in] == region_in]
    cidvalnit_in = df_valnit_ID[df_valnit_ID[clust_meth_in] == region_in]

# Water yield
train_resp_in = df_train_mnthWY.loc[
    df_train_mnthWY['site_no'].isin(cidtrain_in['STAID']), 'Mnth_WY_cm'
    ].reset_index(drop = True)

valnit_resp_in = df_valnit_mnthWY.loc[
    df_valnit_mnthWY['site_no'].isin(cidvalnit_in['STAID']), 'Mnth_WY_cm'
    ].reset_index(drop = True)


# explanatory variables
train_expl_in = df_train_expl[
    df_train_expl['STAID'].isin(cidtrain_in['STAID'])
    ].reset_index(drop = True)
valnit_expl_in = df_valnit_expl[
    df_valnit_expl['STAID'].isin(cidvalnit_in['STAID'])
    ].reset_index(drop = True)



# Add DAYMET to explanatory vars
train_expl_in = pd.merge(
    train_expl_in, df_train_mnthDMT, 
    left_on = ['STAID', 'year'], 
    right_on = ['site_no', 'year']
    ).drop('site_no', axis = 1)
valnit_expl_in = pd.merge(
    valnit_expl_in, df_valnit_mnthDMT, 
    left_on = ['STAID', 'year'], 
    right_on = ['site_no', 'year']
    ).drop('site_no', axis = 1)


# ID dataframes
train_ID_in = df_train_ID.loc[
    df_train_ID['STAID'].isin(cidtrain_in['STAID']), 'AggEcoregion'
    ].reset_index(drop = True)

valnit_ID_in = df_valnit_ID.loc[
    df_valnit_ID['STAID'].isin(cidvalnit_in['STAID']), 'AggEcoregion'
    ].reset_index(drop = True)

##########

# %%
#####
# Remove variables with a VIF > defined threshold (e.g., 10)
#####
 
X_in = train_expl_in.drop(
    ['STAID', 'year', 'month'], axis = 1
)

vif_th = 10 # 20

# calculate all vifs and store in dataframe
df_vif = VIF(X_in)


# initiate array to hold varibles that have been removed
df_removed = []


# find locations where vif = na due to zero variance in that column and remove
# that column
navif = np.where(df_vif.isna()) 
# append df_removed with VIFs of na which represent
# columns of zero variancedf_vif.loc[df_vif.isna()].index.values
df_removed.extend(df_vif.loc[df_vif.isna()].index.values, )
# # drop na vifs
# X_in.drop(df_removed, axis = 1, inplace = True)

# calculate new vifs excluding na values
df_vif = VIF(X_in.drop(df_removed, axis = 1))


while any(df_vif > vif_th):

    
    # sort df_vif by VIF
    temp_vif = df_vif.sort_values()

    # if largest VIF is for precipitation, then use next to max value
    if temp_vif.index[-1] == 'prcp':
        # append to removed list
        df_removed.append(temp_vif.index[-2])
        # drop from df_vif
        df_vif.drop(temp_vif.index[-2], inplace = True)

    else:
        # append to removed list
        df_removed.append(temp_vif.index[-1])
        # drop from df_vif
        df_vif.drop(temp_vif.index[-1], inplace = True)

    
    # calculate new vifs
    df_vif = VIF(X_in.drop(df_removed, axis = 1))

# redefine mean explanatory var df by dropping 'df_removed' vars and year colu
# drop columns from mean and timeseries explanatory vars
# training data
train_expl_in.drop(
    df_removed, axis = 1, inplace = True
)
# valnit data
valnit_expl_in.drop(
    df_removed, axis = 1, inplace = True
)

# print columns removed
print(df_removed)

# write csv with removed columns
import os
if not os.path.exists(dir_VIF):
    os.mkdir(dir_VIF)

df_vif_write = pd.DataFrame({
    'columns_Removed': df_removed
})

df_vif_write.to_csv(f'{dir_VIF}/VIF_ClsRemoved_MonthlyTS_{clust_meth_in}_{region_in}.csv')




# %% 
# Call function to perform modeling

regress_fun(df_train_expl = train_expl_in, # training data explanatory variables. Expects STAID to be a column
            df_valnit_expl = valnit_expl_in, # validation data explanatory variables using different catchments than were trained on
            train_resp = train_resp_in, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_mnthWY['Ann_WY_ft'])
            valnit_resp = valnit_resp_in, # validation data response variables using different catchments than were trained on
            train_ID = train_ID_in, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
            valnit_ID = valnit_ID_in, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
            clust_meth = clust_meth_in, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
            reg_in = region_in, # region label, i.e., 'NorthEast'
            models_in = [ # list of which models to run
                # 'regr_precip', 
                # 'strd_lasso', 
                # 'strd_mlr', 
                # 'strd_PCA_lasso', 
                'strd_PCA_mlr', 
                # 'XGBoost'
                ],               
            grid_in = { # dict with XGBoost parameters
                'n_estimators': [500, 750, 1000], # , 1000], # [100, 500, 750], # [100, 250, 500], # [10], # 
                'colsample_bytree': [0.7, 0.8], # [1], #
                'max_depth':  [3, 5, 7], #, 8], # [6], #
                'gamma': [0.1, 1, 2], # [0], # 
                'reg_lambda': [0.01, 0.1, 1], # [0], #
                'learning_rate': [0.01, 0.1, 0.2]
                },
            plot_out = False, # Boolean; outputs plots if True,
            train_id_var = train_expl_in['STAID'], # unique identifier for training catchments
            valnit_id_var = valnit_expl_in['STAID'], # unique identifier for valnit catchments
            dir_expl_in = f'{dir_Work}/data_out/monthly', # directory where to write results
            ncores_in = ncores # number of cores to send relevant jobs to
            )




