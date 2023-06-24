# Ben Choat 7/28/2022

# Script:
# perform clustering. For the top two performing clustering results,
# call GAGESii_MeanAnnual_Callable.py which goes through all regression models of interset 
# and asks for input where needed.


# %% import libraries and classes

# from GAGESii_Class import Clusterer
from HPC_Annual_Callable import *
import pandas as pd
import os
import sys



# %% 
# Define variables specifying the clustering method used (e.g., AggEcoregion or UMAPHDBDCAN) and
# the region (e.g., EastHghlnds or 3)

# define clustering method used
# this variable is only used for keeping track fo results
# clust_meth_in =   'AggEcoregion' #'None' #
clust_meth_in =  sys.argv[1] # 'AggEcoregion' #
# clust_meth_in = 'Anth_0'

# list of possible AggEcoregions:
# 'All
# 'NorthEast', 'SECstPlain', 'SEPlains', 'EastHghlnds', 'CntlPlains',
#       'MxWdShld', 'WestMnts', 'WestPlains', 'WestXeric'
# set region_in = 'All' to include all data
# region_in =  'EastHghlnds' # 'All' #
region_in = sys.argv[2] # 'MxWdShld' #
# region_in = -1

# define number of cores to be used for relevant processes
ncores =  int(sys.argv[3]) # 4 #


# %% load data

# main working directory
# NOTE: may need to change '8' below to a different value
# dir_Work = '/media/bchoat/2706253089/GAGES_Work' 
# dir_Work = os.getcwd()[0:(len(os.getcwd()) - 8)]
# dir_Work = '/scratch/bchoat'
dir_Work = '/scratch/summit/bchoat@colostate.edu/GAGES'

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
dir_VIF = f'{dir_Work}/data_out/annual/VIF_Removed'


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
# Annual Water yield
# training
df_train_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    dtype = {"site_no":"string"}
    )


# val_nit
df_valnit_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    dtype = {"site_no":"string"}
    )


# DAYMET
# training
df_train_anDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_train.csv',
    dtype = {"site_no":"string"}
    )


# val_nit
df_valnit_anDMT = pd.read_csv(
    f'{dir_DMT}/annual/DAYMET_Annual_valnit.csv',
    dtype = {"site_no":"string"}
    )

# ID vars (e.g., ecoregion)

# training ID
df_train_ID = pd.read_csv(f'{dir_expl}/ID_train.csv',
    dtype = {'STAID': 'string'})
# val_nit ID
df_valnit_ID = pd.read_csv(f'{dir_expl}/ID_valnit.csv',
    dtype = {'STAID': 'string'})




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
train_resp_in = pd.merge(
    df_train_anWY, cidtrain_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_cm']
valnit_resp_in = pd.merge(
    df_valnit_anWY, cidvalnit_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_cm']

# explanatory variables
train_expl_in = pd.merge(df_train_expl, cidtrain_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
valnit_expl_in = pd.merge(df_valnit_expl, cidvalnit_in, on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)

# remove '_x' from drain_sqkm column name.
if 'DRAIN_SQKM_x' in train_expl_in.columns.values:
    train_expl_in.columns.str.replace('DRAIN_SQKM_x', 'DRAIN_SQKM')
if 'DRAIN_SQKM_x' in valnit_expl_in.columns.values:
    valnit_expl_in.columns.str.replace('DRAIN_SQKM_x', 'DRAIN_SQKM')



# Add DAYMET to explanatory vars
train_expl_in = pd.merge(
    train_expl_in, df_train_anDMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
    ).drop('site_no', axis = 1)
valnit_expl_in = pd.merge(
    valnit_expl_in, df_valnit_anDMT, left_on = ['STAID', 'year'], right_on = ['site_no', 'year']
    ).drop('site_no', axis = 1)

# ID dataframes
train_ID_in = pd.merge(
    df_train_ID, cidtrain_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
valnit_ID_in = pd.merge(
    df_valnit_ID, cidvalnit_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
##########

# %%
#####
# Remove variables with a VIF > defined threshold (e.g., 10)
#####

X_in = train_expl_in.drop(
    ['STAID', 'year'], axis = 1
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

df_vif_write.to_csv(f'{dir_VIF}/VIF_ClsRemoved_AnnualTS_{clust_meth_in}_{region_in}.csv')



# %% 
# Call function to perform modeling

regress_fun(df_train_expl = train_expl_in, # training data explanatory variables. Expects STAID to be a column
            df_valnit_expl = valnit_expl_in, # validation data explanatory variables using different catchments than were trained on
            train_resp = train_resp_in, # training data response variables NOTE: this should be a series, not a dataframe (e.g., df_train_anWY['Ann_WY_ft'])
            valnit_resp = valnit_resp_in, # validation data response variables using different catchments than were trained on
            train_ID = train_ID_in, # training data id's (e.g., clusters or ecoregions; df_train_ID['AggEcoregion'])
            valnit_ID = valnit_ID_in, # # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
            clust_meth = clust_meth_in, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
            reg_in = region_in, # region label, i.e., 'NorthEast'
            models_in = [ # list of which models to run
                'regr_precip', 
                'strd_lasso', 
                'strd_mlr', 
                'strd_PCA_lasso', 
                'strd_PCA_mlr', 
                'XGBoost'
                ],               
            grid_in = { # dict with XGBoost parameters
                'n_estimators': [500, 750, 1000], # 1000], # [100, 500, 750], # [100, 250, 500], # [10], # 
                'colsample_bytree': [0.7, 0.8], # [1], #
                'max_depth':  [3, 5, 7], #, 8], # [6], #
                'gamma': [0.1, 1, 2], # [0], # 
                'reg_lambda': [0.01, 0.1, 1], # [0], #
                'learning_rate': [0.01, 0.1, 0.2]
                },
            plot_out = False, # Boolean; outputs plots if True,
            train_id_var = train_expl_in['STAID'], # unique identifier for training catchments
            valnit_id_var = valnit_expl_in['STAID'], # unique identifier for valnit catchments
            dir_expl_in = f'{dir_Work}/data_out/annual', # directory where to write results
            ncores_in = ncores # number of cores to distribute relevant jobs to
            )


# %% Calculate NSE and KGE for each catchment


