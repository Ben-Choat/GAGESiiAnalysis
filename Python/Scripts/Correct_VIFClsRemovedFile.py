'''
BChoat 2023/04/15

For an unknown reason, the VIF_ClsRevmoed_AnnualTS_Anth_0_1.csv file ended up
with incorrect values in it.

To correct this, I read in the xgboost model for that scenario, because it 
included all variables remaining after removing variables with VIF > 10.

That is what this code is for.
'''

'''
BChoat 2022/10/05
Script to plot timeseries results
'''


# %% 
# Import Libraries
###########
import xgboost as xgb
import pandas as pd
import shutil
import os
from Load_Data import load_data_fun





# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'Anth_0' # 'Class' # 'None' # 'AggEcoregion', 'None', 

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
region =  '16' # 'CntlPlains' # 'Non-ref' # 'All'
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
time_scale = 'annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# define which model you want to work with
model_work = 'XGBoost' # ['XGBoost', 'strd_mlr', 'strd_lasso]

# which data to plot/work with
train_val = 'valnit' # 'train' or 'valnit

# specifiy whether or not you want the explanatory vars to be standardized
strd_in =  False #True #

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# # directory where to place outputs
# dir_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/TEST_RESULTS'

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


# load train data (explanatory, water yield, ID)
df_trainexpl, df_trainWY, df_trainID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'train',
    clust_meth = clust_meth,
    region = region,
    standardize = strd_in # whether or not to standardize data
)



# get difference between vars in xgboost model and in df_trainexpl columns
removed = df_trainexpl.columns.difference(model.get_booster().feature_names)
removed = removed.drop(['year', 'STAID'])
# create dataframe and write to csv

df_out = pd.DataFrame({'columns_Removed': removed})

file_work = f'{dir_work}/data_out/annual/VIF_' \
            f'Removed/VIF_ClsRemoved_AnnualTS_{clust_meth}_{region}.csv'

# copy back up of file_work with suffix '_error'
# define output file name
prefix_temp = file_work.split('.csv')[0]
file_out = f'{prefix_temp}_error.csv'

if os.path.exists(file_out):
    raise ValueError('Output file appears to have already been copied. \n' \
                     'This suggests that this script has aleady been executed. \n'
                     'Cancelling job.')
shutil.copyfile(file_work, file_out)

df_out.to_csv(
    file_work
    )