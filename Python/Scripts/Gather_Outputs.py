# BChoat 2022/09/20
# Script to gather outputs from different regions.
# It reads in variables from a given folder, combines them into a 
# single dataframe and writes them out as a single .pkl file

#%% 
# Load Libraries


# from genericpath import exists
import pandas as pd
# import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import xgboost as xgb

# %% 
# Define working directories and variables

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'AggEcoregion' # 'AggEcoregion', 'None', 'Class'

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# source directory
dir_source = ('D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'
                f'data_out/{time_scale}/{clust_meth}')

# out directory (where to place combined files)
dir_out = ('D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/'
                f'{time_scale}/combined')

# create out directory if it does not exist
if not os.path.exists(dir_out):
    os.mkdir(dir_out)


# %%
# Read in file, combine them, and write to pkl

# get list of files in source directory
# results from independent catchments
file_ind_list = glob.glob(f'{dir_source}/Results_*IndCatch*csv')

# summary results (first read in all files)
file_summary_list = glob.glob(f'{dir_source}/Results*csv')
# now drop the items that are also located in file_ind_list
file_summary_list = [x for x in file_summary_list if x not in file_ind_list]


# define output file path and name for independent catchment results
ind_out_name = f'{dir_out}/All_IndResults_{time_scale}.pkl'

# if combined file already exist, read it in,
# else, initiate empty DataFrame to hold independent results
if os.path.exists(ind_out_name):
    df_ind = pd.read_pickle(ind_out_name)
else:
    df_ind = pd.DataFrame()


# loop through independent files and combine them
for x in file_ind_list:
    temp_df = pd.read_csv(x,
        dtype = {'STAID': 'string'}
        )

    # concatenate to df_ind
    df_ind = pd.concat([df_ind, temp_df], axis = 0, ignore_index = True)


# define output file path and name for independent catchment results
sum_out_name = f'{dir_out}/All_SummaryResults_{time_scale}.pkl'

# if combined file already exist, read it in,
# else, initiate empty DataFrame to hold summary results
if os.path.exists(sum_out_name):
    df_summ = pd.read_pickle(sum_out_name)
else:
    df_summ = pd.DataFrame()


# loop through independent files and combine them
for x in file_summary_list:
    temp_df = pd.read_csv(x)
        
    # concatenate to df_ind
    df_summ = pd.concat([df_summ, temp_df], axis = 0, ignore_index = True)



# write concatenated results to pickle file
df_ind.to_pickle(
    ind_out_name
)

df_summ.to_pickle(
    sum_out_name
)

# %%

# read in files just written to check they are written as expected
ind_test = pd.read_pickle(
    f'{dir_out}/All_IndResults_{time_scale}.pkl'
)

summ_test = pd.read_pickle(
    f'{dir_out}/All_SummaryResults_{time_scale}.pkl'
)

print('Independent Results:')
print(ind_test)

print('summary results:')
print(summ_test)
# %%
