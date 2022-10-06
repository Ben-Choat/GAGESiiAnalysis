'''
BChoat, 2022/09/28
The script I wrote to download DAYMET data resulted in a couple of errors
that need to be cleaned up.
1. There are some catchments (~6) that are missing a day or two. Since these are
so few, I remove them here.
2. There are some duplicated rows which need to be removed
'''

# %%
# Import libraries and define directories/environmental variabels

import pandas as pd
import os
import shutil
import glob

dmt_dir = 'D:/DataWorking/Daymet'

# %% 
# move current DAYMET data to a back up directory

# make temp dir to hold current files
if not os.path.exists(f'{dmt_dir}/Daymet_Daily/del_later'):
    os.mkdir(f'{dmt_dir}/Daymet_Daily/del_later')

# get list of files
files_list = glob.glob(f'{dmt_dir}/Daymet_Daily/*ref*csv')

# define empty array to hold catchment IDs that are removed
id_removed = []

# calculate the number of dates that should be present for each catchment
# 15 years, 365 days/year
expected_count = 15 * 365

# loop through files, read them in and remove rows or catchments
for file in files_list:
    df_temp = pd.read_csv(file, dtype = {'site_no': 'string'})
    
    # move file to back up folder
    shutil.move(file, f'{dmt_dir}/Daymet_Daily/del_later/')
    
    # remove duplicates
    df_temp.drop_duplicates(inplace = True)
    
    # identify catchments without all dates present 
    temp_count = df_temp.groupby('site_no').count()['date']
    temp_drop = temp_count.index[temp_count < expected_count]
    
    # drop those catchments
    df_temp = df_temp[~df_temp['site_no'].isin(temp_drop)]

    # write the updated dataframe back to the original directory/file name
    df_temp.to_csv(file, index = False)

    # append dropped catchments to id_removed
    id_removed.extend(temp_drop)







# %%
# for some reason, monthly DAYMET data has different labels (e.g., 
# has 'retreived' attached to var name)
# fix that here

files_list = glob.glob(f'{dmt_dir}/train_val_test/monthly/*')

# loop through files and edit column names

for file in files_list:
    # read csv
    temp = pd.read_csv(file, dtype = {'site_no': 'string'})
    # rename columns
    temp.columns = temp.columns.str.replace('_retreived', '')
    # write back to file
    temp.to_csv(file, index = False)
