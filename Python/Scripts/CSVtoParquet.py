# Ben Choat, 2022/09/16

# script to read in csvs and combine them into a single pickle file


# %%
# load libraries
import pandas as pd
import os
import glob

# %%
# define directory to read files from
dir_read = 'D:/DataWorking/Daymet/train_val_test/daily'
# dir_read = "D:/DataWorking/Daymet/Daymet_Daily"
# dir_read = "D:/DataWorking/Daymet/Daymet_Daily"
# dir_read =  'D:/DataWorking/USGS_discharge/train_val_test/daily'

# define directory to write files to
# dir_write = 'D:/DataWorking/USGS_discharge/train_val_test/daily/pickle'
# dir_write = '/media/bchoat/2706253089/GAGES_Work/DataWorking/Daymet/train_val_test/daily'
# dir_write = 'D:/GAGES_Work/DataWorking/Daymet/train_val_test/daily'
dir_write = 'D:/DataWorking/Daymet/train_val_test/daily'


# if dir_write doesn't exist, then create it.
if not os.path.exists(dir_write):
    os.mkdir(dir_write)


# %%
########
# This section was written for the DAYMET daily data. The USGS daily data is stored with a different
# directory structure so I am writing a different chunk of code for that,
###########

# define variable specifying if working with train, testin, testnit, or valnit partition
part_work = 'valnit'

# read in training data, combine, and write to parquet

# read in list of files from dir_read
csv_files = glob.glob(f'{dir_read}/*{part_work}.csv')
# csv_files = glob.glob(f'{dir_read}/*ref*')

# define empty dataframe to write csvs to
df_write = pd.DataFrame()

# loop through files, read them in, combine them
for i in csv_files:
    # read
    temp_df = pd.read_csv(i, dtype = {'site_no': 'string'})
    # add to df_write
    df_write = pd.concat([df_write, temp_df], axis = 0, ignore_index = True)


# # write to parquet
df_write.to_parquet(f'{dir_write}/DAYMET_daily_{part_work}.parquet')


# test readingin pickle file just writen
test_prq = pd.read_parquet(f'{dir_write}/DAYMET_daily_{part_work}.parquet')

print(test_prq)


# %%
#######
# This section was written for the USGS daily data
##########

# define directory to read files from
dir_read =  'D:/DataWorking/USGS_discharge/train_val_test/daily'

# define directory to write files to
dir_write = 'D:/DataWorking/USGS_discharge/train_val_test/daily/parquet'

# if dir_write doesn't exist, then create it.
if not os.path.exists(dir_write):
    os.mkdir(dir_write)
###########

# %%
# define variable specifying if working with train, testin, testnit, or valnit partition
part_work = 'train'

# read in training data, combine, and write to parquet

# read in list of files from dir_read
csv_files = glob.glob(f'{dir_read}/{part_work}/*.csv')

# define empty dataframe to write csvs to
df_write = pd.DataFrame()

# loop through files, read them in, combine them
for i in csv_files:
    # read
    temp_df = pd.read_csv(i, dtype = {'site_no': 'string'})
    # add to df_write
    df_write = pd.concat([df_write, temp_df], axis = 0, ignore_index = True)


# write to pickle
df_write.to_parquet(f'{dir_write}/WY_daily_{part_work}.parquet')


# test readingin pickle file just writen
test_pkl = pd.read_parquet(f'{dir_write}/WY_daily_{part_work}.parquet')

print(test_pkl)

# %%
