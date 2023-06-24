# Ben Choat, 2022/09/16

# script to read in csvs and combine them into a single 

# %%
# load libraries
import pandas as pd
import os
import glob

# %%
# define directory to read files from
dir_read = '/media/bchoat/2706253089/GAGES_Work/DataWorking/USGS_discharge/daily'
# define directory to write files to
dir_write = '/media/bchoat/2706253089/GAGES_Work/DataWorking/USGS_discharge/daily/pickle'

# if dir_write doesn't exist, then create it.
if not os.path.exists(dir_write):
    os.mkdir(dir_write)

# define variable specifying if working with train, testin, testnit, or valnit partition
part_work = 'valnit'

# %%
########
# This section was written for the DAYMET daily data. The USGS daily data is stored with a different
# directory structure so I am writing a different chunk of code for that,
###########

# read in training data, combine, and write to pickle

# read in list of files from dir_read
# csv_files = glob.glob(f'{dir_read}/*{part_work}.csv')

# # define empty dataframe to write csvs to
# df_write = pd.DataFrame()

# # loop through files, read them in, combine them
# for i in csv_files:
#     # read
#     temp_df = pd.read_csv(i, dtype = {'site_no': 'string'})
#     # add to df_write
#     df_write = pd.concat([df_write, temp_df], axis = 0, ignore_index = True)


# # write to pickle
# df_write.to_pickle(f'{dir_write}/DAYMET_daily_{part_work}.pkl')


# # test readingin pickle file just writen
# # test_pkl = pd.read_pickle(f'{dir_write}/DAYMET_daily_{part_work}.pkl')


# %%
#######
# This section was written for the USGS daily data
##########

# read in training data, combine, and write to pickle

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
df_write.to_pickle(f'{dir_write}/DAYMET_daily_{part_work}.pkl')


# test readingin pickle file just writen
# test_pkl = pd.read_pickle(f'{dir_write}/DAYMET_daily_{part_work}.pkl')


# %%
