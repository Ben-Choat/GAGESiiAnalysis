'''
BChoat 11/17/2022

Script to add a years worth of antecedent DAYMET 
weather explanatory variables.

First for annual data, then monthly, then daily.

Data is also cleaned up a bit along the way.
'''


# %%
# Import libraries
##############

import pandas as pd # data manipulate
# import shutil # for copying new files to HPC working directory
import numpy as np
import glob
import os
import dask.dataframe as dd


# %%
# Load ID info for study catchments
###########

df_ID = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
    )



# %% annual data
############
# load data
###########


# DAYMET
df_DMT = pd.read_csv(
    'D:/DataWorking/Daymet/Daymet_Annual.csv',
    dtype = {'site_no': 'string'}
    )

# subset to working catchments
df_DMT = df_DMT[
    df_DMT['site_no'].isin(df_ID['STAID'])
    ]

# convert year to integers
df_DMT['year'] = df_DMT['year'].astype(int)


##########
# calculate previous 1 year of daymet vars

# 1-year
# training
d1_precip = df_DMT['prcp'].shift(1)

d1_tmax = df_DMT['tmax'].shift(1)

d1_tmin = df_DMT['tmin'].shift(1)

d1_vp = df_DMT['vp'].shift(1)

d1_swe = df_DMT['swe'].shift(1)

# assign new variables to DAYMET dfs
# train
df_DMT = df_DMT.assign(**{
    'tmin_1': d1_tmin,
    'tmax_1': d1_tmax,
    'prcp_1': d1_precip,
    'vp_1': d1_vp,
    'swe_1': d1_swe
    })

del(
    d1_precip,
    d1_tmax,
    d1_tmin,
    d1_vp,
    d1_swe
    )

df_DMT.to_csv(
    'D:/DataWorking/Daymet/Daymet_Annual_GAGES.csv',
    index = False
    )

# %% monthly data
############
# load data
###########

# DAYMET
# training
df_DMT = pd.read_csv(
    'D:/DataWorking/Daymet/Daymet_Monthly.csv',
    dtype = {'site_no': 'string'}
    )

# subset to working catchments
df_DMT = df_DMT[
    df_DMT['site_no'].isin(df_ID['STAID'])
    ]

# for some reason, when monthly daymet data was downloaded, I had
# 'retreived' included as part of the column names.
# rename columns
df_DMT.columns = df_DMT.columns.str.replace('_retreived', '')

##########
# calculate previous 12 months of antecedent daymet vars

# training/testin
# define working df
df_work = df_DMT[[
    'tmin', 'tmax', 'prcp', 'vp', 'swe'
    ]]

# define column names to append a label to 
# for labeling antecedent vars
cols_in = df_work.columns

# loop through number of months to get antecedent vars for
for i in range(1, 13):
        print(i)
        
        # create shifted df
        temp2 = df_DMT[cols_in].shift(i)
        
        # rename columns of shifted df
        temp2.columns = cols_in + f'_{i}'

        # append new columns to df_temp
        df_work = pd.concat(
            [df_work, temp2], axis = 1, copy = False
            )

df_DMT = pd.merge(
    df_DMT,
    df_work,
    on = ['tmin', 'tmax', 'prcp', 'vp', 'swe']
    )

# since there were some catchments with identical DAYMET forcings,
# remove duplicates after joining DMT and work df_'s.
# (note that duplicated catchments are removed in partition script)

df_DMT = df_DMT.drop_duplicates()

df_DMT.to_parquet(
    'D:/DataWorking/Daymet/Daymet_Monthly_GAGES.parquet',
    index = False
    )




# %% daily data
############
# load data
###########

# # DAYMET
# # training
# df_trainDMT = pd.read_parquet(
#     f'{dir_DMT}/daily/DAYMET_daily_train.parquet'
#     )
    
# # testin
# df_testinDMT = pd.read_parquet(
#     f'{dir_DMT}/daily/DAYMET_daily_testin.parquet'
# )
# # valnit
# df_valnitDMT = pd.read_parquet(
#     f'{dir_DMT}/daily/DAYMET_daily_valnit.parquet'
# )


# # combine training and testin data into single datasets for calculations
# # of DAYMET data
# df_tDMT = pd.concat([df_trainDMT, df_testinDMT])
# # df_tDMT = df_tDMT.reset_index(drop = True)

# # df_tDMT = df_tDMT.repartition(
# #     npartitions = df_tDMT.shape[0].compute() // 10000
# #     )

# ###################
# # calculate previous 365 days of antecedent daymet vars

# # training/testin
# # define working dfs (two to split up for memory)
# df_work = df_tDMT[[
#     'tmin', 'tmax', 'prcp', 'vp', 'swe'
# ]]


# # define column names to append a label to 
# # for labeling antecedent vars
# cols_in = df_work.columns

# # return memory being used by df_work
# df_work.info(memory_usage = 'deep')

# # change to smaller memory dtypes
# df_work = df_work.astype('float16')
# df_work.info(memory_usage = 'deep')
# # define empty list to assign output dataframes to
# # dfs_out = [0] * 73

# # loop through number of days to get antecedent vars for
# # in 73 chunks for memory

# # check if temp_out directory exists, which will hold
# # temp dataframes
# # if not, make it
# # if not os.path.isdir(f'{dir_DMT}/TEMP_DELETE'):
# #     os.mkdir(f'{dir_DMT}/TEMP_DELETE')

# # from dask.distributed import Client

# # loop through number of months to get antecedent vars for
# # for i in range(1, 3):
# #         print(i)
        
# #         # create shifted df
# #         temp2 = df_tDMT[cols_in].shift(i).astype('float16')
        
# #         # rename columns of shifted df
# #         temp2.columns = cols_in + f'_{i}'

# #         # append new columns to df_temp
# #         df_work = pd.concat(
# #             [df_work, temp2], axis = 1
# #         )

# # loop through 1:73 to divide 
# for j in range(1, 74):
    
#     # define lower and upper limits for each chunk
#     llim = (j - 1) * 5 + 1
#     ulim = j * 5 + 1

#     # define temporary dataframe to hold chunks of results
#     df_temp = pd.DataFrame()

#     # get antecedent forcings for range between llim and ulim
#     for i in range(llim, ulim):
#         print(f'{j}-{i}')
        
#         # create shifted df
#         temp2 = df_tDMT[cols_in].shift(i).astype('float32') # float16 not supported by parquet
        
#         # rename columns of shifted df
#         temp2.columns = cols_in + f'_{i}'
#         # append new columns to df_temp

#          # define temporary dataframe to hold chunks of results
#         df_temp = pd.concat([df_temp, temp2], axis = 1)

#         # dfs_out[i-1] = temp2

#         del(temp2)
#     # write temp to csv
#     df_temp.to_parquet(f'{dir_DMT}/TEMP_DELETE/cols_in_{j}.parquet')
#     del(df_temp)
        
        
# pq_list = glob.glob(
#     f'{dir_DMT}/TEMP_DELETE/*parquet'
# )

# df_pq = [dd.read_parquet(x) for x in pq_list]

# df_combined = dd.concat(df_pq, axis = 1)



# df_tDMT = pd.merge(
#     df_tDMT,
#     df_work,
#     on = ['tmin', 'tmax', 'prcp', 'vp', 'swe']
#     )

# ####

# # valnit
# # define working df
# df_work = df_valnitDMT[[
#     'tmin', 'tmax', 'prcp', 'vp', 'swe'
# ]]

# # define column names to append a label to 
# # for labeling antecedent vars
# cols_in = df_work.columns

# # loop through number of months to get antecedent vars for
# for i in range(1, 13):
#         print(i)
        
#         # create shifted df
#         temp2 = df_valnitDMT[cols_in].shift(i)
        
#         # rename columns of shifted df
#         temp2.columns = cols_in + f'_{i}'

#         # append new columns to df_temp
#         df_work = pd.concat(
#             [df_work, temp2], axis = 1, copy = False
#         )

# df_valnitDMT = pd.merge(
#     df_valnitDMT,
#     df_work,
#     on = ['tmin', 'tmax', 'prcp', 'vp', 'swe']
#     )

#################



#####
# repartition data to appropriate datasets
# get rid of year 1998
#####

# # DMT
# # training
# df_trainDMT = df_tDMT[
#     df_tDMT['year'].isin(np.arange(1999, 2008, 1))
#     ]
# df_trainDMT.to_pickle(
#     f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_train.pkl'
#     )
# # testin
# df_testinDMT = df_tDMT[
#     df_tDMT['year'].isin(np.arange(2008, 2013, 1))
#     ]
# df_testinDMT.to_pickle(
#     f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_testin.pkl'
#     )
# # valnit
# df_valnitDMT = df_valnitDMT[
#     df_valnitDMT['year'] != 1998
#     ]
# df_valnitDMT.to_pickle(
#     f'{dir_DMT}/W_AntecedentForcings/DAYMET_daily_valnit.pkl'
#     )


# # water yield
# # training
# df_trainWY = df_trainWY[
#     df_trainWY['yr'].isin(np.arange(1999, 2008, 1))
#     ]
# df_trainWY.to_pickle(
#     f'{dir_WY}/W_AntecedentForcings/WY_daily_train.pkl'
# )
# # testin
# df_testinWY = df_testinWY[
#     df_testinWY['yr'].isin(np.arange(2008, 2013, 1))
#     ]
# df_testinWY.to_pickle(
#     f'{dir_WY}/W_AntecedentForcings/WY_daily_testin.pkl'
# )
# # valnit
# df_valnitWY = df_valnitWY[
#     df_valnitWY['yr'] != 1998
#     ]
# df_valnitWY.to_pickle(
#     f'{dir_WY}/W_AntecedentForcings/WY_daily_valnit.pkl'
# )

# del(df_tDMT,
#     df_testinDMT,
#     df_testinWY,
#     df_trainDMT,
#     df_trainWY,
#     df_valnitDMT,
#     df_valnitWY)
