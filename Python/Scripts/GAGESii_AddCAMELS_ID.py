'''
BChoat 10/16/2022

Subset my data to CAMELS catchments. There are 671 
CAMELS catchments.
'''

# %%
# import libraries

import pandas as pd


# %%
# Define directories
############

# location holding camels info
dir_cam = 'D:/DataWorkingCAMELS/camels_attributes_v2.0/camels_attribues_v2.0'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'


# %%
# read in data
##########

# read in ID file (e.g., holds aggecoregions) with gauges available from '98-'12
df_ID = pd.read_csv(
    f'{dir_expl}/ID_all_avail98_12.csv',
    dtype = {'STAID': 'string'}
).drop(['CLASS', 'AGGECOREGION'], axis = 1)

# # read in ID data from HPC working data and edit to have CAMELS column
# df_IDtrain = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/'
#     'GAGESiiVariables/ID_train.csv',
#     dtype = {'STAID': 'string'}
# )
# df_IDvalnit = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/'
#     'GAGESiiVariables/ID_valnit.csv',
#     dtype = {'STAID': 'string'}
# )

# read in data to see how many catchments in common there are
# camels
df_camid = pd.read_csv(
    'D:/DataWorking/CAMELS/camels_attributes_v2.0/camels_attributes_v2.0/camels_name.txt',
    dtype = {'gauge_id': 'string'},
    sep = ';')

# # mine
# df_gagesiitrain = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_train.csv',
#     dtype = {'STAID': 'string'}
# )

# df_gagesiivalnit = pd.read_csv(
#     'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_valnit.csv',
#     dtype = {'STAID': 'string'}
# )

# # stations in common
# st_commontrain = df_camid.loc[
#     df_camid['gauge_id'].isin(df_gagesiitrain['STAID']), 'gauge_id'
#     ]

# print(f'total stations in common: {len(st_commontrain)}')

# st_commonvalnit = df_camid.loc[
#     df_camid['gauge_id'].isin(df_gagesiivalnit['STAID']), 'gauge_id'
#     ]

# print(f'total stations in common: {len(st_commonvalnit)}')


# %%
# identify and label camels catchments
##########

df_ID['CAMELS'] = [
    'y' if x in df_camid['gauge_id'].values
    else 'n' for x in df_ID['STAID']
    ]

df_ID.to_csv(
    f'{dir_expl}/ID_all_avail98_12.csv',
    index = False
)

