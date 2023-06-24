# A script to combine ID data for training and validating data
# just writing for reproducability


# %% import libraries
import pandas as pd

# %% load data

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_Train_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])
# val_in
df_valin_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_ValIn_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE', 'GEOL_REEDBUSH_DOM_anorthositic'])
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/ExplVars_Model_In/All_ExplVars_ValNit_Interp_98_12.csv',
    dtype = {'STAID': 'string'}
).drop(columns = ['LAT_GAGE', 'LNG_GAGE'])

#########################
# ID vars (e.g., ecoregion)
# vars to color plots with (e.g., ecoregion)
df_ID = pd.read_csv(
    f'{dir_expl}/GAGES_idVars.csv',
    dtype = {'STAID': 'string'}
)

# training ID
df_train_ID = df_ID[df_ID.STAID.isin(df_train_expl.STAID)].reset_index(drop = True)
# val_in ID
df_valin_ID = df_train_ID
# val_nit ID
df_valnit_ID = df_ID[df_ID.STAID.isin(df_valnit_expl.STAID)].reset_index(drop = True)

#########################
# read in BasinID sheet from All_GAGESiiTS.xlsx
df_work = pd.read_excel('D:/DataWorking/GAGESii_TS/All_GAGESiiTS.xlsx', 
    sheet_name='BasinID',
    dtype = {'STAID': 'string'}
    ).drop(columns = ['STANAME', 'STATE', 'HCDN-2009'])

# subset df_work to train, and valnit dfs
df_work_train = pd.merge(
    df_work, df_train_ID,
    left_on = ['STAID', 'AGGECOREGION'],
    right_on = ['STAID', 'AggEcoregion']
)

df_work_valnit = pd.merge(
    df_work, df_valnit_ID,
    left_on = ['STAID', 'AGGECOREGION'],
    right_on = ['STAID', 'AggEcoregion']
)

# 

# write new ID dfs 

df_work_train.to_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/ID_train.csv')
df_work_valnit.to_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/ID_valnit.csv')

# %%
