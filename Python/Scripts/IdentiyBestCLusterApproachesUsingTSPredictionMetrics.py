'''
2023/07/02 BChoat

Script to read in results from either CacluateComponentsOfKGE.py
or CalculatePerformanceMetrics_MeanAnnual.py, and uses those values
to identify best performing models.

Looking for best clustering approach or two out of each of the following
groups which consistently clustered shared > 0.6 AMI.

Consistent groups​
1. 
Class
CAMELS (0.68)​
2.
Eco3 – ​
USDA_LRR_Site (0.65)​
3.
All_0​
All_1 (0.91)​
All_2 (0.82)​
Nat_0 (0.75)​
Nat_1 (0.74)​
Nat_2 (0.75)​
4.
Nat_3​
Nat_4 (0.94)​
5.
Anth_0 ​
Anth_1 (0.87)
'''


# %% import libraries
###############################################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import os


# %% define directories, variables, and such
#############################################


# dir holding performance metrics
dir_work = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Results'

# dir where to place figs
dir_figs = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Figures'

# should csv of score ranks be saved?
write_rankCSV = True


# get STAIDs from All_1 noise
# df_all1_temp = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
#                             dtype = {'STAID': 'string'})

# All1_noise = df_all1_temp.loc[
#     (df_all1_temp['clust_method'] == 'All_1') &\
#     (df_all1_temp['region'] == '-1'), 
#     'STAID'
# ].unique()




# %% Run loop to get all ranks and scores by clustering method
################################

# delete output dfs, in case they are left over from a previous execution
if 'df_rankSave' in globals():
        del(df_rankSave)
if 'df_rankSavema' in globals():
        del(df_rankSavema)


# create list to hold top 5 performing clustering approaches for each model
# annual or monthly
list_bestPerf = []
# list to hold rankings results
list_rankings = []
# mean annual
# create list to hold top 5 performing clustering approaches for each model
list_bestPerfma = []
# list to hold rankings results
list_rankingsma = []


# read in main dfs
df_month_ann = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
                            dtype = {'STAID': 'string',
                                     'region': 'string'})

df_ma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
                            dtype = {'STAID': 'string',
                                     'region': 'string'})



# quantiles to include
qnts_in = np.round(np.arange(0.05, 1.0, 0.05), 2)



# %% Calculate mean perf_metric of quantiles
###############################

###############
# monthly and annual
##############

if 'df_qntls' in globals():
        del(df_qntls)

# w/NOISE

df_work = df_month_ann.copy()

for q in qnts_in:
    if not 'df_qntls' in globals():
        # NSE
        df_qntls = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['NSE'].quantile(q).reset_index()
        df_qntls = df_qntls.rename(columns = {'NSE': f'NSE_q{q}'})
        # KGE
        df_qntls[f'KGE_q{q}'] = df_work.groupby(
            ['clust_method', 'model', 'train_val', 'time_scale']
            )['KGE'].quantile(q).reset_index()['KGE']
    else:
        # NSE
        df_qntls[f'NSE_q{q}'] = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['NSE'].quantile(q).reset_index()['NSE']
        # KGE
        df_qntls[f'KGE_q{q}'] = df_work.groupby(
            ['clust_method', 'model', 'train_val', 'time_scale']
            )['KGE'].quantile(q).reset_index()['KGE']


# NSE
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('NSE_q')].values.tolist()
df_qntls['NSE_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)
# KGE
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('KGE_q')].values.tolist()
df_qntls['KGE_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)
# note if noise dropped or not
df_meanqntl = df_qntls[[
      'clust_method', 'model', 'train_val', 'time_scale', 'NSE_qmean', 'KGE_qmean'
      ]]
df_meanqntl['dropNoise'] = np.repeat('False', df_meanqntl.shape[0])



# wo/NOISE

if 'df_qntls' in globals():
        del(df_qntls)

df_work = df_month_ann.copy()
df_work = df_work[df_work['region'] != '-1']

for q in qnts_in:
    if not 'df_qntls' in globals():
        # NSE
        df_qntls = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['NSE'].quantile(q).reset_index()
        df_qntls = df_qntls.rename(columns = {'NSE': f'NSE_q{q}'})
        # KGE
        df_qntls[f'KGE_q{q}'] = df_work.groupby(
            ['clust_method', 'model', 'train_val', 'time_scale']
            )['KGE'].quantile(q).reset_index()['KGE']
    else:
        # NSE
        df_qntls[f'NSE_q{q}'] = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['NSE'].quantile(q).reset_index()['NSE']
        # KGE
        df_qntls[f'KGE_q{q}'] = df_work.groupby(
            ['clust_method', 'model', 'train_val', 'time_scale']
            )['KGE'].quantile(q).reset_index()['KGE']


# NSE
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('NSE_q')].values.tolist()
df_qntls['NSE_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)
# KGE
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('KGE_q')].values.tolist()
df_qntls['KGE_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)
# note if noise dropped or not
df_meanqntldn = df_qntls[[
      'clust_method', 'model', 'train_val', 'time_scale', 'NSE_qmean', 'KGE_qmean'
      ]]
df_meanqntldn['dropNoise'] = np.repeat('True', df_meanqntldn.shape[0])


df_meanqntl = pd.concat([df_meanqntl, df_meanqntldn])


df_meanqntl['NSE_rank'] = df_meanqntl.groupby([
        'model', 'train_val', 'time_scale', 'dropNoise'
    ])['NSE_qmean'].rank(ascending = False)

df_meanqntl['KGE_rank'] = df_meanqntl.groupby([
        'model', 'train_val', 'time_scale', 'dropNoise'
    ])['KGE_qmean'].rank(ascending = False)

if write_rankCSV:
    df_meanqntl.to_csv(
            f'{dir_work}/PerfRankMonthAnnual_ByClustMethModel.csv',
            index = False
        )








#############################
# mean_annual
#############################


if 'df_qntls' in globals():
        del(df_qntls)

# w/NOISE

df_work = df_ma.copy()

df_work['|residuals|'] = df_work['residuals'].abs()

for q in qnts_in:
    if not 'df_qntls' in globals():
        # residuals
        df_qntls = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['|residuals|'].quantile(q).reset_index()
        df_qntls = df_qntls.rename(columns = {'|residuals|': f'|residuals|_q{q}'})

    else:
        # residuals
        df_qntls[f'|residuals|_q{q}'] = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['|residuals|'].quantile(q).reset_index()['|residuals|']



# residuals
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('|residuals|_q')].values.tolist()
df_qntls['|residuals|_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)

# note if noise dropped or not
df_meanqntl = df_qntls[[
      'clust_method', 'model', 'train_val', 'time_scale', '|residuals|_qmean'
      ]]
df_meanqntl['dropNoise'] = np.repeat('False', df_meanqntl.shape[0])



# wo/NOISE

if 'df_qntls' in globals():
        del(df_qntls)

df_work = df_ma.copy()
df_work = df_work[df_work['region'] != '-1']

df_work['|residuals|'] = df_work['residuals'].abs()

for q in qnts_in:
    if not 'df_qntls' in globals():
        # residuals
        df_qntls = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['|residuals|'].quantile(q).reset_index()
        df_qntls = df_qntls.rename(columns = {'|residuals|': f'|residuals|_q{q}'})

    else:
        # residuals
        df_qntls[f'|residuals|_q{q}'] = df_work.groupby(
                ['clust_method', 'model', 'train_val', 'time_scale']
                )['|residuals|'].quantile(q).reset_index()['|residuals|']

# residuals
subset_cols = df_qntls.columns[df_qntls.columns.str.contains('|residuals|_q')].values.tolist()
df_qntls['|residuals|_qmean'] = df_qntls[subset_cols].apply('mean', axis = 1)

# note if noise dropped or not
df_meanqntldn = df_qntls[[
      'clust_method', 'model', 'train_val', 'time_scale', '|residuals|_qmean'
      ]]
df_meanqntldn['dropNoise'] = np.repeat('True', df_meanqntldn.shape[0])


df_meanqntl = pd.concat([df_meanqntl, df_meanqntldn])


df_meanqntl['|residuals|_rank'] = df_meanqntl.groupby([
        'model', 'train_val', 'time_scale', 'dropNoise'
    ])['|residuals|_qmean'].rank(ascending = True)


if write_rankCSV:
    df_meanqntl.to_csv(
            f'{dir_work}/PerfRankMeanAnnual_ByClustMethModel.csv',
            index = False
        )


