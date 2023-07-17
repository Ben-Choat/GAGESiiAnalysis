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

# define time_scale working with (monthly, annual, mean_annual)
time_scale = ['monthly', 'annual', 'mean_annual']


# train or test data (test = 'valnit')
trainvals_in = ['train', 'valnit']
# mean annual, annual, or monthly for plotting ranking of regionalization
timescales_in = ['mean_annual', 'annual', 'monthly']
# which model to plot eCDFs for? (model_in)
# ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost']
models_in = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost']
# drop noise (True or False)
drop_noises = [True, False]


# should csv of score ranks be saved?
write_rankCSV = True


# get STAIDs from All_1 noise
df_all1_temp = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
                            dtype = {'STAID': 'string'})

All1_noise = df_all1_temp.loc[
    (df_all1_temp['clust_method'] == 'All_1') &\
    (df_all1_temp['region'] == '-1'), 
    'STAID'
].unique()




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
                            dtype = {'STAID': 'string'})

df_ma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
                            dtype = {'STAID': 'string'})

# turn into for loop producing plots for each combo of variables above
for trainval_in, timescale_in, model_in, drop_noise in \
    product(trainvals_in, timescales_in, models_in, drop_noises):
    print(trainval_in, timescale_in, model_in, drop_noise) 

    # %% read in metrics to df
    #####################################

    # calc quantile df
    # quantiles to loop through
    qntls = np.round(np.arange(0.05, 1.0, 0.05), 2)

    if timescale_in in ['monthly', 'annual']:
        # if metric_in == 'Res':
        #     continue
        df_work = df_month_ann.copy()
        # df_work = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
        #                     dtype = {'STAID': 'string'})
        # edit region labels for those with labels in common
        df_work['region'] = df_work['clust_method'].str.cat(\
            df_work['region'].astype(str), sep = '_')
        
        if drop_noise:
            df_work = df_work[~df_work['region'].str.contains('-1')]

        # monthly and annual
        df_qntl = pd.DataFrame()
        for q in qntls:

            temp_df = df_work.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                    )[
                        ['NSE', 'KGE', 'r', 'alpha', 'beta', 'PercBias', 'RMSE']
                        ].quantile(q).reset_index()
            
            if q == qntls[0]:
                df_qntl[['clust_method', 'model', 'train_val', 'time_scale']] = \
                    temp_df[['clust_method', 'model', 'train_val', 'time_scale']]
            df_qntl[
                [f'NSE_Q{q}', f'KGE_Q{q}', f'r_Q{q}', f'beta_Q{q}', f'alpha_Q{q}']
                ] = temp_df[['NSE', 'KGE', 'r', 'beta', 'alpha']]


        # get means across all quantiles for each model, trainval and timescale row
        df_PerfMean = df_qntl.iloc[:, 0:4]
        df_PerfMean['meanKGE'] = df_qntl.loc[
            :, df_qntl.columns.str.contains('KGE')
            ].apply('mean', axis = 1)

        df_PerfMean['meanNSE'] = df_qntl.loc[
            :, df_qntl.columns.str.contains('NSE')
            ].apply('mean', axis = 1)
        

        # annual or monthly

        # create dataframe to add results, if not already exist
        if not 'df_rankSave' in globals():
            df_rankSave = pd.DataFrame(columns = df_PerfMean.columns)
            # df_rankSavema = pd.DataFrame(columns = df_PerfMeanma.columns)

        temp_df = df_PerfMean[(df_PerfMean['model'] == model_in) &\
                        (df_PerfMean['train_val'] == trainval_in) &\
                            (df_PerfMean['time_scale'] == timescale_in)].reset_index(
                                drop = True
                            )
        
        # get rankings
        tempRnk_df = temp_df.copy()
        tempRnk_df = tempRnk_df.reset_index(drop = True)
        tempRnk_df['RankKGE'] = temp_df[f'meanKGE'].rank(ascending = False)
        tempRnk_df['RankNSE'] = temp_df[f'meanNSE'].rank(ascending = False)
        # add noise column to df_rankSave
        # print(drop_noise)
        tempRnk_df['dropNoise'] = np.repeat(drop_noise, temp_df.shape[0])
        # tempRnk_df[f'mean{metric_in}'] = temp_df[f'mean{metric_in}']
        list_rankings.append(tempRnk_df)
               
        
    elif 'mean_annual' == timescale_in:
        # if metric_in in ['NSE', 'KGE']:
        #     continue
        df_workma = df_ma.copy()
        # df_workma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
        #                     dtype = {'STAID': 'string'})
        # edit region labels for those with labels in common
        df_workma['region'] = df_workma['clust_method'].str.cat(\
            df_workma['region'].astype(str), sep = '_')

        if drop_noise:
            df_workma = df_workma[~df_workma['region'].str.contains('-1')]

        # drop all noise identified by All_1 and see how scores change
        df_workma = df_workma[~df_workma['STAID'].isin(All1_noise)]


        # mean annual
        df_qntlma = pd.DataFrame()
        df_workma['|residuals|'] = np.abs(df_workma['residuals'])
        for q in qntls:

            temp_df = df_workma.groupby(
                        ['clust_method', 'model', 'train_val', 'time_scale']
                    )[
                        ['|residuals|']
                        ].quantile(q).reset_index()
            
            if q == qntls[0]:
                df_qntlma[['clust_method', 'model', 'train_val', 'time_scale']] = \
                    temp_df[['clust_method', 'model', 'train_val', 'time_scale']]
            df_qntlma[f'|residuals|_Q{q}'] = temp_df['|residuals|']


        # get means across all quantiles for each model, trainval and timescale row
        df_PerfMeanma = df_qntlma.iloc[:, 0:4]
        df_PerfMeanma['meanRes'] = df_qntlma.loc[
            :, df_qntlma.columns.str.contains('|residuals|')
            ].apply('mean', axis = 1)
        

        # annual or monthly
        # create dataframe to add results, if not already exist
        if not 'df_rankSavema' in globals():
            # df_rankSave = pd.DataFrame(columns = df_PerfMean.columns)
            df_rankSavema = pd.DataFrame(columns = df_PerfMeanma.columns)

        temp_df = df_PerfMeanma[(df_PerfMeanma['model'] == model_in) &\
                    (df_PerfMeanma['train_val'] == trainval_in) &\
                        (df_PerfMeanma['time_scale'] == timescale_in)].reset_index(
                            drop = True
                        )
        
        # get rankings
        tempRnk_df = temp_df.copy()
        tempRnk_df = tempRnk_df.reset_index(drop = True)
        tempRnk_df['RankRes'] = temp_df[f'meanRes'].rank(ascending = False)
        # add noise column to df_rankSave
        # print(drop_noise)
        tempRnk_df['dropNoise'] = np.repeat(drop_noise, temp_df.shape[0])
        # tempRnk_df[f'mean{metric_in}'] = temp_df[f'mean{metric_in}']
        list_rankingsma.append(tempRnk_df)

 
    else:
        print('Incorrect time_scale provided. Should be monthly, annual, or mean_annual')

# compile all results
df_rankSave = pd.concat(list_rankings)
df_rankSavema = pd.concat(list_rankingsma)

    
if write_rankCSV:
    # write rank dfs to csvs
    df_rankSave.to_csv(
        f'{dir_work}/PerfRankMonthAnnual_ByClustMeth.csv',
        index = False
    )
    # first remove meanNSE and/or meanKGE from df_rankSavema, since it is an error
    # df_rankSavema = df_rankSavema.drop(f'mean{metric_in}', axis = 1)
    df_rankSavema.to_csv(
        f'{dir_work}/PerfRankMeanAnnual_ByClustMeth.csv',
        index = False
    )
