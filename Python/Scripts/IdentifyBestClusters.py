'''
BChoat 2023/04/16

Code to identify best cluster from each clustering method based
on performance metrics from regression and other predictive models.
For example, was Anth_0 or Anth_1 better?


Thoughts:
- comapre clustering improvements compared to none, for each catchment.
- count the number of catchments that performed better and worse with
    clustering applied.
- 

NOTE:
0.5 > NSE: unsatisfactory
0.5 <= NSE < 0.65: satisfactory
0.65 <= NSE < 0.46: good
0.75 <= NSE: very good
'''

#%% 
# Load Libraries


# from genericpath import exists
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# import numpy as np
# import glob
# import os
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
# import xgboost as xgb

# %% 
# Define working directories and variables

# define which clustering methods to compare.
clust_meths = ['None', 'Class', 'AggEcoregion', 
                'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1', 
                'CAMELS', 'HLR', 'Nat_0', 'Nat_1', 'Nat_2',
                'Nat_3', 'Nat_4']
# clust_meths = ['Nat_3']
# clust_meths = ['All_0', 'All_1', 'All_2']
# clust_meths = ['Anth_0', 'Anth_1', 'Anth_2']
# clust_meths = ['All_0', 'All_1', 'All_2',
#                'Anth_0', 'Anth_1',
#                'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4']

# define time scale working with.
time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly'

# performance metric to explore (just used in plotting)
perf_metr = 'KGE' # residuals, r2, NSE, KGE, percBias


# investigate 'train', 'valnit', or 'both' partitions
partition_in = 'valnit'

# define list of quantiles to use when identifying the best 
# clustering method
# qnts_in = [0.025, 0.05, 0.95, 0.975]
# qnts_in = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
qnts_in = [0.05, 0.25, 0.5, 0.75, 0.95]
# qnts_in = [0.25, 0.5, 0.75]

# drop noise cluster (True or False)
drop_noise = True

# define list of models to be subset to
# if model_in is not a list, then data will not be subset
# ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost']
model_in = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'] # ['XGBoost'] # None

# directory/file holding combined results
# summary results
# dir_in = f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out' \
#         f'/{time_scale}/combined/All_SummaryResults_{time_scale}.pkl'
# independent catchment results
dir_in = f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out' \
            f'/{time_scale}/combined/All_IndResults_{time_scale}.pkl'

# directory holding scores from clustering
dir_clustScores = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'

# write out figs? (True or False)
write_figs = False
# directory where to write out folders
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/ModelPerformance'



# %%
# define function to calc weighted average of ranks
###########################


def wtd_avg(df, cols):
    '''
    inputs:
        df: pandas dataframe
        cols: names of columns in dataframe, for which to calulate
            the weighted average - rowwise

    outputs:
        array of weighted averages with len = nrows(df)
    '''



# %%
# read in data
########################


# read in results
df_results = pd.read_pickle(dir_in)

# subset to clust_meths currently investigating
df_results = df_results[df_results['clust_method'].isin(clust_meths)]

# subset to train or valnit if partition_in does not equal 'both'
if partition_in != 'both':
    df_results = df_results[df_results['train_val'] == partition_in]


# subset to model of choice if model_in is list
if isinstance(model_in, list):
    df_results = df_results[df_results['model'].isin(model_in)]



# %%
# generate noise dataframes
###################################


# read in cluster scores
df_clustScores_train = pd.read_csv(
    f'{dir_clustScores}/ClusterScores_train.csv',
    dtype = {'STAID': 'string'}
)
df_clustScores_valnit = pd.read_csv(
    f'{dir_clustScores}/ClusterScores_valnit.csv',
    dtype = {'STAID': 'string'}
) 
# combine into one dataframe
df_clustScores = pd.concat([
    df_clustScores_train, df_clustScores_valnit
])

del(df_clustScores_train, df_clustScores_valnit)

# if 'XGBoost' in model_in:
# get count of noise group for each method
df_noise = df_results[(df_results['model'] == model_in[0])]
# create list to append noise counts to and to hold clust_methods
ns_cnts = []
cls_mths = []
ratios = []
for i in df_noise.clust_method.unique():
    temp_ns = df_noise[
        (df_noise['clust_method'] == i) &
    (df_results['region'] == -1)].shape[0]
    ns_cnts.append(temp_ns)
    
    temp_tot = df_noise[df_noise['clust_method'] == i].shape[0]

    temp_ratio = round(temp_ns/temp_tot, 3)
    cls_mths.append(i)
    ratios.append(temp_ratio)


df_noise = pd.DataFrame({
    'clust_meth': cls_mths,
    'noise_count': ns_cnts,
    'ratio_noise': ratios
}).sort_values(by = 'ratio_noise', ascending = True)

df_noise['noise_rank'] = df_noise.ratio_noise.rank(method = 'dense')
# .groupby(
#     ['clust_method', 'model']
#     ).count()

# drop -1 'noise' group
if drop_noise:
    df_results = df_results.loc[df_results['region'] != -1]


# %%
# bar plot of noise
############################


# sns.barplot(
#     data = df_noise[df_noise['noise_rank'] != 1.0],
#     x = 'clust_meth',
#     y = 'ratio_noise'
# )


# df_results = df_results[df_results['region'].isin(np.linspace(14, 26, 10))]

# %% 
# calc summary stats for comparing clustering methods
#####################################


# calc dataframe of mean values by model, cluster, region
df_meanvalues = df_results.groupby([
    'model', 'clust_method', 'region'
    ]).mean().reset_index()

# calc percentiles 2.5, 5, 25, 50, 75, 95, 97.5
# of NSE, KGE, and %bias
# for mean annual, use 
# mean annual
if time_scale == 'mean_annual':

    # calc additional metrics
    df_results['|residuals|'] = np.abs(df_results['residuals'])

    df_ranking = df_results.groupby(
        ['clust_method', 'model']
        ).quantile(
        qnts_in
        ).reset_index()
        
    
    df_ranking.columns = ['clust_method', 
                          'model', 
                          'quantile', 
                          'residuals',
                          '|residuals|']
    
    df_meanrank = df_ranking.groupby(
        ['clust_method', 'model']
        ).mean(
        ['|residuals|'] #, 'percBias_rank'] # , 'RMSEts']
        ).reset_index()

    # take mean of each metric for comparing
    df_meanrank['mean_rank'] = df_meanrank[[
        '|residuals|' # , 'percBias_rank' # , 'RMSEts_rank'
    ]].mean(axis = 1)#.reset_index()


    # drop original scores
    df_meanrank = df_meanrank.drop(
        ['residuals', 'quantile'],
          axis = 1).sort_values(
        by = 'mean_rank', ascending = True
        ).reset_index(drop = True)
    


# not mean annual
if time_scale != 'mean_annual':


    # calc additional metrics
    # get absvalue percBias
    df_results['percBias'] = np.abs(df_results['percBias'])
    # normalized NSE
    df_results['NSE_norm'] = 1/(2 - df_results['NSE'])
    # normalized KGE
    df_results['KGE_norm'] = 1/(2 - df_results['KGE'])



    df_ranking = df_results.groupby(
        ['clust_method', 'model']
        ).quantile(
        qnts_in
        ).reset_index()
    
    df_ranking.columns = ['clust_method', 
                          'model', 
                          'quantile', 
                          'NSE',
                          'KGE',
                          'percBias',
                          'RMSEts',
                          'NSE_norm',
                          'KGE_norm']


    # get ranking for score within each group
    for metric in ['NSE', 'KGE', 'percBias', 'RMSEts']:
        # define ascending or not (NSE: larger is better, RMSE: smaller is better)
        if metric in ['NSE', 'KGE']:
            ascending_in = False
        else:
            ascending_in = True
        df_ranking[f'{metric}_rank'] = df_ranking.groupby(
            ['quantile'] # ['model', 'quantile']
        )[metric].rank(method = 'dense', ascending = ascending_in)

    df_meanrank = df_ranking.groupby(
        ['clust_method', 'model']
        ).mean(
        ['NSE_rank', 'KGE_rank'] #, 'percBias_rank'] # , 'RMSEts']
        ).reset_index()
    
    
    # if 'XGBoost' in model_in:
    df_meanrank = pd.merge(
        df_noise,
        df_meanrank,
        left_on = 'clust_meth',
        right_on = 'clust_method'
    ).drop('clust_meth', axis = 1)

    # take mean of each metric for comparing
    df_meanrank['mean_rank'] = df_meanrank[[
        'NSE_rank', 'KGE_rank' # , 'noise_rank' # , 'percBias_rank' # , 'RMSEts_rank'
    ]].mean(axis = 1)#.reset_index()

    # else:
    #     # take mean of each metric for comparing
    #     df_meanrank['mean_rank'] = df_meanrank[[
    #         'NSE_rank', 'KGE_rank' # , 'percBias_rank' # , 'RMSEts_rank'
    #     ]].mean(axis = 1)#.reset_index()

    # # drop original scores
    # df_meanrank = df_meanrank.drop(
    #     ['NSE', 'KGE', 'RMSEts', 'percBias', 'quantile'],
    #       axis = 1).sort_values(
    #     by = 'mean_rank', ascending = True
    #     ).reset_index(drop = True)
    
    
        

    # mean_rank = df_ranking.groupby(
    #     ['clust_method', 'model']
    # ).mean().reset_index()


# %%
# plot summary plots for comparisons
####################################

df_meanrank = df_meanrank.sort_values(by = 'mean_rank').reset_index(drop = True)

best_meth = df_meanrank.loc[0, 'clust_method']
# best_meth = 'Nat_3'
best_mod = df_meanrank.loc[0, 'model']

# best_meth = 'AggEcoregion'
# best_mod = 'strd_mlr'


# define dataframe to plot as clust_method with best mean ranking
# and best model combo
df_results_plot = df_results[
    (df_results['clust_method'] == best_meth) &
    (df_results['model'] == best_mod)
    ]


if time_scale == 'mean_annual':
    y_min_in = df_results_plot['residuals'].min()
    y_min_in = y_min_in + 0.01*y_min_in
    y_max_in = df_results_plot['residuals'].max()
    y_max_in = y_max_in + 0.01*y_max_in

else:
    y_min_in = -1.1
    y_max_in = 1.1

model_plt = np.unique(df_results_plot['model'])

fig, axs = plt.subplots(nrows = 1, ncols = 1)
p = sns.boxplot(data = df_results_plot,
                x = 'region', # 'clust_method',
                y = perf_metr,
                ax = axs,
                hue = 'region',
                dodge = False) # 'train_val')
# p2 = sns.swarmplot(data = df_results_plot,
#                 x = 'clust_method',
#                 y = perf_metr,
#                 ax = axs,
#                 hue = 'train_val')
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.set_ylim(y_min_in, y_max_in)
axs.set_title(f'{perf_metr}-{best_meth} by {best_mod}\n{time_scale}')
plt.legend(framealpha = 0.5, ncol = 4)

if write_figs:
    plt.savefig(
        f'{dir_figs}/{time_scale}_{perf_metr}_ByMeth_AllModels_1fig.png', dpi = 300
        )
else:
    plt.show()


# get min-y for axis limit
y_min_in = df_results_plot[perf_metr].min()
y_min_in = y_min_in + 0.05 * y_min_in

# boxplot of scores
fig, axs = plt.subplots(nrows = 2, ncols = 1, sharex = True)
p = sns.boxplot(data = df_results_plot,
                x = 'region', # 'clust_method',
                y = perf_metr,
                ax = axs[0],
                hue = 'region',
                dodge = False) # 'train_val')
axs[0].set_ylim(-1.2, 1.1)
# plt.ylim(-1, 1.5)
axs[0].set_title(f'{perf_metr}-{best_meth} by {best_mod}\n{time_scale}')
axs[0].legend().remove()
axs[0].set_xlabel(None)

p = sns.boxplot(data = df_results_plot,
                x = 'region', # 'clust_method',,
                y = perf_metr,
                ax = axs[1],
                hue = 'region') # 'train_val')
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90)
axs[1].set_ylim(y_min_in, -1.0)
axs[1].legend(framealpha=0.5, loc = 'best', ncol = 4)
plt.subplots_adjust(wspace = 0, hspace = 0.02)

if write_figs:
    plt.savefig(
        f'{dir_figs}/{time_scale}__{perf_metr}_ByMeth_AllModels_2fig.png', dpi = 300
        )
else:
    plt.show()








# %%
# investigate cluster score vs performance
###############################

# # merge with results
# df_resclust = pd.merge(
#     df_results,
#     df_clustScores,
#     on = 'STAID'
# )

# df_clust_plot = df_resclust[
#     df_resclust['model'] == 'XGBoost'
#     ]


# # df_clust_plot
# p = sns.scatterplot(
#     data = df_clust_plot,
#     x = 'All_1_score',
#     y = 'KGE_norm'
# )

# p.set(ylim = (0, 1))

# pd.wide_to_long(
#     df_clust_plot.rename(columns=lambda x: '_'.join(x.split('_')[::-1])), # df_clust_plot,
#     stubnames = 'score',
#     i = ['STAID', 'model', 'clust_method', 'region'],
#     j = 'Score'
# )









# %%
# compare clustering methods in clust_meths
###########################


# return the model and parameters for the model performing best on
# valnit data
# use NSE if time series data, or r2 if non-time series (e.g., mean annaul)
# use median values of the score for each model
#  from all regions to identify best model


# subset to train and valnit, groupby model and cluster method, sort values by nse
df_median_train = df_results[
    df_results['train_val'] == 'train'
    ].groupby(
    ['model', 'clust_method']
    ).median().reset_index().sort_values(
        by = perf_metr, ascending = False
        )
df_median_valnit = df_results[
    df_results['train_val'] == 'valnit'
    ].groupby(
    ['model', 'clust_method']
    ).median().reset_index().sort_values(
        by = perf_metr, ascending = False
        )

# find max valnit value
# if time_scale == 'mean_annual':
max_valnit = np.max(df_results.loc[
df_results['train_val'] == 'valnit', perf_metr
])
# subset to best model based on max valnit NSE    
best_model = df_results.loc[
    (df_results['train_val'] == 'valnit') &
    (df_results[perf_metr] == max_valnit),
    'model'
]

# else:
#     max_valnit = np.max(df_results.loc[
#         df_results['train_val'] == 'valnit', 'NSE'
#         ])
#     # subset to best model based on max valnit NSE    
#     best_model = df_results.loc[
#         (df_results['train_val'] == 'valnit') &
#         (df_results['NSE'] == max_valnit),
#         'model'
#     ]

# # append best_model to list of best_models
# # and best score to list of best scores
# best_models.append(best_model)
# best_scores.append(max_valnit)


