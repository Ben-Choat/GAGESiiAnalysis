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

# %% define directories, variables, and such
#############################################


# dir holding performance metrics
dir_work = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Results'

# dir where to place figs
dir_figs = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Figures'

# define time_scale working with (monthly, annual, mean_annual)
time_scale = ['monthly', 'annual', 'mean_annual']

# train or test data (test = 'valnit')
trainval_in = 'valnit'
# mean annual, annual, or monthly for plotting ranking of regionalization
timescale_in = 'mean_annual'
# drop noise (True or False)
drop_noise = False
# should heat map be saved?
write_hmfig = False
# should ecdfs be svaed?
write_ecdf = False


# %% read in metrics to df
#####################################


if 'monthly' in time_scale or 'annual' in time_scale:
    df_work = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
                          dtype = {'STAID': 'string'})
    # edit region labels for those with labels in common
    df_work['region'] = df_work['clust_method'].str.cat(\
        df_work['region'].astype(str), sep = '_')
    
    if drop_noise:
        df_work = df_work[~df_work['region'].str.contains('-1')]

    
if 'mean_annual' in time_scale:
    df_workma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
                          dtype = {'STAID': 'string'})
    # edit region labels for those with labels in common
    df_workma['region'] = df_workma['clust_method'].str.cat(\
        df_workma['region'].astype(str), sep = '_')

    if drop_noise:
        df_workma = df_workma[~df_workma['region'].str.contains('-1')]

else:
    print('Incorrect time_scale provided. Should be monthly, annual, or mean_annual')



# %% Calcualte some stats
##################################

# # mean
# df_meanKGE = df_work.groupby(
#     ['clust_method', 'model', 'train_val', 'time_scale']
#     ).mean().reset_index().sort_values(by = 'KGE')

# # median
# df_medianKGE = df_work.groupby(
#     ['clust_method', 'model', 'train_val', 'time_scale']
#     ).median().reset_index().sort_values(by = 'KGE')

# # q25
# df_q25KGE = df_work.groupby(
#     ['clust_method', 'model', 'train_val', 'time_scale']
#     ).quantile(0.25).reset_index().sort_values(by = 'KGE')

# # q05
# df_q05KGE = df_work.groupby(
#     ['clust_method', 'model', 'train_val', 'time_scale']
#     ).quantile(0.05).reset_index().sort_values(by = 'KGE')



# %% get quantiles for all clust_methods
#####################################


# calc quantile df
# quantiles to loop through
qntls = np.round(np.arange(0.05, 1.0, 0.05), 2)

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
# create list to hold top 5 performing clustering approaches for each model
list_bestPerf = []
# list to hold rankings results
list_rankings = []

for ts_in in ['monthly', 'annual']:
    for model in df_PerfMean.model.unique():
        temp_df = df_PerfMean[(df_PerfMean['model'] == model) &\
                            (df_PerfMean['train_val'] == trainval_in) &\
                                (df_PerfMean['time_scale'] == ts_in)]
        
        # get rankings
        tempRnk_df = temp_df.iloc[:, 0:4]
        tempRnk_df['Rank'] = temp_df['meanKGE'].rank(ascending = False)
        list_rankings.append(tempRnk_df)

        # get top 5 best performing groupings
        temp_df = temp_df.sort_values(by = 'meanKGE').tail(5)
        # print(temp_df)
        list_bestPerf.append(temp_df)

df_ranked = pd.concat(list_rankings)
df_bestPerf = pd.concat(list_bestPerf)


# mean annual
# create list to hold top 5 performing clustering approaches for each model
list_bestPerf = []
# list to hold rankings results
list_rankings = []

for model in df_PerfMeanma.model.unique():
    temp_df = df_PerfMeanma[(df_PerfMeanma['model'] == model) &\
                        (df_PerfMeanma['train_val'] == trainval_in) &\
                            (df_PerfMeanma['time_scale'] == 'mean_annual')]
    
    # get rankings
    tempRnk_df = temp_df.iloc[:, 0:4]
    tempRnk_df['Rank'] = temp_df['meanRes'].rank(ascending = True)
    list_rankings.append(tempRnk_df)

    # get top 5 best performing groupings
    temp_df = temp_df.sort_values(by = 'meanRes').tail(5)
    # print(temp_df)
    list_bestPerf.append(temp_df)

df_rankedma = pd.concat(list_rankings)
df_bestPerfma = pd.concat(list_bestPerf)

# %% do some exploring
###################################


# # just best test scores
# df_medianKGE[df_medianKGE['train_val'] == trainval_in].tail(20)

# # just best test scores
# # ['regr_precip', 'strd_PCA_mlr', 'strd_mlr', 'XGBoost']
# model_in = ['regr_precip', 'strd_PCA_mlr', 'strd_mlr', 'XGBoost']
# df_medianKGE[(df_medianKGE['train_val'] == trainval_in) &\
#              (df_medianKGE['model'].isin(model_in))].tail(30)

# df_q25KGE[(df_q25KGE['train_val'] == trainval_in) &\
#              (df_q25KGE['model'].isin(model_in))].tail(30)

# df_q05KGE[(df_q25KGE['train_val'] == trainval_in) &\
#              (df_q25KGE['model'].isin(model_in))].tail(30)

# df_meanKGE[(df_meanKGE['train_val'] == trainval_in) &\
#              (df_meanKGE['model'].isin(model_in))].tail(30)

# %% filter to best performing cluster approach for each 
# group that consistently shares mutual information >0.6
#################################################


# GrClassCam = [ 'Class', 'CAMELS']
# AllNat = ['All_0', 'All_1', 'All_2', 'Nat_1', 'Nat_2']
# Nat34 = ['Nat_3', 'Nat_4']
# Anth = ['Anth_0', 'Anth_1']
# AggEco = ['AggEcoregion']
# HLR = ['HLR']






# %% Boxplots
###############################


# ax = sns.boxplot(data = df_work[df_work['train_val'] == trainval_in], 
#                  x = 'clust_method',
#                  y = 'KGE', 
#                  hue = 'model')
# # ax.set_yscale('log')
# ax.set_ylim(-1, 1)
# plt.xticks(rotation = 45,
#            ha = 'right',
#            rotation_mode = 'anchor')
# plt.legend(framealpha = 0.25)
# plt.show()



# ax = sns.boxplot(data = df_q25KGE, 
#                  x = 'clust_method',
#                  y = 'KGE', 
#                  hue = 'model')
# # ax.set_yscale('log')
# ax.set_ylim(-1, 1)
# plt.xticks(rotation = 45,
#            ha = 'right',
#            rotation_mode = 'anchor')
# plt.legend(framealpha = 0.25)
# plt.show()



# ax = sns.boxplot(data = df_meanKGE, 
#                  x = 'clust_method',
#                  y = 'KGE', 
#                  hue = 'model')
# # ax.set_yscale('log')
# ax.set_ylim(-1, 1)
# plt.xticks(rotation = 45,
#            ha = 'right',
#            rotation_mode = 'anchor')
# plt.legend(framealpha = 0.25)
# plt.show()




# %%
# eCDF plots
##############


# mean annual
data_in_mannual = df_workma[
    df_workma['train_val'] == trainval_in
]

# data_in_mannual = pd.merge(
#     data_in_mannual, df_shap_mannual,
#     left_on = ['region', 'clust_method', 'model'],
#     right_on = ['region', 'clust_meth', 'best_model']
# )[df_resind_mannual.columns]

data_in_mannual['|residuals|'] = np.abs(data_in_mannual['residuals'])

data_in_mannual.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


# annual
data_in_annual = df_work[
    (df_work['train_val'] == trainval_in) &\
    (df_work['time_scale'] == 'annual')
]

# data_in_annual = pd.merge(
#     data_in_annual, df_shap_annual,
#     left_on = ['region', 'clust_method', 'model'],
#     right_on = ['region', 'clust_meth', 'best_model']
# )[df_resind_annual.columns]

data_in_annual.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


# month
data_in_month = df_work[
    (df_work['train_val'] == trainval_in) &\
    (df_work['time_scale'] == 'monthly')
]

# data_in_month = pd.merge(
#     data_in_month, df_shap_monthly,
#     left_on = ['region', 'clust_method', 'model'],
#     right_on = ['region', 'clust_meth', 'best_model']
# )[df_resind_monthly.columns]


data_in_month.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


####
# define list of columns to use when calculating ami scores
clust_meth_in = [
    'None',
    'Class', 'CAMELS',
    'AggEcoregion',
    'HLR', 
    'All_0', 'All_1', 'All_2',  'Nat_0', 'Nat_1', 'Nat_2', 
    'Nat_3', 'Nat_4', 
    'Anth_0', 'Anth_1'
]
# ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost']
model_in = 'XGBoost'
metric_in = 'KGE'
# palette_in = 'Paired'
palette_in = 'gist_ncar'
data_in_mannual = data_in_mannual[
    (data_in_mannual['clust_method'].isin(clust_meth_in)) &
    (data_in_mannual['model'] == model_in)
]

data_in_annual = data_in_annual[
    (data_in_annual['clust_method'].isin(clust_meth_in)) &
    (data_in_annual['model'] == model_in)
]

data_in_month = data_in_month[
    (data_in_month['clust_method'].isin(clust_meth_in)) &
    (data_in_month['model'] == model_in)
]

# if drop_noise, drop -1 regions
if drop_noise:
    data_in_mannual = data_in_mannual[~data_in_mannual['region'].str.contains('-1')]
    data_in_annual = data_in_annual[~data_in_annual['region'].str.contains('-1')]
    data_in_month = data_in_month[~data_in_month['region'].str.contains('-1')]

####

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 5), sharey = True)

ax1.set(xlabel = '|residuals| [cm]', 
        ylabel = 'Non-Exceedence Probability')
ax1.set_xlim(0, 50)
ax1.annotate('(a)', xy = (2.7, 0.02))
ax1.grid()
ax1.title.set_text('Mean Annual')
sns.ecdfplot(
    data = data_in_mannual,
    x = '|residuals|',
    hue = 'clust_method', # 'region',
    linestyle = '--',
    palette = palette_in,
    ax = ax1
)
# sns.move_legend(ax1, 'lower center')
# ax1.legend(framealpha = 0.5)
# ax1.get_legend().set_title('Region')



ax2.set_xlim(-1, 1)
ax2.set(xlabel = metric_in)
ax2.annotate('(b)', xy = (0.8, 0.02))
ax2.grid()
ax2.title.set_text('Annual')
sns.ecdfplot(
    data = data_in_annual,
    x = metric_in,
    hue = 'clust_method', # 'region',
    linestyle = '--',
    palette = palette_in,
    ax = ax2,
    legend = False
)


ax3.set_xlim(-1, 1)
ax3.set(xlabel = metric_in)
ax3.annotate('(c)', xy = (0.8, 0.02))
ax3.grid()
ax3.title.set_text('Monthly')
sns.ecdfplot(
    data = data_in_month,
    x = metric_in,
    hue = 'clust_method', # 'region',
    linestyle = '--',
    palette = palette_in,
    ax = ax3,
    legend = False
)

# edit string to use in title
if trainval_in == 'train':
    trainval_temp = 'train'
else:
    trainval_temp = 'test'

plt.suptitle(
    f'eCDFs {trainval_temp} data'
)

# save fig
if write_ecdf:
    plt.savefig(
        f'{dir_figs}/ecdfs.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )

else:
    plt.show()



# %% get heatmap presenting order of performance for various models and timescales
#######################################################

df_plot = pd.DataFrame('', 
                       index = df_ranked['clust_method'].unique(), 
                       columns = df_ranked['model'].unique()
                       )


# define colors to use
colors_in = [
    'orangered', 'orange', 'yellow',  'sandybrown', 'darkkhaki',   
    'yellowgreen', 'green', 'springgreen', 'aquamarine', 'blue', 
    'darkblue', 'blueviolet', 'purple','orchid', 'deeppink'
]

from itertools import product
for i, j in product(range(len(df_ranked['clust_method'].unique())),
             range(len(df_ranked['model'].unique()))):

    # monthly or annual
    if (timescale_in == 'annual') | (timescale_in == 'monthly'):
    
        # get temp clust method and model
        temp_clustmeth = df_ranked['clust_method'].unique()[i]
        temp_model = df_ranked['model'].unique()[j]

        # populate df_plot
        df_plot.iloc[i, j] = df_ranked.loc[
            (df_ranked['clust_method'] == temp_clustmeth) &\
            (df_ranked['model'] == temp_model) &\
                (df_ranked['time_scale'] == timescale_in),
            'Rank'
        ].values[0]
    
    # monthly or annual
    else:
    
        # get temp clust method and model
        temp_clustmeth = df_rankedma['clust_method'].unique()[i]
        temp_model = df_rankedma['model'].unique()[j]

        # populate df_plot
        df_plot.iloc[i, j] = df_rankedma.loc[
            (df_rankedma['clust_method'] == temp_clustmeth) &\
            (df_rankedma['model'] == temp_model) &\
                (df_rankedma['time_scale'] == timescale_in),
            'Rank'
        ].values[0]
    
# plot heatmap

# Plot the correlogram heatmap
plt.figure(figsize=(3, 6))
ax = sns.heatmap(df_plot.astype(int),
            annot = True, # show ami value
            fmt = '.0f', # '.2f'two decimal places
            cmap = colors_in,
            vmin = 1,
            vmax = 15,
            annot_kws = {'fontsize': 10})

# cbar tick locations
# cbar_tickloc = map(str, list(np.arange(0.5, 15.5, 1)))

# # make cbar labels larger
# # cbar = ax.collections[0].colorbar
# cbar = plt.colorbar(ticks = range(15))
# # cbar.set_ticks(15)
# plt.clim(-0.5, 15)
# cbar.set_ticklabels(list(range(0, 15)))
# cbar.ax.tick_params(labelsize = 10)
# # cbar.ax.set_yticklabels(range(1, 16))
# cbar.set_label('Rank', fontsize = 10)


# rotate x tick lables
ax.tick_params(axis = 'x', rotation = 45)
# make tick labels larger
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10)


if drop_noise:
    title_in = 'Rank of Regionalization Approach\n'\
              f'({timescale_in} {trainval_temp} data) - Noise Removed'
    fig_name = f'{dir_figs}/RegionalizationRank_{timescale_in}_DropNoise'\
                f'{trainval_temp} Data.png'
else:
    title_in = 'Rank of Regionalization Approach\n'\
              f'({timescale_in} {trainval_temp}  Data) - W/Noise'
    fig_name = f'{dir_figs}/RegionalizationRank_{timescale_in}_{trainval_temp} Data.png'  
plt.title(title_in, fontsize = 12)



if write_hmfig:
    plt.savefig(fig_name,
                bbox_inches = 'tight',
                dpi = 300)
else:
    plt.show()