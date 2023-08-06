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
import string
# import os


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
# which metric to plot eCDFs for, and to use when ranking models
# for time series plots, mean_annual always uses residuals
metrics_in = ['|residuals|', 'KGE', 'NSE']
# drop noise (True or False)
drop_noises = [True, False]

# which metric to plot in eCDF plots?
ecdf_metrin = 'KGE'


# should heat map of performance metrics and rank be saved?
write_hmfig = True
# should ecdfs be svaed?
write_ecdf = True



# %% Run loop to get all ranks and scores by clustering method
################################



# read in main 
df_moann = pd.read_csv(f'{dir_work}/PerfRankMonthAnnual_ByClustMethModel.csv')
df_ma = pd.read_csv(f'{dir_work}/PerfRankMeanAnnual_ByClustMethModel.csv')

df_work = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
                            dtype = {'STAID': 'string'})
df_workma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
                            dtype = {'STAID': 'string'})

df_work['region'] = df_work['clust_method'].str.cat(\
            df_work['region'].astype(str), sep = '_')
df_workma['region'] = df_work['clust_method'].str.cat(\
            df_work['region'].astype(str), sep = '_')

# %% Loop through combos of variables and make some plots
#########################################

# turn into for loop producing plots for each combo of variables above
# for trainval_in, timescale_in, model_in, metric_in, drop_noise in \
#     product(trainvals_in, timescales_in, models_in, \
#             metrics_in, drop_noises):
#     print(trainval_in, timescale_in, model_in, metric_in, drop_noise)
for trainval_in, timescale_in, metric_in,in \
    product(trainvals_in, timescales_in, \
            metrics_in,):
    print(trainval_in, timescale_in, metric_in)


    # edit string to use in title to test if file already exsits
    if trainval_in == 'train':
        trainval_temp = 'train'
    else:
        trainval_temp = 'test'

    if timescale_in in ['monthly', 'annual']:
        if metric_in == '|residuals|':
            continue
        df_ranked = df_moann.copy()
        metric_temp = metric_in
    else:
        if metric_in in ['NSE', 'KGE']:
            continue
        df_ranked = df_ma.copy()
        metric_temp = 'residuals'

    # define file name for ranking performance based on current set of data
    fig_name = f'{dir_figs}/Regionalization_{metric_temp}_{timescale_in}'\
                f'{trainval_temp}.png'


    # %% get heatmap presenting order of performance for various models and timescales
    #######################################################

    # define file name for ranking performance based on current set of data
    title_in = f'Rank of Regionalization Approach Based on {metric_temp} and '\
                f'{timescale_in} {trainval_temp}ing data'

    df_plotDrop = pd.DataFrame('', 
                        index = df_ranked['clust_method'].unique(), 
                        columns = df_ranked['model'].unique()
                        )
    
    df_plotWnoise = pd.DataFrame('', 
                        index = df_ranked['clust_method'].unique(), 
                        columns = df_ranked['model'].unique()
                        )
    
    # create dataframe to hold annotations of perf metric
    df_annotDrop = pd.DataFrame('', 
                        index = df_ranked['clust_method'].unique(), 
                        columns = df_ranked['model'].unique()
                        )
    df_annotWnoise = pd.DataFrame('', 
                        index = df_ranked['clust_method'].unique(), 
                        columns = df_ranked['model'].unique()
                        )


    # define colors to use
    # colors_in = [
    #     'orangered', 'orange', 'sandybrown', 'darkkhaki', 'yellow',  
    #     'yellowgreen', 'springgreen', 'aquamarine', 'green', 'blue', 
    #     'darkblue', 'purple', 'blueviolet', 'orchid', 'deeppink'
    # ]
    # colors_in = 'Greys_r'
    colors_in = 'pink'

    # define order for y- and x-axes
    yaxis_order = [
        'None', 'Class', 'CAMELS',
        'AggEcoregion', 'HLR',
        'All_0', 'All_1', 'All_2',
        'Anth_0', 'Anth_1',
        'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4'
    ]

    xaxis_order = [
        'regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'
    ]
    
    for i, j in product(range(len(df_ranked['clust_method'].unique())),
                range(len(df_ranked['model'].unique()))):
        
        # print(i,j)

        # monthly or annual
        # if (timescale_in == 'annual') | (timescale_in == 'monthly'):
        
        # get temp clust method and model
        temp_clustmeth = df_ranked['clust_method'].unique()[i]
        temp_model = df_ranked['model'].unique()[j]

        # populate df_plotDrop
        df_plotDrop.iloc[i, j] = df_ranked.loc[
            (df_ranked['clust_method'] == temp_clustmeth) &\
            (df_ranked['model'] == temp_model) &\
                (df_ranked['time_scale'] == timescale_in) &\
                    (df_ranked['dropNoise'] == True) &\
                        (df_ranked['train_val'] == trainval_in),
            f'{metric_in}_rank'
        ].values[0]

        df_plotWnoise.iloc[i, j] = df_ranked.loc[
            (df_ranked['clust_method'] == temp_clustmeth) &\
            (df_ranked['model'] == temp_model) &\
                (df_ranked['time_scale'] == timescale_in) &\
                    (df_ranked['dropNoise'] == False) &\
                        (df_ranked['train_val'] == trainval_in),
            f'{metric_in}_rank'
        ].values[0]

        # if mean_annual, make sure using smaller as better
        if timescale_in == 'mean_annual':
            temp_df = df_ranked.loc[
                (df_ranked['model'] == temp_model) &\
                (df_ranked['time_scale'] == timescale_in) &\
                (df_ranked['dropNoise'] == True) &\
                (df_ranked['train_val'] == trainval_in)
            ]
            df_plotDrop.iloc[i, j] = temp_df[f'{metric_in}_rank'].max()+1 - df_plotDrop.iloc[i, j]

            temp_df = df_ranked.loc[
                (df_ranked['model'] == temp_model) &\
                (df_ranked['time_scale'] == timescale_in) &\
                (df_ranked['dropNoise'] == False) &\
                (df_ranked['train_val'] == trainval_in)
            ]
            df_plotWnoise.iloc[i, j] = temp_df[f'{metric_in}_rank'].max()+1 - df_plotWnoise.iloc[i, j]

        # populate df_plotDrop
        df_annotDrop.iloc[i, j] = df_ranked.loc[
            (df_ranked['clust_method'] == temp_clustmeth) &\
            (df_ranked['model'] == temp_model) &\
            (df_ranked['time_scale'] == timescale_in) &\
            (df_ranked['dropNoise'] == True) &\
                        (df_ranked['train_val'] == trainval_in),
            f'{metric_in}_qmean'
        ].values[0]
        
        df_annotWnoise.iloc[i, j] = df_ranked.loc[
            (df_ranked['clust_method'] == temp_clustmeth) &\
            (df_ranked['model'] == temp_model) &\
            (df_ranked['time_scale'] == timescale_in) &\
            (df_ranked['dropNoise'] == False) &\
            (df_ranked['train_val'] == trainval_in),
            f'{metric_in}_qmean'
        ].values[0]
        
        
    # reorder df_plotDrop to match yaxis_order define just above
    df_plotDrop = df_plotDrop.reindex(index = yaxis_order, columns = xaxis_order)
    df_plotWnoise = df_plotWnoise.reindex(index = yaxis_order, columns = xaxis_order)
    df_annotDrop = df_annotDrop.reindex(index = yaxis_order, columns = xaxis_order)
    df_annotWnoise = df_annotWnoise.reindex(index = yaxis_order, columns = xaxis_order)
    
    # get difference dataframe
    # if mean annual, make sure smaller is better
    df_diff = (df_annotDrop - df_annotWnoise).abs().rank(ascending = False)
    df_temp = df_work[df_work['region'].str.contains('-1')]
    df_noiseCnt = df_temp.groupby(['clust_method', 'train_val']).\
        count()['STAID'].reset_index()

    df_annotdiff = df_annotDrop - df_annotWnoise
    df_annotdiff = df_annotdiff.astype(float).round(2)
    df_annotdiff.iloc[0:5] = '-'
    df_annotdiff = df_annotdiff.astype(str)

    df_annotDrop = df_annotDrop.astype(float).round(2).astype(str)
    df_annotWnoise = df_annotWnoise.astype(float).round(2).astype(str)
        
    # plot heatmap

    # Plot the correlogram heatmap
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6), sharey = True)
    sns.heatmap(df_plotDrop.astype(int),
                annot = df_annotDrop, # show ami value
                fmt = 's', # '.02f', # '.2f'two decimal places
                cmap = colors_in,
                cbar = False,
                vmin = 1,
                vmax = 15,
                annot_kws = {'fontsize': 9},
                ax = ax1)
    
    sns.heatmap(df_plotWnoise.astype(int),
                annot = df_annotWnoise, # show ami value
                fmt = 's', #  '.02f', # '.2f'two decimal places
                cmap = colors_in,
                cbar = False,
                vmin = 1,
                vmax = 15,
                annot_kws = {'fontsize': 9},
                ax = ax2)
    
    sns.heatmap(df_diff.astype(int),
                annot = df_annotdiff, # show ami value
                fmt = 's',#'.2f'two decimal places
                cmap = colors_in,
                vmin = 1,
                vmax = 15,
                annot_kws = {'fontsize': 9},
                ax = ax3)
    

    # cbar tick locations
    # cbar_tickloc = map(str, list(np.arange(0.5, 15.5, 1)))

    # # make cbar labels larger
    cbar = ax3.collections[0].colorbar
    cbar.set_label('Within-Model Rank')
    # cbar = plt.colorbar(ticks = range(15))
    # # cbar.set_ticks(15)
    # plt.clim(-0.5, 15)
    # cbar.set_ticklabels(list(range(0, 15)))
    # cbar.ax.tick_params(labelsize = 10)
    # # cbar.ax.set_yticklabels(range(1, 16))
    # cbar.set_label('Rank', fontsize = 10)

    # add titles to individual plots
    ax1.set_title('Noise Dropped', fontsize = 10)
    ax2.set_title('With Noise', fontsize = 10)
    ax3.set_title('Noise Dropped-With Noise', fontsize = 10)


    # rotate x tick labels
    for ax in (ax1, ax2, ax3):
        ax.set_xticklabels(
            labels = xaxis_order,
            rotation = 45,
            ha = 'right',
            rotation_mode = 'anchor'
        )
    # ax.tick_params(axis = 'x', rotation = 45)
    # make tick labels larger
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10)
 
    plt.suptitle(title_in, fontsize = 12)



    if write_hmfig:
        plt.savefig(fig_name,
                    bbox_inches = 'tight',
                    dpi = 300)
    else:
        plt.show()

    # plt.close()


    # %% eCDF plots
    ###############################
# %% Loop through combos of variables and make ecdf plts
#########################################

models_plot = models_in.copy()

# turn into for loop producing plots for each combo of variables above
for trainval_in, drop_noise in \
    product(trainvals_in, drop_noises):
    print(trainval_in, drop_noise)

    # edit string to use in title to test if file already exsits
    if trainval_in == 'train':
        trainval_temp = 'train'
    else:
        trainval_temp = 'test'

    # mean annual
    data_in_mannual = df_workma[
        df_workma['train_val'] == trainval_in
    ]

    data_in_mannual['|residuals|'] = np.abs(data_in_mannual['residuals'])

    data_in_mannual.sort_values(
        by = ['clust_method', 'region'], inplace = True
        )


    # annual
    data_in_annual = df_work[
        (df_work['train_val'] == trainval_in) &\
        (df_work['time_scale'] == 'annual')
    ]


    data_in_annual.sort_values(
        by = ['clust_method', 'region'], inplace = True
        )


    # month
    data_in_month = df_work[
        (df_work['train_val'] == trainval_in) &\
        (df_work['time_scale'] == 'monthly')
    ]

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

    palette_in = 'Paired'
    # palette_in = 'gist_ncar'
    data_in_mannual = data_in_mannual[
        data_in_mannual['clust_method'].isin(clust_meth_in)
    ]

    data_in_annual = data_in_annual[
        data_in_annual['clust_method'].isin(clust_meth_in)
    ]

    data_in_month = data_in_month[
        data_in_month['clust_method'].isin(clust_meth_in)
    ]

    # if drop_noise, drop -1 regions
    if drop_noise:
        data_in_mannual = data_in_mannual[~data_in_mannual['region'].str.contains('-1')]
        data_in_annual = data_in_annual[~data_in_annual['region'].str.contains('-1')]
        data_in_month = data_in_month[~data_in_month['region'].str.contains('-1')]

    ####

    # define order for hues
    hue_order = [
        'None', 'Class', 'CAMELS',
        'AggEcoregion', 'HLR',
        'All_0', 'All_1', 'All_2',
        'Anth_0', 'Anth_1',
        'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4'
    ]

    ls_in = ':' # [':', '--', '-.']
    lw_in = 2

    ## %% make plot

    fig, axs = plt.subplots(
        4, 5, figsize = (10, 12), sharey = True, sharex = 'col'
        )

    for cnt, ax in enumerate(axs.flatten()):

        if cnt == 19:
            legend_on = True
        else:
            legend_on = False

        # set model to work with, to label y-axes with
        if cnt in [0, 1, 2, 3, 4]:
            model_plot = models_plot[0]
        elif cnt in [5, 6, 7, 8, 9]:
            model_plot = models_plot[1]
        elif cnt in [10, 11, 12, 13, 14]:
            model_plot = models_plot[2]
        elif cnt in [15, 16, 17, 18, 19]:
            model_plot = models_plot[3]

        # define parameters for plot
        if cnt in [0, 5, 10, 15]:
            xlabel = '|residuals| [cm]'
            xlims = [0, 50]
            xticks = [0, 10, 20, 30, 40]            
            xin = '|residuals|'
            data_in = data_in_mannual
            ylabel_in = f'Percentile - {model_plot}'
            annot_xy = (42, 0.02)
            
        elif cnt in [1, 6, 11, 16]:
            xlabel = 'KGE'
            xlims = [-1, 1]
            xticks = [-1, -0.5, 0, 0.5, 1]
            xin = 'KGE'
            data_in = data_in_annual
            ylabel_in = ''
            annot_xy = (0.65, 0.02)

        elif cnt in [2, 7, 12, 17]:
            xlabel = 'NSE'
            xlims = [-1, 1]
            xticks = [-1, -0.5, 0, 0.5, 1]
            xin = 'NSE'
            data_in = data_in_annual
            ylabel_in = ''
            annot_xy = (0.65, 0.02)

        elif cnt in [3, 8, 13, 18]:
            xlabel = 'KGE'
            xlims = [-1, 1]
            xticks = [-1, -0.5, 0, 0.5, 1]
            xin = 'KGE'
            data_in = data_in_month
            ylabel_in = ''
            annot_xy = (0.65, 0.02)

        elif cnt in [4, 9, 14, 19]:
            xlabel = 'NSE'
            xlims = [-1, 1]
            xticks = [-1, -0.5, 0, 0.5, 1]
            xin = 'NSE'
            data_in = data_in_month
            ylabel_in = ''
            annot_xy = (0.65, 0.02)

        if cnt == 0:
            axtitle = 'Mean Annual'
        elif cnt in [1, 2]:
            axtitle = 'Annual'
        elif cnt in [3, 4]:
            axtitle = 'Monthly'
        else:
            axtitle = ''


        data_in = data_in[data_in['model'] == model_plot]


        ax.set(xlabel = xlabel, 
                ylabel = ylabel_in)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xticks(xticks)
        ax.annotate(f'({string.ascii_lowercase[cnt]})', xy = annot_xy)
        ax.grid()
        ax.title.set_text(axtitle)
        ecdf = sns.ecdfplot(
            data = data_in,
            x = xin,
            hue = 'clust_method', # 'region',
            linestyle = ls_in,
            linewidth = lw_in,
            palette = palette_in,
            ax = ax,
            legend = legend_on,
            hue_order = hue_order
        )

        if cnt in [0, 5, 10, 15]:
            ax.set_xlim(xlims[0], xlims[1])



    # build legend
    legend = ax.get_legend()
    ax.get_legend().remove()
    # legend.get_frame().set_alpha(0)
    plt.legend(handles = legend.legendHandles,
        labels = [t.get_text() for t in legend.get_texts()],
        loc = 'upper right', 
        bbox_to_anchor = (0.2, -0.25),
        ncol = 5,
        framealpha = 0
    )

    if drop_noise:
        title_in = f'eCDFs: {trainval_temp} data-Noise Removed'
        fig_name = f'{dir_figs}/ecdfs_{trainval_temp}'\
                    '_dropNoise.png'
    else:
        title_in = f'eCDFs: {trainval_temp} data-W/Noise'
        fig_name = f'{dir_figs}/ecdfs_{trainval_temp}'\
                        '_WNoise.png'  

    plt.suptitle(
        title_in,
        x = 0.5,
        y = 0.92
    )

    plt.tight_layout() # rect = [0, 0.0, 1, 0.96])
    
    # save fig
    if write_ecdf:
        plt.savefig(
            fig_name, 
            dpi = 300,
            bbox_inches = 'tight'
            )

    else:
        plt.show()
    # plt.close()



    # ## %% make plot

    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
    #     1, 5, figsize = (10, 3.5), sharey = True
    #     )
    
    # ls_in = ':'
    # lw_in = 4

    # ax1.set(xlabel = '|residuals| [cm]', 
    #         ylabel = 'Non-Exceedence Probability')
    # ax1.set_xlim(0, 50)
    # ax1.set_xticks([0, 10, 20, 30, 40])
    # ax1.annotate('(a)', xy = (42, 0.02))
    # ax1.grid()
    # ax1.title.set_text('Mean Annual')
    # ecdf = sns.ecdfplot(
    #     data = data_in_mannual,
    #     x = '|residuals|',
    #     hue = 'clust_method', # 'region',
    #     linestyle = ls_in,
    #     linewidth = lw_in,
    #     palette = palette_in,
    #     ax = ax1
    # )

    # ax2.set_xlim(-1, 1)
    # ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
    # ax2.set(xlabel = 'KGE')
    # ax2.annotate('(b)', xy = (0.65, 0.02))
    # ax2.grid()
    # ax2.title.set_text('Annual')
    # sns.ecdfplot(
    #     data = data_in_annual,
    #     x = 'KGE',
    #     hue = 'clust_method', # 'region',
    #     linestyle = ls_in,
    #     linewidth = lw_in,
    #     palette = palette_in,
    #     ax = ax2,
    #     legend = False
    # )


    # ax3.set_xlim(-1, 1)
    # ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
    # ax3.set(xlabel = 'NSE')
    # ax3.annotate('(c)', xy = (0.65, 0.02))
    # ax3.grid()
    # ax3.title.set_text('Annual')
    # sns.ecdfplot(
    #     data = data_in_annual,
    #     x = 'NSE',
    #     hue = 'clust_method', # 'region',
    #     linestyle = ls_in,
    #     linewidth = lw_in,
    #     palette = palette_in,
    #     ax = ax3,
    #     legend = False
    # )


    # ax4.set_xlim(-1, 1)
    # ax4.set_xticks([-1, -0.5, 0, 0.5, 1])
    # ax4.set(xlabel = 'KGE')
    # ax4.annotate('(d)', xy = (0.65, 0.02))
    # ax4.grid()
    # ax4.title.set_text('Monthly')
    # sns.ecdfplot(
    #     data = data_in_month,
    #     x = 'KGE',
    #     hue = 'clust_method', # 'region',
    #     linestyle = ls_in,
    #     linewidth = lw_in,
    #     palette = palette_in,
    #     ax = ax4,
    #     legend = False
    # )

    # ax5.set_xlim(-1, 1)
    # ax5.set_xticks([-1, -0.5, 0, 0.5, 1])
    # ax5.set(xlabel = 'NSE')
    # ax5.annotate('(e)', xy = (0.65, 0.02))
    # ax5.grid()
    # ax5.title.set_text('Monthly')
    # sns.ecdfplot(
    #     data = data_in_month,
    #     x = 'NSE',
    #     hue = 'clust_method', # 'region',
    #     linestyle = ls_in,
    #     linewidth = lw_in,
    #     palette = palette_in,
    #     ax = ax5,
    #     legend = False
    # )

    # # build legend
    # legend = ax1.get_legend()
    # ax1.get_legend().remove()
    # # legend.get_frame().set_alpha(0)
    # plt.legend(handles = legend.legendHandles,
    #     labels = [t.get_text() for t in legend.get_texts()],
    #     loc = 'upper right', 
    #     bbox_to_anchor = (0.2, -0.15),
    #     ncol = 5,
    #     framealpha = 0
    # )

    # if drop_noise:
    #     title_in = f'eCDFs: {model_in} on {trainval_temp} data-Noise Removed'
    #     fig_name = f'{dir_figs}/ecdfs_{model_in}_{trainval_temp}'\
    #                 '_dropNoise.png'
    # else:
    #     title_in = f'eCDFs: {model_in} on {trainval_temp} data-W/Noise'
    #     fig_name = f'{dir_figs}/ecdfs_{model_in}_{trainval_temp}'\
    #                     '_WNoise.png'  

    # plt.suptitle(
    #     title_in
    # )

    
    # # save fig
    # if write_ecdf:
    #     plt.savefig(
    #         fig_name, 
    #         dpi = 300,
    #         bbox_inches = 'tight'
    #         )

    # else:
    #     plt.show()
    # # plt.close()












# %% regress gains in performance with noise removed against # catchments dropped
#############################


df_temp = df_work[df_work['region'].str.contains('-1')]
df_noiseCnt = df_temp.groupby(['clust_method', 'train_val']).\
    count()['STAID'].reset_index()


df_moann_drop = df_moann[df_moann['dropNoise']]
df_moann_with = df_moann[~df_moann['dropNoise']]
df_moannGain = pd.DataFrame({
    'clust_method': df_moann_with['clust_method'].reset_index(drop = True),
    'model': df_moann_with['model'].reset_index(drop = True),
    'train_val': df_moann_with['train_val'].reset_index(drop = True),
    'time_scale': df_moann_with['time_scale'].reset_index(drop = True),
    'KGEgain': df_moann_drop['KGE_qmean'].reset_index(drop = True) - \
        df_moann_with['KGE_qmean'].reset_index(drop = True),
    'NSEgain': df_moann_drop['NSE_qmean'].reset_index(drop = True) - \
        df_moann_with['NSE_qmean'].reset_index(drop = True)
})

df_plot = pd.merge(
    df_moannGain, df_noiseCnt,
    on = ['clust_method', 'train_val']
)

df_plot.columns = df_plot.columns.str.replace('STAID', 'Noise')


sns.scatterplot(data = df_plot,
             x = 'Noise',
             y = 'KGEgain',
             hue = 'model',
             style = 'train_val')

plt.grid()

plt.show()


####################


# show which variables were most responsive to identifying noise
sns.boxplot(data = df_plot,
             x = 'train_val',
             y = 'KGEgain',
             hue = 'model')

plt.grid()

plt.show()


####################

df_plot2 = df_plot.copy()

df_plot2['clust_method'] = [
    f'{x} ({y})' for x, y in zip(df_plot2['clust_method'], df_plot2['Noise'])
]

# show which variables were most responsive to identifying noise
sns.boxplot(data = df_plot2[df_plot2['train_val'] == 'train'],
             x = 'model',
             y = 'KGEgain',
             hue = 'clust_method',
             linewidth = 1)

plt.grid()

plt.legend(ncol = 2)

plt.show()


####################

df_plot2 = df_plot.copy()

df_plot2['clust_method'] = [
    f'{x} ({y})' for x, y in zip(df_plot2['clust_method'], df_plot2['Noise'])
]

# show which variables were most responsive to identifying noise
sns.boxplot(data = df_plot2[df_plot2['train_val'] == 'valnit'],
             x = 'model',
             y = 'KGEgain',
            #  hue = 'clust_method',
             linewidth = 1)

plt.grid()

plt.legend(ncol = 2)

plt.show()


####################

df_plot2 = df_work.copy()

df_plot2 = df_plot2[df_plot2['region'].str.contains('-1')]

# df_plot2['clust_method'] = [
#     f'{x} ({y})' for x, y in zip(df_plot2['clust_method'], df_plot2['Noise'])
# ]

# show which variables were most responsive to identifying noise
ax = sns.boxplot(data = df_plot2[df_plot2['train_val'] == 'valnit'],
             x = 'model',
             y = 'KGE',
             hue = 'clust_method',
             linewidth = 1)

ax.set_ylim(-5, 1)
plt.grid()

plt.legend(ncol = 2)

plt.show()

# df_moann_wide = pd.melt(df_moann,
#                         id_vars = ['model', 'train_val', 'clust_method'],
#                         value_vars = ['meanKGE', 'meanNSE'])
# df_moann_wide = df_moann.pivot_table(
#     index = ['clust_method', 'model', 'train_val', 'time_scale'],
#     columns = 'dropNoise',
#     values =  ['meanNSE', 'meanKGE']
# ).reset_index()

# df_moann_wide.columns = df_moann_wide.columns.str.replace('_x', '')

# # %% get heatmap presenting performance gains for various models and timescales
# #######################################################

# # define file name for ranking performance based on current set of data
# if drop_noise:
#     # title_in = 'Rank of Regionalization Approach\n'\
#     #             f'Based on {metric_temp}\n'\
#     #             f'({timescale_in} {trainval_temp} data-Noise Removed)'

#     title_in = f'Gain in mean{metric_temp}q from removing noise \n'\
#                 'by Regionalization Approach\n'\
#                 f'({timescale_in} {trainval_temp} data-Noise Removed)'
# else:
#     title_in = f'Gain in mean{metric_temp}q from removing noise \n'\
#                 f'by Regionalization Approach\n'\
#                 f'({timescale_in} {trainval_temp} data-W/Noise)'

# df_plot = pd.DataFrame('', 
#                     index = df_ranked['clust_method'].unique(), 
#                     columns = df_ranked['model'].unique()
#                     )
# # create dataframe to hold annotations of perf metric
# df_annotDrop = pd.DataFrame('', 
#                     index = df_ranked['clust_method'].unique(), 
#                     columns = df_ranked['model'].unique()
#                     )


# # define colors to use
# # colors_in = [
# #     'orangered', 'orange', 'sandybrown', 'darkkhaki', 'yellow',  
# #     'yellowgreen', 'springgreen', 'aquamarine', 'green', 'blue', 
# #     'darkblue', 'purple', 'blueviolet', 'orchid', 'deeppink'
# # ]
# # colors_in = 'Greys_r'
# colors_in = 'pink'

# # define order for y- and x-axes
# yaxis_order = [
#     'None', 'Class', 'CAMELS',
#     'AggEcoregion', 'HLR',
#     'All_0', 'All_1', 'All_2',
#     'Anth_0', 'Anth_1',
#     'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4'
# ]

# xaxis_order = [
#     'regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'
# ]

# for i, j in product(range(len(df_ranked['clust_method'].unique())),
#             range(len(df_ranked['model'].unique()))):

#     # monthly or annual
#     if (timescale_in == 'annual') | (timescale_in == 'monthly'):
    
#         # get temp clust method and model
#         temp_clustmeth = df_ranked['clust_method'].unique()[i]
#         temp_model = df_ranked['model'].unique()[j]

#         # populate df_plot
#         temp1 = df_ranked.loc[
#             (df_ranked['clust_method'] == temp_clustmeth) &\
#             (df_ranked['model'] == temp_model) &\
#                 (df_ranked['time_scale'] == timescale_in),
#             'Rank'
#         ].values[0]

#         df_plot.iloc[i, j] = 

#         # populate df_plot
#         df_annotDrop.iloc[i, j] = df_ranked.loc[
#             (df_ranked['clust_method'] == temp_clustmeth) &\
#             (df_ranked['model'] == temp_model) &\
#                 (df_ranked['time_scale'] == timescale_in),
#             f'mean{metric_in}'
#         ].values[0]
    
#     # monthly or annual
#     else:
    
#         # get temp clust method and model
#         temp_clustmeth = df_rankedma['clust_method'].unique()[i]
#         temp_model = df_rankedma['model'].unique()[j]

#         # populate df_plot
#         df_plot.iloc[i, j] = df_rankedma.loc[
#             (df_rankedma['clust_method'] == temp_clustmeth) &\
#             (df_rankedma['model'] == temp_model) &\
#                 (df_rankedma['time_scale'] == timescale_in),
#             'Rank'
#         ].values[0]

#         # populate df_plot
#         df_annotDrop.iloc[i, j] = df_rankedma.loc[
#             (df_rankedma['clust_method'] == temp_clustmeth) &\
#             (df_rankedma['model'] == temp_model) &\
#                 (df_rankedma['time_scale'] == timescale_in),
#             'meanRes'
#         ].values[0]

# # reorder df_plot to match yaxis_order define just above
# df_plot = df_plot.reindex(index = yaxis_order, columns = xaxis_order)
# df_annotDrop = df_annotDrop.reindex(index = yaxis_order, columns = xaxis_order)
    
# # plot heatmap

# # Plot the correlogram heatmap
# plt.figure(figsize=(3, 6))
# ax = sns.heatmap(df_plot.astype(int),
#             annot = df_annotDrop, # show ami value
#             fmt = '.02f', # '.2f'two decimal places
#             cmap = colors_in,
#             vmin = 1,
#             vmax = 15,
#             annot_kws = {'fontsize': 10})

# # cbar tick locations
# # cbar_tickloc = map(str, list(np.arange(0.5, 15.5, 1)))

# # # make cbar labels larger
# cbar = ax.collections[0].colorbar
# cbar.set_label('Within-Model Rank')
# # cbar = plt.colorbar(ticks = range(15))
# # # cbar.set_ticks(15)
# # plt.clim(-0.5, 15)
# # cbar.set_ticklabels(list(range(0, 15)))
# # cbar.ax.tick_params(labelsize = 10)
# # # cbar.ax.set_yticklabels(range(1, 16))
# # cbar.set_label('Rank', fontsize = 10)


# # rotate x tick lables
# ax.set_xticklabels(
#     labels = xaxis_order,
#     rotation = 45,
#     ha = 'right',
#     rotation_mode = 'anchor'
# )
# # ax.tick_params(axis = 'x', rotation = 45)
# # make tick labels larger
# plt.xticks(fontsize = 10)
# plt.yticks(fontsize = 10)

# plt.title(title_in, fontsize = 12)



# if write_hmfig:
#     plt.savefig(fig_name,
#                 bbox_inches = 'tight',
#                 dpi = 300)
# else:
#     plt.show()

# plt.close()






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