'''
2023/09/10 BChoat

Code to explore/identify the best performing regionalization and/or predictive
model combos (RPMs) based on:

- the number of catchments that were best predicted by the RPMs.


'''



# %% Import libraries
##############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import product
import os


# %% Define input variables, directories and such
###########################################

# directory with input files
dir_in = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results'

# directory where to place output figs
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/Rank_Count'
if not os.path.exists(dir_figs): os.mkdir(dir_figs)

# define which timescales at which to process figures
# ('mean_annual', 'annual', 'monthly')
timescales = ['mean_annual', 'annual', 'monthly']

# which metric to work with ('KGE' or 'NSE', only applies to timescales
# other than 'mean_annual')
metrics_in = ['KGE', 'NSE'] # , 'r']

# which model(s) to work with. Provide as list of lists even if only one
# ['XGBoost', 'strd_pca_mlr', 'strd_mlr', 'regr_precip']
models_in = [['XGBoost'], ['strd_mlr'], ['regr_precip']] #  'strd_PCA_mlr','XGBoost', 

# which partition of data to work with. Provide as list even if only one
#['train', 'valnit']
parts_in = [['train'], ['valnit']]

# write figs to file (True or False)?
write_figs = False


# %% read in data
#########################################

# mean annual
ma_results = pd.read_csv(
                f'{dir_in}/PerfMetrics_MeanAnnual.csv',
                dtype = {'STAID': 'string'}
            )

ma_results['abs(residuals)'] = ma_results.residuals.abs()

ma_results = ma_results[[
                'STAID', 'abs(residuals)', 'clust_method', 'region',\
                            'model', 'time_scale', 'train_val'
            ]]

# annual, montly
annmy_results = pd.read_csv(
                f'{dir_in}/NSEComponents_KGE.csv',
                dtype = {'STAID': 'string'}
            )


# %% Loop through timescales and process figures

# define order of inputs
order_in = ['None', 'Class', 'CAMELS', 'AggEcoregion', 'HLR',
            'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1',
            'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4']

# create counters to track position
mod_count = 0
time_count = 0

# set intial values to compare and update new values
init_time = timescales[0]
init_model = models_in[0]
init_part = parts_in[0]
init_metr = metrics_in[0]

for part_in, metric_in in product(parts_in, metrics_in):


    init_part = part_in

    df_work = annmy_results[
        (annmy_results['train_val'].isin(part_in))
    ]
    df_work = df_work[[
        'STAID', metric_in, 'clust_method', 'region',\
        'model', 'time_scale'
    ]]

    ma_work = ma_results.rename({'abs(residuals)': metric_in}, axis = 1)
    ma_work = ma_work['train_val'].isin(part_in)
    ma_work = ma_work.drop(columns = 'train_val')

    df_work = pd.concat([df_work, ma_work], axis = 0)

    fig, axs = plt.subplots(3, 3, sharex = True, sharey = True)
    for i, ax in enumerate(axs.flatten()):
    # for timescale, model_in in \
        # product(timescales, models_in): # parts_in,
        print(f'\n\n{timescale, metric_in, part_in}\n\n')
        # print(i)
        if i in range(0, 3):
            timescale = 'mean_annual'
        elif i in range(3, 6):
            timescale = 'annual'
        else:
            timescale = 'monthly'

        if i in [0, 3, 6]:
            model_in = 'regr_precip'
        elif i in [1, 4, 7]:
            model_in = 'strd_mlr'
        else:
            model_in = 'XGBoost'

        if i in [6, 7, 8]:
            xlabel = 'Cluster Method'
        else:
            xlabel = ''
        
        if i == 0:
            title_in = 'SLR'
            ylabel = 'mean annual\ncount'
        elif i == 1:
            title_in = 'MLR'
            ylabel = ''
        elif i == 2:
            title_in = 'XGBoost'
        elif i == 3:
            ylabel = 'annual\ncount'
            title_in = ''
        elif i == 6:
            ylabel = 'monthly\ncount'
        else:
            title_in = ''
            ylabel = ''
        

        print(f'\n\n\n {timescale}, {part_in}, {metric_in}, {model_in}\n\n\n') #

        if timescale ==  'mean_annual':
            metric_temp = '|residuals|'

            df_plot = df_work[df_work['model'] == model_in]
            # df_plot = df_work[df_work['train_val'].isin(part_in)]

            max_temp = df_plot.groupby('STAID')[metric_in].min().reset_index()

            ind_max = pd.merge(
                max_temp, df_plot, 
                on = ['STAID', metric_in], how = 'left'
            )

        elif timescale != 'mean_annual':
            metric_temp = metric_in

            df_plot = df_work[df_work['time_scale'] == timescale]
            # df_plot = df_plot[df_plot['train_val'].isin(part_in)]
            df_plot = df_plot[df_plot['model'] == model_in]

            max_temp = df_plot.groupby('STAID')[metric_in].max().reset_index()

            ind_max = pd.merge(
                max_temp, df_plot, 
                on = ['STAID', metric_in], how = 'left'
            )
        
        else:
            continue

        # print(ind_max.groupby('clust_method').count()['STAID'])

        # # count plots

        # define some inputs for plots
        models_name = '_'.join(str(j) for j in model_in)
        parts_name = '_'.join(str(j) for j in part_in)
        if part_in == ['train']:
            part_name = 'training data'
        elif part_in == ['valnit']:
            part_name = 'testing data'
        elif part_in == ['train', 'valnit']:
            part_name = 'all data'

        # clust_method counts
        # xlabels = ind_max['clust_method'].value_counts().index.values
        hue_order  = ind_max['model'].value_counts().index.values
        g = sns.countplot(
            data = ind_max.sort_values(by = 'clust_method'), 
            x = 'clust_method', 
            # hue = 'model',
            order = order_in,
            hue_order = hue_order,
            ax = ax
            )
        g.set_xticklabels(order_in, 
                        rotation = 45, 
                        ha = 'right', 
                        rotation_mode = 'anchor',
                        fontsize = 8
                        )
        # ax.set_title(
        #     f'Number of {timescale} models performing best\nbased on {metric_temp} & {part_name}'
        #     )
        g.set(
            title = title_in,
            # title = \
            #     f'{model_in[0]}: Number of {timescale} models performing best\n'\
            #         f'based on {metric_temp} & {part_name}',
            xlabel = xlabel,
            ylabel = ylabel
        )
        # if 'train' in part_in:
        #     g.set_ylim([0, 750])
        # else:
        #     g.set_ylim([0, 350])
        plt.legend([],[], frameon = False)
        # legend = ax.legend(title = 'Cluster Method', loc = 'upper right', ncol = 3, frameon = True)
        # legend.get_frame().set_alpha(0.5)

        plt.suptitle(
            f'{part_in}: {metric_in}'
        )

    if write_figs:
        plt.savefig(
            f'{dir_figs}/NumbCatchBestPredicted_{metric_in}_{timescale}_{parts_name}_{models_name}.csv',
            dpi = 300,
            bbox_inches = 'tight')
    else:
        plt.show()


        # model counts
        # xlabels = ind_max['model'].value_counts().index.values
        # hue_order  = ind_max['clust_method'].value_counts().index.values
        # ax = sns.countplot(
        #     data = ind_max.sort_values(by = 'model'), 
        #     x = 'model', 
        #     hue = 'clust_method',
        #     order = xlabels,
        #     hue_order = order_in
        #     )
        # ax.set_xticklabels(xlabels, 
        #                    rotation = 45, 
        #                    ha = 'right', 
        #                    rotation_mode = 'anchor'
        #                    )
        
        
        # ax.set_title(
        #     f'Number of {timescale} models performing best\nbased on {metric_temp} & {part_name}'
        #     )
        # legend = ax.legend(title = 'Cluster Method', loc = 'upper right', ncol = 3, frameon = True)
        # legend.get_frame().set_alpha(0.5)

        # if write_figs:
        #     plt.savefig(
        #         f'{dir_figs}/NumbCatchBestPredicted_{metric_in}_{timescale}_{parts_name}_{models_name}.csv',
        #         dpi = 300,
        #         bbox_inches = 'tight')
        # else:
        #     plt.show()

    # print(ind_max.groupby('model')['STAID'].count())
# %%


