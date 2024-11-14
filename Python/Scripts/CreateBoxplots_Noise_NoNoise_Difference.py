'''
BChoat 2024/05/24

Plot bolxplots showing range of trimmed mean scores for NSE and/or KGE.
Show scores with and without noise, and the difference between the two.
Show points as unique colors and/or shapes indicating which regionalization
approach is being represented.

'''

'''
2023/07/02 BChoat

Script to read in results from either CacluateComponentsOfKGE.py
or CalculatePerformanceMetrics_MeanAnnual.py, and uses those values
to identify best performing models.
'''


# %% import libraries
###############################################


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
# import os


# %% define directories, variables, and such
#############################################


# dir holding performance metrics
# dir_work = 'D:/Projects/GAGESii_ANNStuff/Data_Out/Results'
dir_work = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'

# dir where to place figs
dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures'
# dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'

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

# should figs be saved?
save_figs = True




# %% Run loop to get all ranks and scores by clustering method
################################

# read in main 
df_moann = pd.read_csv(f'{dir_work}/PerfRankMonthAnnual_ByClustMethModel.csv')
df_ma = pd.read_csv(f'{dir_work}/PerfRankMeanAnnual_ByClustMethModel.csv')

# df_work = pd.read_csv(f'{dir_work}/NSEComponents_KGE.csv',
#                             dtype = {'STAID': 'string'})
# df_workma = pd.read_csv(f'{dir_work}/PerfMetrics_MeanAnnual.csv',
#                             dtype = {'STAID': 'string'})

# df_work['region'] = df_work['clust_method'].str.cat(\
#             df_work['region'].astype(str), sep = '_')
# df_workma['region'] = df_work['clust_method'].str.cat(\
#             df_work['region'].astype(str), sep = '_')

# %% define plotting properties
#########################################

# define regionalization approaches that you want to make
# larger in the plots
large_schemes = ['All_1', 'Nat_3', 'Anth_0']

# define properties for boxplots
props_in = {
    'boxprops': {'facecolor':'none', 'edgecolor':'black'},
    'medianprops': {'color':'black'},
    'whiskerprops': {'color':'black'},
    'capprops': {'color':'black'}
}

# define order for legend
hue_order_in1 = ['None', 'Class', 'CAMELS', 'HLR', 'AggEcoregion', 'All_0', 
                'All_2', 'Anth_1', 
                'Nat_0', 'Nat_1', 'Nat_2', 'Nat_4']
hue_order_in2 = ['All_1', 'Anth_0', 'Nat_3']

from matplotlib.colors import ListedColormap
color_dict = {
    'None': 'darkorange',
    'AggEcoregion': 'goldenrod',
    'All_0': 'blue',
    'All_1': (20/255, 138/255, 186/255), # 'cyan',
    'All_2': 'darkgoldenrod',
    'Anth_0': (253/255, 184/255, 19/255), # 'orange',
    'Anth_1': 'green',
    'CAMELS': 'lime',
    'Class': 'firebrick',
    'HLR': 'tomato',
    'Nat_0': 'saddlebrown',
    'Nat_1': 'purple',
    'Nat_2': 'palevioletred',
    'Nat_3':  (83/255, 128/255, 58/255), # 'dimgray',
    'Nat_4': 'cyan'
}
custom_cmap = ListedColormap([color_dict[key] for key in color_dict])
custom_pallete = [color_dict[key] for key in color_dict]


# set baseline font
plt.rcParams.update({'font.size': 14})

# indices of regions to make larger
# Select indices of points to be larger
# large_indices = [8, 9, 10, ]


# %% Loop through combos of variables and make some plots
#########################################

# turn into for loop producing plots for each combo of variables above
for trainval_in, timescale_in, metric_in in \
    product(trainvals_in, timescales_in, \
            metrics_in):
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
        df_ranked = df_ranked.query("time_scale == @timescale_in")
        metric_temp = metric_in
    else:
        if metric_in in ['NSE', 'KGE']:
            continue
        df_ranked = df_ma.copy()
        metric_temp = 'residuals'

    # define file name for ranking performance based on current set of data
    fig_name = f'{dir_figs}/Rank_HeatMap/Boxplot_Regionalization_{metric_temp}_{timescale_in}'\
                f'{trainval_temp}.png'
    
    # create dataframes to be used in boxplots   
    df_boxdrop = df_ranked[
        (df_ranked['dropNoise'] == True) &
        (df_ranked['train_val'] == trainval_in)].reset_index(drop = True)
    
    df_boxwith = df_ranked[
        (df_ranked['dropNoise'] == False) &
        (df_ranked['train_val'] == trainval_in)].reset_index(drop = True)
    
    df_boxdiff = pd.DataFrame({
        'clust_method': df_boxdrop['clust_method'],
        'model': df_boxdrop['model'],
        'diff': df_boxdrop[f'{metric_in}_qmean'] - \
                df_boxwith[f'{metric_in}_qmean']
    })

    # get separate dataframe for large poits (to emphasize)
    if len(large_schemes) > 0:
        # with noise
        df_stripwith1 = df_boxwith.query("not clust_method in @large_schemes")
        df_stripwith2 = df_boxwith.query("clust_method in @large_schemes")
        # without noise
        df_stripdrop1 = df_boxdrop.query("not clust_method in @large_schemes")
        df_stripdrop2 = df_boxdrop.query("clust_method in @large_schemes")
        # diff
        df_stripdiff1 = df_boxdiff.query("not clust_method in @large_schemes")
        df_stripdiff2 = df_boxdiff.query("clust_method in @large_schemes")
        

    # %% Create a 2x3 grid with different row heights
    #######################


    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(2, 3, height_ratios=[3, 1])  # First row twice as tall as the second row

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2])
    

    sns.boxplot(
        x = df_boxdrop['model'],
        y = df_boxdrop[f'{metric_in}_qmean'],
        order = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'],
        color='white',
        ax = ax1,
        showfliers=False,
        **props_in)
    
    sns.boxplot(
        x = df_boxwith['model'],
        y = df_boxwith[f'{metric_in}_qmean'],
        order = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'],
        ax = ax2,
        showfliers=False,
        **props_in)
    
    sns.boxplot(
        x = df_boxdiff['model'],
        y = df_boxdiff['diff'],
        order = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'],
        ax = ax3,
        showfliers=False,
        **props_in)
    
    if len(large_schemes) > 0:
            # normal stripplots
        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_stripdrop1, 
                    hue='clust_method', 
                    palette=color_dict,
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax1,
                    legend=False,
                    alpha = 0.5)
        
        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_stripdrop2, 
                    hue='clust_method', 
                    palette=color_dict,
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=10,
                    ax=ax1,
                    legend=False)

        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_stripwith1, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax2,
                    legend=False,
                    alpha = 0.5)
        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_stripwith2, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=10,
                    ax=ax2,
                    legend=False)
        
        sns.stripplot(x='model', 
                    y='diff', 
                    data=df_stripdiff1, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax3,
                    legend=True,
                    alpha = 0.5,
                    hue_order = hue_order_in1)
        sns.stripplot(x='model', 
                    y='diff', 
                    data=df_stripdiff2, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=10,
                    ax=ax3,
                    legend=True,
                    hue_order = hue_order_in2)


    else:
        # normal stripplots
        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_boxdrop, 
                    hue='clust_method', 
                    palette=color_dict,
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax1,
                    legend=False)

        sns.stripplot(x='model', 
                    y=f'{metric_in}_qmean', 
                    data=df_boxwith, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax2,
                    legend=False)
        
        sns.stripplot(x='model', 
                    y='diff', 
                    data=df_boxdiff, 
                    hue='clust_method', 
                    palette=color_dict, 
                    #   palette='Set1', 
                    dodge=True, 
                    marker='o', 
                    edgecolor='gray', 
                    size=5,
                    ax=ax3,
                    legend=True)

    # add titles to plots
    ax1.set(title = f'{metric_in} W/O Noise', ylabel = f"{metric_in} (Trimmed mean - Inner 90%)")
    ax2.set(title = f'{metric_in} W/Noise', ylabel = "")
    ax3.set(title = f'W/O Noise - W/Noise', ylabel = "Difference")

    # move ylabel for ax3
    ax3.yaxis.set_label_coords(-0.25, 0.5)

    # Add labels to plots
    for i, ax in zip(['(a)', '(b)', '(c)'], [ax1, ax2, ax3]):
        ax.text(0.95, 0.05, i, transform=ax.transAxes, ha='right', va='top', fontsize=12)
        ax.yaxis.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # ax1.text(-0.03, 0.98, f'(a)', transform=ax1.transAxes, ha='right', va='top', fontsize=12)
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    # ax2.text(-0.03, 0.98, f'(b)', transform=ax2.transAxes, ha='right', va='top', fontsize=12)
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    # ax3.text(-0.03, 0.98, f'(c)', transform=ax3.transAxes, ha='right', va='top', fontsize=12)
    # plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    # plt.xticks(rotation=45, horizontalalignment='right')


    # Improve legend
    plt.legend(title='Regionalization\nScheme', bbox_to_anchor=(1.05, 1), loc='upper left')
    # change position of plots
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

    # plt.tight_layout()

    # Display or save the plot
    if save_figs:
        plt.savefig(fig_name, dpi=300, bbox_inches = 'tight')
    else:
        plt.show()
