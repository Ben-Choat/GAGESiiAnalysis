
'''
BChoat 2024/05/07

Plot summary plots of explanatory variables from gagesii analysis
Should be 3200 catchments.
'''





# %% import libraries
############################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Load_Data import load_data_fun

# %% define dirs, vars, etc.
##########################################

# read in explanatory variables
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# load training data
df_expl_train, _, df_ID_train = load_data_fun(
        dir_work = dir_expl, 
        time_scale = 'annual',
        train_val = 'train',
        clust_meth = 'None',
        region = 'All',
        standardize = False # whether or not to standardize data
        )
df_expl_train['Partition'] = 'train'
# testing data
df_expl_test, _, df_ID_test = load_data_fun(
        dir_work = dir_expl, 
        time_scale = 'annual',
        train_val = 'valnit',
        clust_meth = 'None',
        region = 'All',
        standardize = False # whether or not to standardize data
        )
df_expl_test['Partition'] = 'test'

df_expl = pd.concat([df_expl_train, df_expl_test],
                    axis = 0)

df_expl.loc[df_expl["TS_WaterUse_wu"] < 0, 'TS_WaterUse_wu'] = 0
df_expl.loc[df_expl["TS_NLCD_42"] < 0, 'TS_NLCD_42'] = 0

# get mean values for each STAID
df_expl_mean = (
        df_expl.groupby(['STAID', 'Partition'])
        .mean()
        .drop('year', axis = 1)
        .reset_index(drop=False)
                )

df_ID = pd.concat([df_ID_train, df_ID_test],
                    axis = 0)

df_all = pd.merge(df_expl_mean, df_ID.drop('DRAIN_SQKM', axis=1), 
                  on = 'STAID')

# read in feature categories to be used for subsetting explanatory vars
# into cats of interest
feat_cats = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
    # 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories_wUnits.csv'
    )

# %% plots
############################################
# avg high water table, high intensity dev, open space dev, evergreen forest
# freshwater withdrawals, housing density, dam storage, dam density
vars_in = ['TS_Housing_HDEN', 'TS_NLCD_24', 'TS_NLCD_21',
           'TS_NLCD_42', 'TS_WaterUse_wu', 'TS_Housing_HDEN',
           'DDENS_2009', 'STOR_NID_2009', 'DRAIN_SQKM'] #  'WWTP_Effluent'

# should y-scale be log?
logs_in = [True, True, True, False, True, True, True, True, 
           True]

for var, log_in in zip(vars_in, logs_in):
# y-variable
    y_in = var
    # y_lable
    alias_in = feat_cats.query("Features == @y_in")['Alias'].values[0]
    unit_in = feat_cats.query("Features == @y_in")['Units_plot'].values[0]
    y_lab = f'{alias_in} ({unit_in})'
    # Create violin plot
    ax = sns.violinplot(x='Partition', y=y_in, data=df_expl)

    # Add points colored based on Category
    sns.stripplot(x='Partition', y=y_in, 
                data=df_all, hue = 'AggEcoregion', 
                alpha=0.7, ax=ax)  # black points
    # sns.despine()
    if log_in:
        ax.set_yscale('log')
    ax.set_ylabel(y_lab)
    ax.legend(loc = 'upper center')


    plt.savefig(
        f'{dir_expl}/Figures/SummaryPlots/{alias_in}_summary.png', dpi=300,
        bbox_inches = 'tight'
    )
    plt.show()


