'''
BChoat 2024/07/07

Script to produce boxplots showing range of metrics
'''

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# define working directory
dir_res = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'
# df_in = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv',
df_in = pd.read_csv(f'{dir_res}/NSEComponents_KGE.csv',
                dtype={'STAID': 'string'})
df_in = df_in.drop_duplicates()
# df_inma = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/PerfMetrics_MeanAnnual.csv',
df_inma = pd.read_csv(f'{dir_res}/PerfMetrics_MeanAnnual.csv',
                dtype={'STAID': 'string'})
df_inma = df_inma.drop_duplicates()
# which model to work with?
model_in = [ 'XGBoost']

# getDiffByCatch(df_inma, 'region', 'STAID', 'residuals')

# regionanlization schemes to consider
rgns_schemes = ['AggEcoregion', 'Class', 'None']

# subset working dataframes to regionalization schemes of interest
df_in = df_in.query("clust_method in @rgns_schemes & model in @model_in")
df_inma = df_inma.query("clust_method in @rgns_schemes & model in @model_in")


# monthly and annual
dfs_out = {"AggEcoregion": [], "Class": [], "None": []}
    
for scheme in ['Class', 'None']:
    df_temp1 = df_in.query("clust_method == @scheme").sort_values(
        by=["STAID", 'time_scale', 'train_val']).drop_duplicates()
    df_temp2 = df_in.query("clust_method == 'AggEcoregion'").sort_values(
        by=["STAID", 'time_scale', 'train_val'])
    # dftemp1['region'] = dftemp2['region']
    assert df_temp1.shape == df_temp2.shape
    df_temp3 = pd.merge(df_temp1, df_temp2, 
                        on = ['STAID', 'train_val', 'time_scale'])
    df_temp3['region_x'] = df_temp3['region_y']

    df_temp3.columns = df_temp3.columns.str.replace("_x", "")
    df_temp3 = df_temp3[df_temp1.columns]

    dfs_out[scheme] = df_temp3

dfs_out['AggEcoregion'] = df_temp2
df_moan = pd.concat(dfs_out, axis=0).reset_index(drop=True)
# df_moan.columns = df_moan.columns.str.replace('level_0', 'Region')

# mean annual
dfs_out = {"AggEcoregion": [], "Class": [], "None": []}
    
for scheme in ['Class', 'None']:
    df_temp1 = df_inma.query("clust_method == @scheme").sort_values(
        by=["STAID", 'time_scale', 'train_val']).drop_duplicates()
    df_temp2 = df_inma.query("clust_method == 'AggEcoregion'").sort_values(
        by=["STAID", 'time_scale', 'train_val'])
    # dftemp1['region'] = dftemp2['region']
    assert df_temp1.shape == df_temp2.shape
    df_temp3 = pd.merge(df_temp1, df_temp2, 
                        on = ['STAID', 'train_val', 'time_scale'])
    df_temp3['region_x'] = df_temp3['region_y']

    df_temp3.columns = df_temp3.columns.str.replace("_x", "")
    df_temp3 = df_temp3[df_temp1.columns]

    dfs_out[scheme] = df_temp3

dfs_out['AggEcoregion'] = df_temp2
df_ma = pd.concat(dfs_out, axis=0).reset_index(drop=True)
df_ma['res_abs'] = df_ma.residuals.abs()
# df_ma.columns = df_ma.columns.str.replace('level_0', 'Region')

# further prep input dataframes so alebsl are good
df_moan = df_moan.replace('regr_precip', 'SLR')
df_moan = df_moan.replace('strd_mlr', 'MLR')

df_ma = df_ma.replace('regr_precip', 'SLR')
df_ma = df_ma.replace('strd_mlr', 'MLR')



# %% plotting options
######################################################

# set boxplot props
props_in = {
'boxprops':{'facecolor':'none', 'edgecolor':'black'},
# 'medianprops':{'color':'green'},
# 'whiskerprops':{'color':'blue'},
# 'capprops':{'color':'magenta'}
}

metric_in = 'NSE'

ylabs_in = [ '|Residuals| (cm)',
                    '',
                    metric_in,
                    '',
                    metric_in,
                    '']

annots = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']



# %% make plot comparing performance metrics between groupoing schemes
########################################
# sns.boxplot(df_workma,)
fig, axs = plt.subplots(figsize=(8, 8), nrows=3, ncols=2, sharex=True)
for i, ax in enumerate(axs.flatten()):

    if i == 4:
        df_plot = df_moan.query("time_scale == 'monthly' & train_val == 'train'")
    elif i == 5:
        df_plot = df_moan.query("time_scale == 'monthly' & train_val == 'valnit'")
    elif i == 2:
        df_plot = df_moan.query("time_scale == 'annual' & train_val == 'train'")
    elif i == 3:
        df_plot = df_moan.query("time_scale == 'annual' & train_val == 'valnit'")
        df_temp = df_plot
    elif i == 0:
        df_plot = df_ma.query("train_val == 'train'")
    elif i == 1:
        df_plot = df_ma.query("train_val == 'valnit'")

    partition = 'train' if i in [0, 2, 4] else 'valnit'
    if i in [4, 5]:
        scale = 'monthly'
        ylim_in = [-0.5, 1.1] #[-0.25, 0.26]
        met_in = metric_in
        ytick_in = [-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        annot_loc = [-0.4, -0.4]
        # df_plot = df_moan.copy()
    elif i in [2, 3]:
        scale = 'annual'
        ylim_in = [-1.5, 1.1] # [-0.5, 1]
        met_in = metric_in
        ytick_in = [-1.5, -1, -0.5, 0, 0.5, 1]
        annot_loc = [-0.4, -1.3]
        # df_plot = df_moan.copy()
    else:
        scale = 'mean_annual'
        ylim_in = [-0.5, 18]
        # ylim = [-20, 20]
        # met_in = 'residuals'
        met_in = 'res_abs'
        ytick_in = [0, 5, 10, 15]
        # df_plot = df_ma.copy()
        # df_plot['abs_residuals'] = df_plot['residuals'].abs()
        # ylim_in = (-55, 25)
        annot_loc = [-0.4, 16]
        
    # if i in [2, 4]:
    #     ylab_in = metric_in
    # elif i == 0:
    #     ylab_in = '|Residuals| [cm]'
    # else:
    #     ylab_in = ""

    if i == 0:
        title_in = 'Training'
    elif i == 1:
        title_in = 'Testing'
    else:
        ax.title.set_visible(False) # set on or off

    # leg_in = True if i == 0 else False

    sns.boxplot(df_plot,
                x = 'clust_method',
                y = met_in,
                showfliers = False,
                ax = ax,
                **props_in)

    # sns.boxplot(df_plot, 
    #         # x = 'Region', 
    #         x = 'clust_method',
    #         y = 'Gain',
    #         # hue = 'clust_method',
    #         # legend = leg_in,
    #         showfliers = False,
    #         ax=ax)
    # ax.set(ylim=ylim, ylabel=ylab_in)
    # ax.grid(axis='y')
    # ax.set_yticks(ytick_in)
    # rotation_mode='anchor'
    # plt.xticks()
    
    if i in [4, 5]:
        # ax.tick_params(axis='x', labelrotation=45, adjust='right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # if i == 0:
    #     # Adjust the legend title and labels
    #     legend = ax.legend(title='Clustering Method', loc='best')
    ax.set(ylabel = ylabs_in[i], ylim = ylim_in, title = title_in)
    ax.annotate(annots[i], annot_loc)
    # ax.set(xlabel='', ylabel=ylab_in,
    #             title = title_in)
    ax.set_yticks(ytick_in)

    ax.grid(True, axis='y') # 'both'
    # ax.axhline(0, ls='-', linewidth=0.9, color='red', zorder=3)
    # ax.legend().set_visible(False)

    if i == (len(axs.flatten())-1):
        # add legend 
        # ax.legend(loc='lower center', ncol = 3)
        # Add legend outside of the subplots
        handles, labels = ax.get_legend_handles_labels()
        # fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor = [0.5, -0.02])

        # Adjust the layout to make space for the legend
        plt.subplots_adjust(bottom=0.15)

    if i in [4, 5]:
        plt.sca(ax)  # Set current axis to ax
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    if i in [1, 3, 5]:
        ax.set_yticklabels([])

    if i in [4, 5]:
        ax.set(xlabel = "Grouping Scheme")
    

fig.subplots_adjust(hspace=0.09, wspace=0.1) 

# plt.savefig(
# # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
# # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
# #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
#     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromRegionalizationVsModelClumped_Boxplot_trainAndtest_AllScales.png',
#     dpi = 300, bbox_inches = 'tight'
# )





# %% ##########################################################################
# prep data and plot comparing performance across models
###########################################################################


# %%
# define working directory
dir_res = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'
# df_in = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv',
df_in = pd.read_csv(f'{dir_res}/NSEComponents_KGE.csv',
                dtype={'STAID': 'string'})
df_in = df_in.drop_duplicates()
# df_inma = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/PerfMetrics_MeanAnnual.csv',
df_inma = pd.read_csv(f'{dir_res}/PerfMetrics_MeanAnnual.csv',
                dtype={'STAID': 'string'})
df_inma = df_inma.drop_duplicates()
# which model to work with?
model_in = ['regr_precip', 'strd_mlr', 'XGBoost'] #'strd_PCA_mlr', 

# getDiffByCatch(df_inma, 'region', 'STAID', 'residuals')

# regionanlization schemes to consider
rgns_schemes = ['AggEcoregion', 'Class', 'None']

# subset working dataframes to regionalization schemes of interest
df_in = df_in.query("clust_method in @rgns_schemes & model in @model_in")
df_inma = df_inma.query("clust_method in @rgns_schemes & model in @model_in")


# monthly and annual
dfs_out = {"AggEcoregion": [], "Class": [], "None": []}
    
for scheme in ['Class', 'None']:
    df_temp1 = df_in.query("clust_method == @scheme").sort_values(
        by=["STAID", 'time_scale', 'train_val']).drop_duplicates()
    df_temp2 = df_in.query("clust_method == 'AggEcoregion'").sort_values(
        by=["STAID", 'time_scale', 'train_val'])
    # dftemp1['region'] = dftemp2['region']
    assert df_temp1.shape == df_temp2.shape
    df_temp3 = pd.merge(df_temp1, df_temp2, 
                        on = ['STAID', 'train_val', 'time_scale'])
    df_temp3['region_x'] = df_temp3['region_y']

    df_temp3.columns = df_temp3.columns.str.replace("_x", "")
    df_temp3 = df_temp3[df_temp1.columns]

    dfs_out[scheme] = df_temp3

dfs_out['AggEcoregion'] = df_temp2
df_moan = pd.concat(dfs_out, axis=0).reset_index(drop=True)
# df_moan.columns = df_moan.columns.str.replace('level_0', 'Region')

# mean annual
dfs_out = {"AggEcoregion": [], "Class": [], "None": []}
    
for scheme in ['Class', 'None']:
    df_temp1 = df_inma.query("clust_method == @scheme").sort_values(
        by=["STAID", 'time_scale', 'train_val']).drop_duplicates()
    df_temp2 = df_inma.query("clust_method == 'AggEcoregion'").sort_values(
        by=["STAID", 'time_scale', 'train_val'])
    # dftemp1['region'] = dftemp2['region']
    assert df_temp1.shape == df_temp2.shape
    df_temp3 = pd.merge(df_temp1, df_temp2, 
                        on = ['STAID', 'train_val', 'time_scale'])
    df_temp3['region_x'] = df_temp3['region_y']

    df_temp3.columns = df_temp3.columns.str.replace("_x", "")
    df_temp3 = df_temp3[df_temp1.columns]

    dfs_out[scheme] = df_temp3

dfs_out['AggEcoregion'] = df_temp2
df_ma = pd.concat(dfs_out, axis=0).reset_index(drop=True)
df_ma['res_abs'] = df_ma.residuals.abs()
# df_ma.columns = df_ma.columns.str.replace('level_0', 'Region')


# further prep input dataframes so alebsl are good
df_moan = df_moan.replace('regr_precip', 'SLR')
df_moan = df_moan.replace('strd_mlr', 'MLR')

df_ma = df_ma.replace('regr_precip', 'SLR')
df_ma = df_ma.replace('strd_mlr', 'MLR')




# %% plotting options
######################################################

# set boxplot props
# props_in = {
# 'boxprops':{'facecolor':'none', 'edgecolor':'black'},
# # 'medianprops':{'color':'green'},
# # 'whiskerprops':{'color':'blue'},
# # 'capprops':{'color':'magenta'}
# }

metric_in = 'NSE'

ylabs_in = [ 'Mean Annual |Residuals| (cm)',
                    '',
                    f'Annual {metric_in}',
                    '',
                    f'Monthly {metric_in}',
                    '']

annots = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']





# %% make plot comparing performance metrics between models
########################################
# sns.boxplot(df_workma,)
fig, axs = plt.subplots(figsize=(8, 8), nrows=3, ncols=2, sharex=True)
for i, ax in enumerate(axs.flatten()):

    if i == 4:
        df_plot = df_moan.query("time_scale == 'monthly' & train_val == 'train'")
    elif i == 5:
        df_plot = df_moan.query("time_scale == 'monthly' & train_val == 'valnit'")
    elif i == 2:
        df_plot = df_moan.query("time_scale == 'annual' & train_val == 'train'")
    elif i == 3:
        df_plot = df_moan.query("time_scale == 'annual' & train_val == 'valnit'")
        df_temp = df_plot
    elif i == 0:
        df_plot = df_ma.query("train_val == 'train'")
    elif i == 1:
        df_plot = df_ma.query("train_val == 'valnit'")

    partition = 'train' if i in [0, 2, 4] else 'valnit'
    if i in [4, 5]:
        scale = 'monthly'
        ylim_in = [-8, 1.1] #[-0.25, 0.26]
        met_in = metric_in
        ytick_in = [-7.5, -5, -2.5, 0, 1]
        annot_loc = [-0.4, -7]
        # df_plot = df_moan.copy()
    elif i in [2, 3]:
        scale = 'annual'
        ylim_in = [-18, 1.1] # [-0.5, 1]
        met_in = metric_in
        ytick_in = [-15, -10, -5, 0, 1]
        annot_loc = [-0.4, -16]
        # df_plot = df_moan.copy()
    else:
        scale = 'mean_annual'
        ylim_in = [-0.5, 45]
        # ylim = [-20, 20]
        # met_in = 'residuals'
        met_in = 'res_abs'
        ytick_in = [0, 10, 20, 30, 40]
        # df_plot = df_ma.copy()
        # df_plot['abs_residuals'] = df_plot['residuals'].abs()
        # ylim_in = (-55, 25)
        annot_loc = [-0.4, 40]
        
    # if i in [2, 4]:
    #     ylab_in = metric_in
    # elif i == 0:
    #     ylab_in = '|Residuals| [cm]'
    # else:
    #     ylab_in = ""

    if i == 0:
        title_in = 'Training'
    elif i == 1:
        title_in = 'Testing'
    else:
        ax.title.set_visible(False) # set on or off


    leg_in = True if i == 0 else False

    # leg_in = True if i == 0 else False

    sns.boxplot(df_plot,
                x = 'model',
                y = met_in,
                hue = 'clust_method',
                showfliers = False,
                # legend = leg_in,
                ax = ax
                )# **props_in)

    # sns.boxplot(df_plot, 
    #         # x = 'Region', 
    #         x = 'clust_method',
    #         y = 'Gain',
    #         # hue = 'clust_method',
    #         # legend = leg_in,
    #         showfliers = False,
    #         ax=ax)
    # ax.set(ylim=ylim, ylabel=ylab_in)
    # ax.grid(axis='y')
    # ax.set_yticks(ytick_in)
    # rotation_mode='anchor'
    # plt.xticks()
    
    if i in [4, 5]:
        # ax.tick_params(axis='x', labelrotation=45, adjust='right')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    # if i == 0:
    #     # Adjust the legend title and labels
    #     legend = ax.legend(title='Clustering Method', loc='best')
    ax.set(ylabel = ylabs_in[i], ylim = ylim_in, title = title_in)
    ax.annotate(annots[i], annot_loc)
    # ax.set(xlabel='', ylabel=ylab_in,
    #             title = title_in)
    ax.set_yticks(ytick_in)

    ax.grid(True, axis='y') # 'both'
    # ax.axhline(0, ls='-', linewidth=0.9, color='red', zorder=3)
    ax.legend().set_visible(False)

    if i == (len(axs.flatten())-1):
        # add legend 
        # ax.legend(loc='lower center', ncol = 3)
        # Add legend outside of the subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor = [0.5, 0])

        # Adjust the layout to make space for the legend
        plt.subplots_adjust(bottom=0.15)

    if i in [4, 5]:
        plt.sca(ax)  # Set current axis to ax
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

    if i in [1, 3, 5]:
        ax.set_yticklabels([])

    if i in [4, 5]:
        ax.set(xlabel = "Model")
    

fig.subplots_adjust(hspace=0.09, wspace=0.1) 

# plt.savefig(
# # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
# # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
# #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
#     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromModelsAndRegionalizationVsModelClumped_Boxplot_trainAndtest_AllScales.png',
#     dpi = 300, bbox_inches = 'tight'
# )

