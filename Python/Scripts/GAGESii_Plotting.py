'''
BChoat 10/18/2022

Main script to produce results figures.
'''

# %%
# Import libraries
################


import pandas as pd
import numpy as np
# import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
# from statsmodels.distributions.empirical_distribution import ECDF


# %%
# define directory variables and specify scenario to work with


# Define directory variables
# directory with data to work with
# dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out' 
dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results' 

# directory where SHAP values are located
dir_shap = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'

# directory where to write figs
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'

# directory where to place outputs
# dir_umaphd = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'




# %% load data
###########

# independent results for each catchment
# # mean annual
# df_resind_mannual = pd.read_pickle(
#     f'{dir_work}/mean_annual/combined/All_IndResults_mean_annual.pkl'
# )

# # annual
# df_resind_annual = pd.read_pickle(
#     f'{dir_work}/annual/combined/All_IndResults_annual.pkl'
# )
# # month
# df_resind_monthly = pd.read_pickle(
#     f'{dir_work}/monthly/combined/All_IndResults_monthly.pkl'
# )

# mean annual
df_resind_mannual = pd.read_csv(
        f'{dir_work}/PerfMetrics_MeanAnnual.csv',
        dtype = {'STAID': 'string',
                    'region': 'string'}
    )

df_resind_mannual['|residuals|'] = df_resind_mannual.residuals.abs()

df_resind_mannual = df_resind_mannual[[
        'STAID', 'residuals', '|residuals|', 'clust_method', 'region',\
                    'model', 'time_scale', 'train_val'
    ]]

# annual and monthy 
df_resind_All = pd.read_csv(
        f'{dir_work}/NSEComponents_KGE.csv',
        dtype = {'STAID': 'string',
                'region': 'string'}
    )

df_resind_annual = df_resind_All[
    df_resind_All['time_scale'] == 'annual'
    ]

df_resind_monthly = df_resind_All[
    df_resind_All['time_scale'] == 'monthly'
    ]

# results_summAll = results_summAll[
#     results_summAll['train_val'] == part_in
# ]

# summary results for each catchment
# mean annual
# df_ressumm_mannual = pd.read_pickle(
#     f'{dir_work}/mean_annual/combined/All_SummaryResults_mean_annual.pkl'
# )
# # annual
# df_ressumm_annual = pd.read_pickle(
#     f'{dir_work}/annual/combined/All_SummaryResults_annual.pkl'
# )

# # month
# df_ressumm_monthly = pd.read_pickle(
#     f'{dir_work}/monthly/combined/All_SummaryResults_monthly.pkl'
# )

# SHAP values
# mean annual
df_shap_mannual = pd.read_csv(
    f'{dir_shap}/MeanShap_valnit_mean_annual.csv'
)
# annual
df_shap_annual = pd.read_csv(
    f'{dir_shap}/MeanShap_valnit_annual.csv'
)
# mean annual
df_shap_monthly = pd.read_csv(
    f'{dir_shap}/MeanShap_valnit_monthly.csv'
)



# read in feature categories to be used for subsetting explanatory vars
# into cats of interest
feat_cats = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
    )
# feat_cats['Features'] = feat_cats['Features'].str.replace('TS_', '')


# # define features that are natural features
# nat_feats = feat_cats.loc[
#     feat_cats['Coarsest_Cat'] == 'Natural', 'Features'
# ].reset_index(drop = True)

# # define features that are anthropogenic features
# anthro_feats = feat_cats.loc[
#     feat_cats['Coarsest_Cat'] == 'Anthro', 'Features'
# ].reset_index(drop = True)

# define features that are climate
clim_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Climate', 'Alias'
].reset_index(drop = True)
# clim_feats = clim_feats.str.replace('TS_', '')

# define features that are Physiography features
phys_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Physiography', 'Alias'
].reset_index(drop = True)
# phys_feats = phys_feats.str.replace('TS_', '')

# define features that are Anthro_Hydro
anhyd_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Anthro_Hydro', 'Alias'
].reset_index(drop = True)
# anhyd_feats = anhyd_feats.str.replace('TS_', '')

# define features that are Anthro_Land features
anland_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Anthro_Land', 'Alias'
].reset_index(drop = True)
# anland_feats = anland_feats.str.replace('TS_', '')

# %%
# eCDF plots
##############

# mean annual
data_in_mannual = df_resind_mannual[
    df_resind_mannual['train_val'] == 'valnit'
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
data_in_annual = df_resind_annual[
    df_resind_annual['train_val'] == 'valnit'
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
data_in_month = df_resind_monthly[
    df_resind_monthly['train_val'] == 'valnit'
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
clust_meth_in = 'Nat_3'
model_in = 'XGBoost'
metric_in = 'KGE'

data_in_mannual = data_in_mannual[
    (data_in_mannual['clust_method'] == clust_meth_in) &
    (data_in_mannual['model'] == model_in)
]

data_in_annual = data_in_annual[
    (data_in_annual['clust_method'] == clust_meth_in) &
    (data_in_annual['model'] == model_in)
]

data_in_month = data_in_month[
    (data_in_month['clust_method'] == clust_meth_in) &
    (data_in_month['model'] == model_in)
]
####

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (10, 5), sharey = True)

ax1.set(xlabel = '|residuals| [ft]', 
        ylabel = 'Non-Exceedence Probability')
ax1.set_xlim(0, 3)
ax1.annotate('(a)', xy = (2.7, 0.02))
ax1.grid()
ax1.title.set_text('Mean Annual')
sns.ecdfplot(
    data = data_in_mannual,
    x = '|residuals|',
    hue = 'region',
    linestyle = '--',
    palette = 'Paired',
    ax = ax1
)
sns.move_legend(ax1, 'lower center')
ax1.get_legend().set_title('Region')


ax2.set_xlim(-1, 1)
ax2.set(xlabel = metric_in)
ax2.annotate('(b)', xy = (0.8, 0.02))
ax2.grid()
ax2.title.set_text('Annual')
sns.ecdfplot(
    data = data_in_annual,
    x = metric_in,
    hue = 'region',
    linestyle = '--',
    palette = 'Paired',
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
    hue = 'region',
    linestyle = '--',
    palette = 'Paired',
    ax = ax3,
    legend = False
)


# # save fig
# plt.savefig(
#     f'{dir_figs}/ecdfs.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )


# %%
# heatmap of shap values using all variables
##############


clust_meth_in = 'Nat_3'
model_in = 'XGBoost'
# metric_in = 'KGE'

vmin_mannual = -4
vmax_mannual = 4

vmin_annual = -3
vmax_annual = 3

vmin_month = -1
vmax_month = 1

# read in data
# mean annual
shap_mannual = df_shap_mannual[df_shap_mannual['clust_meth'] == clust_meth_in]
shap_mannual = shap_mannual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score',
     'tmin_1', 'tmax_1', 'prcp_1', 'vp_1', 'swe_1'], axis = 1
)
shap_mannual.index = df_shap_mannual.loc[
    df_shap_mannual['clust_meth'] == clust_meth_in, 'region'
                    ]
shap_mannual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
# shap_mannual.columns = shap_mannual.columns.str.replace(
#     'TS_', ''
# )

# # get number of time variable was not included
# mannual_notinc = pd.DataFrame(
#     shap_mannual.isna().sum(),
#     columns = ['Count']
# ).sort_values(by = 'Count')


# annual
shap_annual = df_shap_annual[df_shap_annual['clust_meth'] == clust_meth_in]
shap_annual = shap_annual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_annual.index = df_shap_annual.loc[
    df_shap_annual['clust_meth'] == clust_meth_in, 'region'
    ]
shap_annual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)

# shap_annual.columns = shap_annual.columns.str.replace(
#     'TS_', ''
# )

# monthly
shap_monthly = df_shap_monthly[df_shap_monthly['clust_meth'] == clust_meth_in]
shap_monthly = shap_monthly.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_monthly.index = df_shap_monthly.loc[
    df_shap_monthly['clust_meth'] == clust_meth_in, 'region'
    ]
shap_monthly.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
shap_monthly['AntPrecip'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('prcp_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant_swe'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('swe_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant_vp'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('vp_')]
    ].apply('sum', axis = 1)

shap_monthly = shap_monthly.drop(
    shap_monthly.columns[
        (shap_monthly.columns.str.contains('prcp_')) | \
        (shap_monthly.columns.str.contains('swe_')) | \
        (shap_monthly.columns.str.contains('vp_'))
    ],  axis = 1  
)
# shap_monthly.columns = shap_monthly.columns.str.replace(
#     'TS_', ''
# )

# heatmaps

# mean annual

# calculate column order
mannual_col_order = shap_mannual.fillna(0).abs().mean().sort_values(
                                                    ascending = False).index

data_mannualin = shap_mannual.reindex(mannual_col_order, axis = 1)

mannual_data_plot = data_mannualin.iloc[:, 0:30].T
# mannual_data_plot.index.name = 'Explanatory Variables'

# annual

annual_col_order = shap_annual.fillna(0).abs().mean().sort_values(ascending = False).index
data_annualin = shap_annual.reindex(annual_col_order, axis = 1)

annual_data_plot = data_annualin.iloc[:, 0:30].T
# annual_data_plot.index.name = 'Explanatory Variables'

# monthly

monthly_col_order = shap_monthly.fillna(0).abs().mean().sort_values(ascending = False).index
data_monthlyin = shap_monthly.reindex(monthly_col_order, axis = 1)

monthly_data_plot = data_monthlyin.iloc[:, 0:30].T
# monthly_data_plot.index.name = 'Explanatory Variables'


# mean annual

fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, 
                        layout = 'constrained',
                        figsize = (11, 7.75)) # ,
                        # gridspec_kw={'width_ratios': [0.3, 0.3, 0.3]})
# fig, ax1 = plt.subplots(figsize = (2.75, 11)) # (12, 4))
ax1.title.set_text('Mean Annual')
sns.heatmap(
    mannual_data_plot, # mannual_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax1, 
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = vmin_mannual, # np.min(np.min(data_mannualin.iloc[:, 0:30])),
    vmax = vmax_mannual,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax1.set(xlabel = 'Region', ylabel = 'Explanatory Variables') # xlabel = 'Explanatory Varaibles',
# ax1.set_xticklabels(mannual_data_plot.columns[0:40], ha = 'center')
ax1.tick_params(axis = 'x',
                rotation = 90)
# color climate variables
# for x in np.where(mannual_data_plot.columns.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
    # ax1.get_xticklabels()[x].set_color('blue')
for x in np.where(mannual_data_plot.index.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
    ax1.get_yticklabels()[x].set_color('blue')
# color physiography variables
# for x in np.where(mannual_data_plot.columns.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
    # ax1.get_xticklabels()[x].set_color('saddlebrown')
for x in np.where(mannual_data_plot.index.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
    ax1.get_yticklabels()[x].set_color('saddlebrown')
# color anthropogenic hydrologic alteration variables
# for x in np.where(mannual_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    # ax1.get_xticklabels()[x].set_color('red')
for x in np.where(mannual_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax1.get_yticklabels()[x].set_color('red')

# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AllVars_meanannual.png', # Heat_AllVars_meanannual_Horz.png',
#     dpi = 300,
#     bbox_inches = 'tight'
#     )



###################


# annual

# annual_col_order = shap_annual.fillna(0).abs().mean().sort_values(ascending = False).index
# data_annualin = shap_annual.reindex(annual_col_order, axis = 1)

# annual_data_plot = data_annualin.iloc[:, 0:40] .T
# # annual_data_plot.index.name = 'Explanatory Variables'




# fig, ax2 = plt.subplots(figsize = (2.75, 11)) # (12, 4))
# ax2.tick_params(axis = 'y',
#                 rotation = 66)
ax2.title.set_text('Annual')
# fig.set_xticklabels(ha = 'left')
# ax2.annotate('(b)', xy = (40,12))
sns.heatmap(
    annual_data_plot, # annual_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax2,
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = vmin_annual, # np.min(np.min(data_annualin.iloc[:, 0:30])),
    vmax = vmax_annual,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax2.set(xlabel = 'Region') # (xlabel = 'Explanatory Varaibles', 
# ax2.set_xticklabels(annual_data_plot.columns[0:40], ha = 'center')
ax2.tick_params(axis = 'x',
                rotation = 90)
# color climate variables
# for x in np.where(annual_data_plot.columns.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
#     ax2.get_xticklabels()[x].set_color('blue')
for x in np.where(annual_data_plot.index.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
    ax2.get_yticklabels()[x].set_color('blue')
# color physiography variables
# for x in np.where(annual_data_plot.columns.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
#     ax2.get_xticklabels()[x].set_color('saddlebrown')
for x in np.where(annual_data_plot.index.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
    ax2.get_yticklabels()[x].set_color('saddlebrown')
# color anthropogenic hydrologic alteration variables
# for x in np.where(annual_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
#     ax2.get_xticklabels()[x].set_color('red')
for x in np.where(annual_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax2.get_yticklabels()[x].set_color('red')


# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AllVars_annual.png', # Heat_AllVars_annual_Horz.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )



######################

# monthly

# monthly_col_order = shap_monthly.fillna(0).abs().mean().sort_values(ascending = False).index
# data_monthlyin = shap_monthly.reindex(monthly_col_order, axis = 1)

# monthly_data_plot = data_monthlyin.iloc[:, 0:40].T
# # monthly_data_plot.index.name = 'Explanatory Variables'




# fig, ax3 = plt.subplots(figsize = (2.75, 11))

ax3.title.set_text('Monthly')
# ax3.annotate('(c)', xy = (40,12))
# fig.set_xticklabels(ha = 'left')
# ax.annotate('Wetter', xy = (12.02,-0.032))
sns.heatmap(
    monthly_data_plot, # monthly_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax3,
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = vmin_month, # np.min(np.min(data_monthlyin.iloc[:, 0:30])),
    vmax = vmax_month,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax3.set(xlabel = 'Region')# ylabel = 'Explanatory Varaibles',
ax3.tick_params(axis = 'x',
                rotation = 90)
# color climate variables
# for x in np.where(monthly_data_plot.columns.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
#     ax3.get_xticklabels()[x].set_color('blue')
for x in np.where(monthly_data_plot.index.isin(clim_feats))[0]: # np.arange(5, 20, 1): 
    ax3.get_yticklabels()[x].set_color('blue')
# color physiography variables
# for x in np.where(monthly_data_plot.columns.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
#     ax3.get_xticklabels()[x].set_color('saddlebrown')
for x in np.where(monthly_data_plot.index.isin(phys_feats))[0]: # np.arange(5, 20, 1): 
    ax3.get_yticklabels()[x].set_color('saddlebrown')
# color anthropogenic hydrologic alteration variables
# for x in np.where(monthly_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
#     ax3.get_xticklabels()[x].set_color('red')
for x in np.where(monthly_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax3.get_yticklabels()[x].set_color('red')

# annotate 
ax1.annotate('(a)', xy = (12, 30.5), annotation_clip = False)
ax2.annotate('(b)', xy = (12, 30.5), annotation_clip = False)
ax3.annotate('(c)', xy = (12, 30.5), annotation_clip = False)

# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AllVars_AllTimescales.png', # Heat_AllVars_monthly_Horz.png
#     dpi = 300,
#     bbox_inches = 'tight'
#     )






# %%
# heatmap of shap values using Anthropogenic variables
##############


clust_meth_in = 'Nat_3'
model_in = 'XGBoost'
# metric_in = 'KGE'

vmin_mannual = -4
vmax_mannual = 4

vmin_annual = -3
vmax_annual = 3

vmin_month = -1
vmax_month = 1


# read in data
# mean annual
shap_mannual = df_shap_mannual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score',
     'tmin_1', 'tmax_1', 'prcp_1', 'vp_1', 'swe_1'], axis = 1
)
shap_mannual.index = df_shap_mannual['region']
shap_mannual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
# shap_mannual.columns = shap_mannual.columns.str.replace(
#     'TS_', ''
# )

# annual
shap_annual = df_shap_annual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_annual.index = df_shap_annual['region']
shap_annual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
# shap_annual.columns = shap_annual.columns.str.replace(
#     'TS_', ''
# )

# monthly
shap_monthly = df_shap_monthly.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_monthly.index = df_shap_monthly['region']
shap_monthly.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
# shap_monthly.columns = shap_monthly.columns.str.replace(
#     'TS_', ''
# )


# heatmaps

# mean annual

# calculate column order
mannual_col_order = shap_mannual.fillna(0).abs().mean().sort_values(ascending = False).index

data_mannualin = shap_mannual.reindex(mannual_col_order, axis = 1)
data_mannualin = data_mannualin.iloc[:,
    (data_mannualin.columns.isin(anhyd_feats)) |
    (data_mannualin.columns.isin(anland_feats))
]
mannual_data_plot = data_mannualin.iloc[:, 0:40].T
# mannual_data_plot.index.name = 'Explanatory Variables'

# annual

annual_col_order = shap_annual.fillna(0).abs().mean().sort_values(ascending = False).index
data_annualin = shap_annual.reindex(annual_col_order, axis = 1)
data_annualin = data_annualin.iloc[:,
    (data_annualin.columns.isin(anhyd_feats)) |
    (data_annualin.columns.isin(anland_feats))
]
annual_data_plot = data_annualin.iloc[:, 0:40].T
# annual_data_plot.index.name = 'Explanatory Variables'

# monthly

monthly_col_order = shap_monthly.fillna(0).abs().mean().sort_values(ascending = False).index
data_monthlyin = shap_monthly.reindex(monthly_col_order, axis = 1)
data_monthlyin = data_monthlyin.iloc[:,
    (data_monthlyin.columns.isin(anhyd_feats)) |
    (data_monthlyin.columns.isin(anland_feats))
]
monthly_data_plot = data_monthlyin.iloc[:, 0:40].T
monthly_data_plot.index.name = 'Explanatory Variables'


# mean annual

fig, (ax1, ax2, ax3) = plt.subplots(ncols = 3, nrows = 1, 
                        layout = 'constrained',
                        figsize = (11, 6.3))
# fig, ax1 = plt.subplots(figsize = (3.5, 6)) # (6, 3))
ax1.title.set_text('Mean Annual')
sns.heatmap(
    mannual_data_plot, # mannual_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax1,
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = -0.04, # np.min(np.min(data_mannualin.iloc[:, 0:30])),
    vmax = 0.04,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax1.set(xlabel = 'Region', ylabel = 'Explanatory Variables') # ylabel = 'Region') # xlabel = 'Explanatory Varaibles',
# ax1.set_xticklabels(mannual_data_plot.columns, ha = 'center')

# ax1.tick_params(axis = 'x',
#                 rotation = 90)
# color climate variables
# plt.axes(ax1)

# color anthropogenic hydrologic alteration variables
# for x in np.where(mannual_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
#     ax1.get_xticklabels()[x].set_color('red')
for x in np.where(mannual_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax1.get_yticklabels()[x].set_color('red')

# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AnthVars_meanannual.png', # Heat_AnthVars_meanannual_Horz.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )



###################


# annual

# annual_col_order = shap_annual.fillna(0).abs().mean().sort_values(ascending = False).index
# data_annualin = shap_annual.reindex(annual_col_order, axis = 1)

# annual_data_plot = data_annualin.iloc[:, 0:40] .T
# # annual_data_plot.index.name = 'Explanatory Variables'


# fig, ax2 = plt.subplots(figsize = (3.5, 6)) # (6, 3))
# ax2.tick_params(axis = 'y',
#                 rotation = 66)
ax2.title.set_text('Annual')
# fig.set_xticklabels(ha = 'left')
sns.heatmap(
    annual_data_plot, # annual_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax2,
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = -0.04, # np.min(np.min(data_annualin.iloc[:, 0:30])),
    vmax = 0.04,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax2.set(xlabel = 'Region') # ylabel = 'Region') # (xlabel = 'Explanatory Varaibles', 
# ax2.set_xticklabels(annual_data_plot.columns, ha = 'center')
# ax2.tick_params(axis = 'x',
#                 rotation = 90)

# color anthropogenic hydrologic alteration variables
# for x in np.where(annual_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
#     ax2.get_xticklabels()[x].set_color('red')
for x in np.where(annual_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax2.get_yticklabels()[x].set_color('red')

# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AnthVars_annual.png',  # Heat_AnthVars_annual_Horz.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )



######################

# monthly

# monthly_col_order = shap_monthly.fillna(0).abs().mean().sort_values(ascending = False).index
# data_monthlyin = shap_monthly.reindex(monthly_col_order, axis = 1)

# monthly_data_plot = data_monthlyin.iloc[:, 0:40].T
# # monthly_data_plot.index.name = 'Explanatory Variables'


# fig, ax3 = plt.subplots(figsize = (3.5, 6)) # (6, 3))

ax3.title.set_text('Monthly')
# ax3.annotate('(c)', xy = (21,12))
# fig.set_xticklabels(ha = 'left')
# ax.annotate('Wetter', xy = (12.02,-0.032))
sns.heatmap(
    monthly_data_plot, # monthly_notinc[0:30].index],
    linewidth = 0.05,
    linecolor = 'black',
    ax = ax3,
    cmap = sns.color_palette('coolwarm_r', 100), # cust_color, # 'RdYlBu', # 'jet'
    center = 0,
    vmin = -0.003, # np.min(np.min(data_monthlyin.iloc[:, 0:30])),
    vmax = 0.003,
    cbar_kws = {'label': 'SHAP (Impact on model output)',
                'use_gridspec': False,
                'location': 'bottom'} # ,
    # robust = True
    )
ax3.set(xlabel = 'Region', ylabel = '') 
# color anthropogenic hydrologic alteration variables
# for x in np.where(monthly_data_plot.columns.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
#     ax3.get_xticklabels()[x].set_color('red')
for x in np.where(monthly_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax3.get_yticklabels()[x].set_color('red')


# annotate 
ax1.annotate('(a)', xy = (12, 21.5), annotation_clip = False)
ax2.annotate('(b)', xy = (12, 21.5), annotation_clip = False)
ax3.annotate('(c)', xy = (12, 21.5), annotation_clip = False)


# # save fig
# plt.savefig(
#     f'{dir_figs}/Heat_AnthVars_AllTimescales.png',  # Heat_AnthVars_monthly_Horz.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )








# %%
# calculate total mean SHAP for each variable type

# mean annual

df_shap_mannual.columns = df_shap_mannual.columns.str.replace('TS_', '')

# climate
tempclim = df_shap_mannual.iloc[
    :, df_shap_mannual.columns.isin(clim_feats)
]
# physiography
tempphys = df_shap_mannual.iloc[
    :, df_shap_mannual.columns.isin(phys_feats)
]
# anthrohydro
temphydro = df_shap_mannual.iloc[
    :, df_shap_mannual.columns.isin(anhyd_feats)
]
# anthroland
templand = df_shap_mannual.iloc[
    :, df_shap_mannual.columns.isin(anland_feats)
]



df_shaptotout_mannual = pd.DataFrame({
    'Region': df_shap_mannual['region'],
    'Climate_Impact': tempclim.abs().sum(axis = 1),
    'Physiography_Impact': tempphys.abs().sum(axis = 1),
    'AnthroHydro_Impact': temphydro.abs().sum(axis = 1),
    'AnthroLand_Impact': templand.abs().sum(axis = 1),

})


# annual

df_shap_annual.columns = df_shap_annual.columns.str.replace('TS_', '')

# climate
tempclim = df_shap_annual.iloc[
    :, df_shap_annual.columns.isin(clim_feats)
]
# physiography
tempphys = df_shap_annual.iloc[
    :, df_shap_annual.columns.isin(phys_feats)
]
# anthrohydro
temphydro = df_shap_annual.iloc[
    :, df_shap_annual.columns.isin(anhyd_feats)
]
# anthroland
templand = df_shap_annual.iloc[
    :, df_shap_annual.columns.isin(anland_feats)
]



df_shaptotout_annual = pd.DataFrame({
    'Region': df_shap_annual['region'],
    'Climate_Impact': tempclim.abs().sum(axis = 1),
    'Physiography_Impact': tempphys.abs().sum(axis = 1),
    'AnthroHydro_Impact': temphydro.abs().sum(axis = 1),
    'AnthroLand_Impact': templand.abs().sum(axis = 1),

})


# monthly

df_shap_monthly.columns = df_shap_monthly.columns.str.replace('TS_', '')

# climate
tempclim = df_shap_monthly.iloc[
    :, df_shap_monthly.columns.isin(clim_feats)
]
# physiography
tempphys = df_shap_monthly.iloc[
    :, df_shap_monthly.columns.isin(phys_feats)
]
# anthrohydro
temphydro = df_shap_monthly.iloc[
    :, df_shap_monthly.columns.isin(anhyd_feats)
]
# anthroland
templand = df_shap_monthly.iloc[
    :, df_shap_monthly.columns.isin(anland_feats)
]



df_shaptotout_monthly = pd.DataFrame({
    'Region': df_shap_monthly['region'],
    'Climate_Impact': tempclim.abs().sum(axis = 1),
    'Physiography_Impact': tempphys.abs().sum(axis = 1),
    'AnthroHydro_Impact': temphydro.abs().sum(axis = 1),
    'AnthroLand_Impact': templand.abs().sum(axis = 1),

})

# %% 
# barplots

# mean annual
df_shaptotout_mannual.index = df_shaptotout_mannual['Region']
df_shaptotout_mannual.drop('Region', axis = 1, inplace = True)


df_inmannual = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)

# # normalize data
# df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']/df_shaptotout_mannual.sum(axis = 1)
# df_inmannual['Physiography'] = df_shaptotout_mannual['Physiography_Impact']/df_shaptotout_mannual.sum(axis = 1)
# df_inmannual['AnthroHydro'] = df_shaptotout_mannual['AnthroHydro_Impact']/df_shaptotout_mannual.sum(axis = 1)
# df_inmannual['AnthroLand'] = df_shaptotout_mannual['AnthroLand_Impact']/df_shaptotout_mannual.sum(axis = 1)
# unnormalized data
df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']
df_inmannual['Physiography'] = df_shaptotout_mannual['Physiography_Impact']
df_inmannual['AnthroHydro'] = df_shaptotout_mannual['AnthroHydro_Impact']
df_inmannual['AnthroLand'] = df_shaptotout_mannual['AnthroLand_Impact']


# annual
df_shaptotout_annual.index = df_shaptotout_annual['Region']
df_shaptotout_annual.drop('Region', axis = 1, inplace = True)

df_inannual = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)
# # Normalize data
# df_inannual['Climate'] = df_shaptotout_annual['Climate_Impact']/df_shaptotout_annual.sum(axis = 1)
# df_inannual['Physiography'] = df_shaptotout_annual['Physiography_Impact']/df_shaptotout_annual.sum(axis = 1)
# df_inannual['AnthroHydro'] = df_shaptotout_annual['AnthroHydro_Impact']/df_shaptotout_annual.sum(axis = 1)
# df_inannual['AnthroLand'] = df_shaptotout_annual['AnthroLand_Impact']/df_shaptotout_annual.sum(axis = 1)
# unormalized data
df_inannual['Climate'] = df_shaptotout_annual['Climate_Impact']
df_inannual['Physiography'] = df_shaptotout_annual['Physiography_Impact']
df_inannual['AnthroHydro'] = df_shaptotout_annual['AnthroHydro_Impact']
df_inannual['AnthroLand'] = df_shaptotout_annual['AnthroLand_Impact']
# monthly
df_shaptotout_monthly.index = df_shaptotout_monthly['Region']
df_shaptotout_monthly.drop('Region', axis = 1, inplace = True)

df_inmonthly = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)

# # Normalize data
# df_inmonthly['Climate'] = df_shaptotout_monthly['Climate_Impact']/df_shaptotout_monthly.sum(axis = 1)
# df_inmonthly['Physiography'] = df_shaptotout_monthly['Physiography_Impact']/df_shaptotout_monthly.sum(axis = 1)
# df_inmonthly['AnthroHydro'] = df_shaptotout_monthly['AnthroHydro_Impact']/df_shaptotout_monthly.sum(axis = 1)
# df_inmonthly['AnthroLand'] = df_shaptotout_monthly['AnthroLand_Impact']/df_shaptotout_monthly.sum(axis = 1)

# unnormalize data
df_inmonthly['Climate'] = df_shaptotout_monthly['Climate_Impact']
df_inmonthly['Physiography'] = df_shaptotout_monthly['Physiography_Impact']
df_inmonthly['AnthroHydro'] = df_shaptotout_monthly['AnthroHydro_Impact']
df_inmonthly['AnthroLand'] = df_shaptotout_monthly['AnthroLand_Impact']


# df_inmannual = df_inmannual.reindex(df_inannual.columns, axis = 1)

# plot
color = ['blue', 'saddlebrown', 'red', 'black']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True)

ax1.title.set_text('Mean Annual')
df_inmannual.plot(
    kind = 'bar', stacked = True, ax = ax1,
    color = color
)
ax1.set(ylabel = 'Relative contribution of each variable type')
ax1.annotate('(a)', xy = (10.51, 1.01))


ax2.title.set_text('Annual')
df_inannual.plot(
    kind = 'bar', stacked = True, ax = ax2, legend = False,
    color = color
)
ax2.annotate('(b)', xy = (10.51, 1.01))

ax3.title.set_text('Monthly')
df_inmonthly.plot(
    kind = 'bar', stacked = True, ax = ax3, legend = False,
    color = color
)
ax3.annotate('(c)', xy = (10.51, 1.01))

# # save fig
# plt.savefig(
#     f'{dir_figs}/RelativeContributions_NonNormalized.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )



# %%
# df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']/df_shaptotout_mannual.sum(axis = 1)

df_shaptotout_mannual.index = df_shaptotout_mannual['Region']
df_shaptotout_mannual.drop('Region', axis = 1, inplace = True)


df_shaptotout_mannual.drop('Climate_Impact', axis = 1, inplace = True)
df_inmannual['Physiography'] = df_shaptotout_mannual['Physiography_Impact']/df_shaptotout_mannual.sum(axis = 1)
df_inmannual['AnthroHydro'] = df_shaptotout_mannual['AnthroHydro_Impact']/df_shaptotout_mannual.sum(axis = 1)
df_inmannual['AnthroLand'] = df_shaptotout_mannual['AnthroLand_Impact']/df_shaptotout_mannual.sum(axis = 1)

# annual
df_shaptotout_annual.drop('Climate_Impact', axis = 1, inplace = True)
df_shaptotout_annual.index = df_shaptotout_annual['Region']
df_shaptotout_annual.drop('Region', axis = 1, inplace = True)

df_inannual = pd.DataFrame(
    columns = ['Physiography', 'AnthroHydro', 'AnthroLand']
)

df_inannual['Physiography'] = df_shaptotout_annual['Physiography_Impact']/df_shaptotout_annual.sum(axis = 1)
df_inannual['AnthroHydro'] = df_shaptotout_annual['AnthroHydro_Impact']/df_shaptotout_annual.sum(axis = 1)
df_inannual['AnthroLand'] = df_shaptotout_annual['AnthroLand_Impact']/df_shaptotout_annual.sum(axis = 1)

# monthly
df_shaptotout_monthly.drop('Climate_Impact', axis = 1, inplace = True)
df_shaptotout_monthly.index = df_shaptotout_monthly['Region']
df_shaptotout_monthly.drop('Region', axis = 1, inplace = True)

df_inmonthly = pd.DataFrame(
    columns = ['Physiography', 'AnthroHydro', 'AnthroLand']
)
df_inmonthly['Physiography'] = df_shaptotout_monthly['Physiography_Impact']/df_shaptotout_monthly.sum(axis = 1)
df_inmonthly['AnthroHydro'] = df_shaptotout_monthly['AnthroHydro_Impact']/df_shaptotout_monthly.sum(axis = 1)
df_inmonthly['AnthroLand'] = df_shaptotout_monthly['AnthroLand_Impact']/df_shaptotout_monthly.sum(axis = 1)

# df_inannual.drop('Climate', axis = 1, inplace = True)
# df_inmonthly.drop('Climate', axis = 1, inplace = True)


# plot
color = ['saddlebrown', 'red', 'black']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True)

ax1.title.set_text('Mean Annual')
df_inmannual.plot(
    kind = 'bar', stacked = True, ax = ax1,
    color = color
)
ax1.set(ylabel = 'Relative contribution of each variable type')
ax1.annotate('(a)', xy = (10.51, 1.01))

ax2.title.set_text('Annual')
df_inannual.plot(
    kind = 'bar', stacked = True, ax = ax2, legend = False,
    color = color
)
ax2.annotate('(b)', xy = (10.51, 1.01))


ax3.title.set_text('Monthly')
df_inmonthly.plot(
    kind = 'bar', stacked = True, ax = ax3, legend = False,
    color = color
)
ax3.annotate('(c)', xy = (10.51, 1.01))

# save fig
# plt.savefig(
#     f'{dir_figs}/RelativeContributions_WO_Climate.png', 
#     dpi = 300,
#     bbox_inches = 'tight'
#     )
# %%
