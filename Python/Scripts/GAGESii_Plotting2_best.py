'''
BChoat 10/18/2022

Main script to produce results figures.

to do:

EDITED TO USE BEST PERFORMING MODEL FOR EACH STAID
...
'''

# %%
# Import libraries
################


import pandas as pd
import numpy as np
# import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from Load_Data import load_data_fun
# from statsmodels.distributions.empirical_distribution import ECDF


# %%
# define directory variables and specify scenario to work with


# Define directory variables
# directory with data to work with
# dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out' 
# dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results' 
dir_work = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'

# directory where SHAP values are located
# dir_shap = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'
dir_shap = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/SHAP_OUT'

# save figs (True of False?)
save_figs = False
# directory where to write figs
# dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/SHAP'
dir_figs = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/SHAP'


# directory where to place outputs
# dir_umaphd = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'

# which partition to use "train" or  "valnit"
# part_in = 'train'
part_in = 'valnit'

# cluster method to use
# ['None', 'Class', 'AggEcoregion', 'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1', 
#         'CAMELS', 'HLR', 'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4']
# clust_meth_in = ['AggEcoregion']
clust_meth_in = ['None', 'Class', 'AggEcoregion']
# which model to calc shaps for
# model_in = ['XGBoost']
# which models to include
# model_in = ['regr_precip', 'strd_mlr', 'XGBoost']
model_in = ['XGBoost']
# which metric to use when plotting eCDFs (')
metric_in = 'NSE'
# drop noise?
drop_noise = False

# plot parameters
cmap_title = 'Impact on model output; mean(SHAP)/mean(Q) [cm/cm]'

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
    f'{dir_shap}/MeanShap_{part_in}_mean_annual_normQ.csv'
)
# annual
df_shap_annual = pd.read_csv(
    f'{dir_shap}/MeanShap_{part_in}_annual_normQ.csv'
)
# mean annual
df_shap_monthly = pd.read_csv(
    f'{dir_shap}/MeanShap_{part_in}_monthly_normQ.csv'
)



# read in feature categories to be used for subsetting explanatory vars
# into cats of interest
feat_cats = pd.read_csv(
    # 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
    # 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories_wUnits.csv'
    
    )
# feat_cats['Features'] = feat_cats['Features'].str.replace('TS_', '')

# dir holding outputs from HPC runs.
dir_removed = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                'HPC_Files/GAGES_Work/data_out/'


#####
# explanatory variables
#####
# load data to get colnames for output dataframe (all expl vars)
# df_expl, df_WY, df_ID = load_data_fun(
#     dir_work = dir_work, 
#     time_scale = timescale,
#     train_val = part_in[0],
#     clust_meth = 'AggEcoregion',
#     region = 'MxWdShld',
#     standardize = False # whether or not to standardize data
#     )

# %% define function that gets explanatory variable and return an array for directional relationships
# between them and shap values


def returnRel(dir_work, dir_removed, timescale, STAID_in,
                part_in, clust_meth, region_in, shap_in, stdz):
    '''
    function reads in explantory variables for current site and timescale and returns
    array indicating the directional relationship between explanatory variables and 
    shap values

    input
    ------------------
    dir_work (str): dir to be passed to load function where data is stored
    dir_removed (str): dir holding tables indicating which variables were remove due to 
        VIF>threshold
    timescale (str): which time scale working with ('monthly', 'annual', 
                        'mean_annual')
    STAID_in (str): numeric value of station id, passed as a string
    part_in (str): 'train' or 'valnit' (for test data)
    clust_meth (str): what clusting method to use
    region_in (str): name of region or group working with
    shap_in (array): shap values for observations at site to be regressed against
                        expl vars
    stdz (bool): should data be standaredized?

    output
    -----------------
    array indicating directional relationship between explanatory variables and 
            shap values

    '''

    import numpy as np
    import glob
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # between shap values and explantory variables
    df_expl, df_WY, df_ID = load_data_fun(
        dir_work = dir_work, 
        time_scale = timescale,
        train_val = part_in,
        clust_meth = clust_meth,
        region = region_in,
        standardize = np.array(stdz) # whether or not to standardize data
        )

    # get shap values for staids in df_expl
    shap_in = shap_in[shap_in['STAID'].isin(df_expl['STAID'])]
    shap_in = shap_in.drop(['tmin_1', 'tmax_1', 'prcp_1', 'vp_1', 'swe_1'], axis=1)
    shap_in = shap_in.dropna(axis=1, how='all')
    shap_in.replace(np.nan, 0, inplace=True)

    # del(df_WY, df_ID)
    # DRAIN_SQKM stored as DRAIN_SQKM_x in some output by accident,
    # so replace it
    if 'SQKM_x' in df_expl.columns.values:
        df_expl.columns = df_expl.columns.str.replace('SQKM_x', 'SQKM')

    # define var holding all explanatory var columns names
    # expl_names = df_expl.columns
    # df_expl_in = df_expl[STAID['STAID'] == row['STAID']].reset_index(drop = True)

    # remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
    # subset WY to version desired (ft)
    # store staid's and date/year/month
    if(timescale == 'mean_annual'):
        STAID_work = df_expl[['STAID']]  
        df_expl.drop('STAID', axis = 1, inplace = True)
    if(timescale == 'annual'):
        STAID_work = df_expl[['STAID', 'year']]  
        df_expl.drop(['STAID', 'year'], axis = 1, inplace = True)
    if(timescale == 'monthly'):
        STAID_work = df_expl[['STAID', 'year', 'month']]   
        df_expl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)

    # read in columns that were previously removed due to high VIF
    file = glob.glob(
        f'{dir_removed}{timescale}/VIF_Removed/*{clust_meth}'\
            f'_{region_in}.csv'
        )[0]
    try:
        vif_removed = pd.read_csv(
            file
        )['columns_Removed']
    except:
        vif_removed = pd.read_csv(
        file
        )['Columns_Removed']

    if 'DRAIN_SQKM_x' in vif_removed.values:
        vif_removed = vif_removed.str.replace(
            'DRAIN_SQKM_x', 'DRAIN_SQKM'
        )

    # make sure only columns in current df_expl are removed
    # vif_removed = [x in vif_removed.values if x in df_expl.columns] # vif_removed.intersection(df_expl.columns)

    # drop columns that were removed due to high VIF
    df_expl.drop(vif_removed, axis = 1, inplace = True)

    # df_expl_in = df_expl[STAID_work['STAID'] == STAID_in].reset_index(drop = True)
                         # row['STAID']].reset_index(drop = True)
    # df_WY_in = df_WY[STAID['STAID'] == row['STAID']].reset_index(drop = True)

    X_in = df_expl.copy()

    # standardaize the explanatory vars for getting slope direction
                # using linear regression
    scaler = StandardScaler()
    scaler.fit(X_in)

    df_expl_instand = pd.DataFrame(
        scaler.transform(X_in),
        columns = X_in.columns
        )

    # define empty list to hold multipliers for direction of
    # shap relationship
    shap_dirxn = []
    # loop through all vars and get direction of relationship
    for colname in df_expl_instand.columns:
        x = np.array(df_expl_instand[colname]).reshape(-1,1)
        y = shap_in[colname]

        lm = LinearRegression()
        lm.fit(x, y)
        if lm.coef_ < 0:
            dirxn_temp = -1
        else:
            dirxn_temp = 1

        # append multiplier to shap_dir
        shap_dirxn.append(dirxn_temp)


    # return array with directional relationships to SHAP values
    return(np.array(shap_dirxn), df_expl)

    # # take mean of shap values, and give direction by 
    # # multiplying by shap_coef
    # df_shapmean = pd.DataFrame(
    #     df_shap_valout.abs().mean() * np.array(shap_dirxn),
    # ).T



# def returnRel(dir_work, timescale, part_in, clust_meth, region_in, shap_in, stdz):
# dir_work, dir_removed, timescale, STAID_in,
#                 part_in, clust_meth, region_in, shap_in, stdz

# test
# read in data
# df_shap_mannual = pd.read_csv(
#     f'{dir_shap}/MeanShap_BestGrouping_All_mean_annual_normQ.csv',
#     dtype = {'STAID': str}
# ).drop_duplicates()

# dir_work = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'
# timescale = 'mean_annual'
# STAID_in = '01059000'
# clust_meth = 'AggEcoregion'
# part_in = df_id.loc[(df_id['clust_method'] == clust_meth) & (df_id['STAID'] == STAID_in), 'train_val'].iloc[0]
# region_in = df_id.loc[(df_id['clust_method'] == clust_meth) & (df_id['STAID'] == STAID_in), 'region'].iloc[0]
# shap_in = df_shap_mannual.copy() # loc[
#         # (df_shap_mannual['region'] == region_in) & \
#         #     (df_shap_mannual['train_val'] == part_in)] # \
#                 # (df_shap_mannual['STAID'] == STAID_in)] # , 'TS_NLCD_11']
# stdz = True



# df_test = returnRel(dir_work, dir_removed, timescale, STAID_in, part_in, clust_meth, region_in, shap_in, stdz)


###############
# %% define features that are natural features
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




# %% create a custom color map to associate specific colors with regions
############################
from matplotlib.colors import ListedColormap
color_dict = {
    'CntlPlains': 'blue',
    'EastHghlnds': 'cyan',
    'MxWdShld': 'darkgoldenrod',
    'NorthEast': 'orange',
    'SECstPlain': 'green',
    'SEPlains': 'lime',
    'WestMnts': 'firebrick',
    'WestPlains': 'tomato',
    'WestXeric': 'saddlebrown',
    'Non-ref': 'purple',
    'Ref': 'palevioletred',
    'All': 'dimgray'
}
custom_cmap = ListedColormap([color_dict[key] for key in color_dict])
custom_palette = [color_dict[key] for key in color_dict]

# %%
# eCDF plots (all three timescales)
##############
# metric_in = 'NSE'

cmap_str = custom_palette

# mean annual
data_in_mannual = df_resind_mannual.copy() #[
#     df_resind_mannual['train_val'] == part_in
# ]

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
data_in_annual = df_resind_annual.copy() # [
#     df_resind_annual['train_val'] == part_in
# ]

# data_in_annual = pd.merge(
#     data_in_annual, df_shap_annual,
#     left_on = ['region', 'clust_method', 'model'],
#     right_on = ['region', 'clust_meth', 'best_model']
# )[df_resind_annual.columns]

data_in_annual.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


# month
data_in_month = df_resind_monthly.copy() # [
#     df_resind_monthly['train_val'] == part_in
# ]

# data_in_month = pd.merge(
#     data_in_month, df_shap_monthly,
#     left_on = ['region', 'clust_method', 'model'],
#     right_on = ['region', 'clust_meth', 'best_model']
# )[df_resind_monthly.columns]


data_in_month.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


####


data_in_mannual = data_in_mannual[
    (data_in_mannual['clust_method'].isin(clust_meth_in)) &
    (data_in_mannual['model'].isin(model_in))
]

data_in_annual = data_in_annual[
    (data_in_annual['clust_method'].isin(clust_meth_in)) &
    (data_in_annual['model'].isin(model_in))
]

data_in_month = data_in_month[
    (data_in_month['clust_method'].isin(clust_meth_in)) &
    (data_in_month['model'].isin(model_in))
]

if drop_noise:
    data_in_mannual = data_in_mannual[
        data_in_mannual['region'] != '-1'
    ]
    data_in_annual = data_in_annual[
        data_in_annual['region'] != '-1'
    ]
    data_in_month = data_in_month[
        data_in_month['region'] != '-1'
    ]


# %% get best results for each df
# drop_regions = ['All', 'Ref', 'Non-ref']
drop_regions = []
## mean annual
# df_best_mannual = pd.DataFrame(columns = data_in_mannual.columns)
df_best_mannual = data_in_mannual.copy().drop_duplicates()
# get ids where best score is
temp_ids = df_best_mannual.groupby('STAID')['|residuals|'].idxmin()
df_best_mannual = df_best_mannual.loc[temp_ids]

# get temp id dataframe to assign agg ecoregions to all staids

# if want to drop other labels run query
# df_id = data_in_mannual.query('clust_method == "AggEcoregion"')
# df_id = df_id[['STAID', 'region']]
# df_id.columns = ['STAID', 'Ecoregion']
# df_best_mannual = pd.merge(df_best_mannual, df_id, on = 'STAID')

# get df_id with a column for each grouping approach
clust_temp = data_in_mannual.clust_method.unique()
dfstemp = [
    data_in_mannual.query(
        'clust_method == @x'
        ) for x in clust_temp # 
        ]
# df_id = reduce(lambda left, right: pd.merge(left, right, on = ['STAID']), 
#                 dfstemp)
# tempcols = ['STAID']
# tempcols.extend(clust_temp)
# df_id.columns = tempcols
df_id = pd.concat(dfstemp)
# df_id.columns = ['STAID', 'clust_method', 'Region', 'model', 'time_scale']


## annual 

# df_best_annual = pd.DataFrame(columns = data_in_annual.columns)
df_best_annual = data_in_annual.copy().drop_duplicates()
# get ids where best score is
temp_ids = df_best_annual.groupby('STAID')[metric_in].idxmax()
df_best_annual = df_best_annual.loc[temp_ids]

# get temp id dataframe to assign agg ecoregions to all staids
# df_id = data_in_annual.query('clust_method == "AggEcoregion"')
# df_id = df_id[['STAID', 'region']]
# df_id.columns = ['STAID', 'Ecoregion']
# df_best_annual = pd.merge(df_best_annual, df_id, on = 'STAID')

df_best_annual = pd.merge(df_best_annual, df_id, on = 'STAID', how = 'right')

## monthly 

# df_best_monthly = pd.DataFrame(columns = data_in_monthly.columns)
df_best_monthly = data_in_month.copy().drop_duplicates()
# get ids where best score is
temp_ids = df_best_monthly.groupby('STAID')[metric_in].idxmax()
df_best_monthly = df_best_monthly.loc[temp_ids]

# get temp id dataframe to assign agg ecoregions to all staids
# df_id = data_in_month.query('clust_method == "AggEcoregion"')
# df_id = df_id[['STAID', 'region']]
# df_id.columns = ['STAID', 'Ecoregion']
df_best_monthly = pd.merge(df_best_monthly, df_id, on = 'STAID')



cmap_str = cmap_str[0:(len(cmap_str) - len(drop_regions))]
# for reg in drop_regions:
#     del color_dict[reg]

# %%

####

fig, axs = plt.subplots(2, 3, figsize = (10, 8), sharey = True)#, sharex = True)
ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

# training
ax1.set_xlim(0, 40)
ax1.annotate('(a)', xy = (2, 0.3)) #(80, 0.02))
ax1.grid()
ax1.title.set_text('Mean Annual')
sns.ecdfplot(
    data = df_best_mannual.query('train_val == "train"'),
    x = '|residuals|',
    hue = 'Region',
    linestyle = '--',
    palette = cmap_str,
    hue_order = color_dict.keys(),
    ax = ax1
)
ax1.set(xlabel = '', # '|residuals| [cm]', 
        ylabel = 'Non-Exceedence Probability (Training Data)')
sns.move_legend(ax1, 'lower right', ncol = 1)
ax1.get_legend().set_title('Region')


ax2.set_xlim(-1, 1)
ax2.annotate('(b)', xy = (-0.9, 0.3)) # (0.8, 0.02))
ax2.grid()
ax2.title.set_text('Annual')
sns.ecdfplot(
    data = df_best_annual.query('train_val == "train"'),
    x = metric_in,
    hue = 'Region',
    hue_order = color_dict.keys(),
    linestyle = '--',
    palette = cmap_str,
    ax = ax2,
    legend = False
)
ax2.set(xlabel = '') #metric_in)


ax3.set_xlim(-1, 1)
ax3.annotate('(c)', xy = (-0.9, 0.3)) # (0.8, 0.02))
ax3.grid()
ax3.title.set_text('Monthly')
sns.ecdfplot(
    data = df_best_monthly.query('train_val == "train"'),
    x = metric_in,
    hue = 'Region',
    hue_order = color_dict.keys(),
    linestyle = '--',
    palette = cmap_str,
    ax = ax3,
    legend = False
)
ax3.set(xlabel = '') # metric_in)

# testing
ax4.set(xlabel = '|residuals| [cm]', 
        ylabel = 'Non-Exceedence Probability (Testing Data)')
ax4.set_xlim(0, 40)
ax4.annotate('(d)', xy = (2, 0.3)) #(80, 0.02))
ax4.grid()
ax4.title.set_text('') # Mean Annual')
sns.ecdfplot(
    data = df_best_mannual.query('train_val == "valnit"'),
    x = '|residuals|',
    hue = 'Region',
    hue_order = color_dict.keys(),
    linestyle = '--',
    palette = cmap_str,
    ax = ax4,
    legend = False
)
# sns.move_legend(ax1, 'lower center')
# ax1.get_legend().set_title('Region')


ax5.set_xlim(-1, 1)
ax5.set(xlabel = metric_in)
ax5.annotate('(e)', xy = (-0.9, 0.3)) # (0.8, 0.02))
ax5.grid()
ax5.title.set_text('') # Annual')
sns.ecdfplot(
    data = df_best_annual.query('train_val == "valnit"'),
    x = metric_in,
    hue = 'Region',
    hue_order = color_dict.keys(),
    linestyle = '--',
    palette = cmap_str,
    ax = ax5,
    legend = False
)


ax6.set_xlim(-1, 1)
ax6.set(xlabel = metric_in)
ax6.annotate('(f)', xy = (-0.9, 0.3)) # (0.8, 0.02))
ax6.grid()
ax6.title.set_text('') # Monthly')
sns.ecdfplot(
    data = df_best_monthly.query('train_val == "valnit"'),
    x = metric_in,
    hue = 'Region',
    hue_order = color_dict.keys(),
    linestyle = '--',
    palette = cmap_str,
    ax = ax6,
    legend = False
)


model_in2 = 'And'.join(model_in)
clust_meth_in2 = 'And'.join(clust_meth_in)
# save fig
if save_figs:
    plt.savefig(
        # f'{dir_figs}/ecdfs_{part_in}_{clust_meth_in}_{model_in}.png', 
        f'{dir_figs}/ecdfs_TrainTest_BestScores_AllRegions_{metric_in}_{clust_meth_in2}_{model_in2}.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )
else:
    plt.show()





# %%
# get df_id with a column for each grouping approach
df_best_mannual = pd.merge(df_best_mannual, 
                            df_id[['STAID', 'region']], 
                            on = 'STAID', 
                            how = 'right')
df_best_mannual.columns = df_best_mannual.columns.str.replace('region_x', 'region')




# %%
# heatmap of shap values using all variables
##############
# part_in = 'train'
part_in = 'valnit'
# define list of columns to drop if they are in shap dfs
coldrops = ['residuals', '|residuals|', 'STAID', 'NSE', 'KGE',
            'r', 'alpha', 'beta', 'PercBias', 'RMSE',
            'clust_method', 'region', 'train_val', 'time_scale',
            'model']

df_shap_mannual = pd.read_csv(
    # f'{dir_shap}/MeanShap_{part_in}_mean_annual_normQ.csv'
    # f'{dir_shap}/MeanShap_BestModel_All_mean_annual_normQ.csv',
    f'{dir_shap}/MeanShap_BestGrouping_All_mean_annual_normQ.csv',
    dtype = {'STAID': str}
)

# df_shap_mannual['prcp'].hist(bins=100)
# annual
df_shap_annual = pd.read_csv(
    # f'{dir_shap}/MeanShap_{part_in}_annual_normQ.csv'
    # f'{dir_shap}/MeanShap_BestModel_All_annual_normQ.csv',
    f'{dir_shap}/MeanShap_BestGrouping_All_annual_normQ.csv',
    dtype = {'STAID': str}
)
# mean annual
df_shap_monthly = pd.read_csv(
    # f'{dir_shap}/MeanShap_{part_in}_monthly_normQ.csv'
    # f'{dir_shap}/MeanShap_BestModel_All_monthly_normQ.csv',
    f'{dir_shap}/MeanShap_BestGrouping_All_monthly_normQ.csv',
    dtype = {'STAID': str}
)

# join df_id and shap dataframes, then take mean for each region
df_shap_mannual = pd.merge(df_shap_mannual, df_id, 
                    on = 'STAID', how = 'right')

# filter to partition working with
df_shap_mannual = df_shap_mannual.query("train_val == @part_in")
df_shap_annual = df_shap_annual.query("train_val == @part_in")
df_shap_monthly = df_shap_monthly.query("train_val == @part_in")

# take mean by grouping approach
df_shap_mannual = df_shap_mannual.groupby('Region').mean().reset_index()

# clust_meth_in = 'Nat_3'
# model_in = 'XGBoost'
# metric_in = 'KGE'

vmin_mannual = -0.15
vmax_mannual = 0.15

vmin_annual = vmin_mannual
vmax_annual = vmax_mannual

vmin_month = vmin_mannual
vmax_month = vmax_mannual

# read in data
# mean annual
# shap_mannual = df_shap_mannual[df_shap_mannual['clust_meth'].isin(clust_meth_in)]
shap_mannual = df_shap_mannual.drop(
    ['Region', 'residuals', '|residuals|', # 'clust_meth', 'best_model', 'best_score',
     'tmin_1', 'tmax_1', 'prcp_1', 'vp_1', 'swe_1'], axis = 1
)
# shap_mannual.index = df_shap_mannual.loc[
#     df_shap_mannual['clust_meth'].isin(clust_meth_in), 'region'
#                     ]
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

##############
 # define empty list to hold multipliers for direction of
# shap relationship
shap_dirxn = []
# loop through all vars and get direction of relationship
for colname in X_in.columns:
    x = np.array(df_expl_instand[colname]).reshape(-1,1)
    y = df_shap_valout[colname]

    lm = LinearRegression()
    lm.fit(x, y)
    if lm.coef_ < 0:
        dirxn_temp = -1
    else:
        dirxn_temp = 1

    # append multiplier to shap_dir
    shap_dirxn.append(dirxn_temp)


# take mean of shap values, and give direction by 
# multiplying by shap_coef
df_shapmean = pd.DataFrame(
    df_shap_valout.abs().mean() * np.array(shap_dirxn),
).T


##############




# annual
shap_annual = df_shap_annual[df_shap_annual['clust_meth'].isin(clust_meth_in)]
shap_annual = shap_annual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_annual.index = df_shap_annual.loc[
    df_shap_annual['clust_meth'].isin(clust_meth_in), 'region'
    ]
shap_annual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)

# shap_annual.columns = shap_annual.columns.str.replace(
#     'TS_', ''
# )

# monthly
shap_monthly = df_shap_monthly[df_shap_monthly['clust_meth'].isin(clust_meth_in)]
shap_monthly = shap_monthly.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_monthly.index = df_shap_monthly.loc[
    df_shap_monthly['clust_meth'].isin(clust_meth_in), 'region'
    ]
shap_monthly.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
shap_monthly['Ant Precip'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('prcp_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant SWE'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('swe_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Vapor Pressure'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('vp_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Max Temp'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('tmax_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Min Temp'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('tmin_')]
    ].apply('sum', axis = 1)


shap_monthly = shap_monthly.drop(
    shap_monthly.columns[
        (shap_monthly.columns.str.contains('prcp_')) | \
        (shap_monthly.columns.str.contains('swe_')) | \
        (shap_monthly.columns.str.contains('vp_')) | \
        (shap_monthly.columns.str.contains('tmax_')) | \
        (shap_monthly.columns.str.contains('tmin_')) | \
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
                        figsize = (11, 8)) # ,
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
    cbar = False
    # cbar_kws = {'label': 'SHAP (Impact on model output)',
    #             'use_gridspec': False,
    #             'location': 'bottom'} # ,
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
    cbar_kws = {'label': cmap_title,
                'extend': 'both',
                'ticks': [vmin_annual, 0, vmax_annual],
                'shrink': 1.5,
                'pad': 0.01,
                'use_gridspec': False,
                'anchor': (0.9, 0), # 'SW',
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
# if save_figs:
#     plt.savefig(
#         f'{dir_figs}/Heat_AllVars_{part_in}_{clust_meth_in}_{model_in}.png', # Heat_AllVars_annual_Horz.png', 
#         dpi = 300,
#         bbox_inches = 'tight'
#         )



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
    cbar = False
    # cbar_kws = {'label': 'SHAP (Impact on model output)',
    #             'use_gridspec': False,
    #             'location': 'bottom'} # ,
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
ax1.annotate('(a)', xy = (-1.5, 30.5), annotation_clip = False)
ax2.annotate('(b)', xy = (-1.5, 30.5), annotation_clip = False)
ax3.annotate('(c)', xy = (-1.5, 30.5), annotation_clip = False)

# save fig
if save_figs:
    plt.savefig(
        f'{dir_figs}/Heat_AllVars_AllTimescales_{part_in}_{clust_meth_in}_{model_in}.png', # Heat_AllVars_monthly_Horz.png
        dpi = 300,
        bbox_inches = 'tight'
        )
else:
    plt.show()





# %%
# heatmap of shap values using Anthropogenic variables
##############


# clust_meth_in = 'Nat_3'
# model_in = 'XGBoost'
# metric_in = 'KGE'

vmin_mannual = -0.025
vmax_mannual = 0.025

vmin_annual = vmin_mannual
vmax_annual = vmax_mannual

vmin_month = vmin_mannual
vmax_month = vmax_mannual


# read in data
# mean annual
shap_mannual = df_shap_mannual[df_shap_mannual['clust_meth'].isin(clust_meth_in)]
shap_mannual = shap_mannual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score',
     'tmin_1', 'tmax_1', 'prcp_1', 'vp_1', 'swe_1'], axis = 1
)
shap_mannual.index = df_shap_mannual.loc[
    df_shap_mannual['clust_meth'].isin(clust_meth_in), 'region'
                    ]
shap_mannual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)

# shap_mannual.columns = shap_mannual.columns.str.replace(
#     'TS_', ''
# )

# annual
shap_annual = df_shap_annual[df_shap_annual['clust_meth'].isin(clust_meth_in)]
shap_annual = shap_annual.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_annual.index = df_shap_annual.loc[
    df_shap_annual['clust_meth'].isin(clust_meth_in), 'region'
    ]
shap_annual.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)

# monthly
shap_monthly = df_shap_monthly[df_shap_monthly['clust_meth'].isin(clust_meth_in)]
shap_monthly = shap_monthly.drop(
    ['clust_meth', 'region', 'best_model', 'best_score'], axis = 1
)
shap_monthly.index = df_shap_monthly.loc[
    df_shap_monthly['clust_meth'].isin(clust_meth_in), 'region'
    ]
shap_monthly.rename(
    columns = dict(zip(feat_cats['Features'], feat_cats['Alias'])),
    inplace = True
)
shap_monthly['Ant Precip'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('prcp_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant SWE'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('swe_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Vapor Pressure'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('vp_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Max Temp'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('tmax_')]
    ].apply('sum', axis = 1)
shap_monthly['Ant Min Temp'] = shap_monthly[
    shap_monthly.columns[shap_monthly.columns.str.contains('tmin_')]
    ].apply('sum', axis = 1)


shap_monthly = shap_monthly.drop(
    shap_monthly.columns[
        (shap_monthly.columns.str.contains('prcp_')) | \
        (shap_monthly.columns.str.contains('swe_')) | \
        (shap_monthly.columns.str.contains('vp_')) | \
        (shap_monthly.columns.str.contains('tmax_')) | \
        (shap_monthly.columns.str.contains('tmin_')) | \
        (shap_monthly.columns.str.contains('vp_'))
    ],  axis = 1  
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
    vmin = vmin_mannual, # np.min(np.min(data_mannualin.iloc[:, 0:30])),
    vmax = vmax_mannual,
    cbar = False
    # cbar_kws = {'label': 'SHAP (Impact on model output)',
    #             'use_gridspec': False,
    #             'location': 'bottom'} # ,
    # robust = True
    )
ax1.set(xlabel = 'Region', ylabel = 'Explanatory Variables') # ylabel = 'Region') # xlabel = 'Explanatory Varaibles',
# ax1.set_xticklabels(mannual_data_plot.columns, ha = 'center')

ax1.tick_params(axis = 'x',
                rotation = 90)
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
    vmin = vmin_annual, # np.min(np.min(data_annualin.iloc[:, 0:30])),
    vmax = vmax_annual,
    cbar_kws = {'label': cmap_title,
                'extend': 'both',
                'ticks': [vmin_annual, 0, vmax_annual],
                'use_gridspec': False,
                'anchor': (1, 0), # 'SW',
                'shrink': 1.5,
                'pad': 0.01,
                'location': 'bottom'} # ,
    # robust = True
    )

ax2.set(xlabel = 'Region') # ylabel = 'Region') # (xlabel = 'Explanatory Varaibles', 
# ax2.set_xticklabels(annual_data_plot.columns, ha = 'center')
ax2.tick_params(axis = 'x',
                rotation = 90)

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
    vmin = vmin_month, # np.min(np.min(data_monthlyin.iloc[:, 0:30])),
    vmax = vmax_month,
    cbar = False
    # cbar_kws = {'label': 'SHAP (Impact on model output)',
    #             'use_gridspec': False,
    #             'location': 'bottom'} # ,
    # robust = True
    )
ax3.set(xlabel = 'Region', ylabel = '') 
ax3.set_yticks([x+0.5 for x in range(len(monthly_data_plot.index))])
ax3.set_yticklabels(monthly_data_plot.index)
# ax3.tick_params(axis = 'y', labelrotation = 45)
ax3.tick_params(axis = 'x',
                rotation = 90)
# color anthropogenic hydrologic alteration variables
for x in np.where(monthly_data_plot.index.isin(anhyd_feats))[0]: # np.arange(5, 20, 1): 
    ax3.get_yticklabels()[x].set_color('red')


# annotate 
# annotate 
ax1.annotate('(a)', xy = (-1.5, 22.5), annotation_clip = False)
ax2.annotate('(b)', xy = (-1.5, 22.5), annotation_clip = False)
ax3.annotate('(c)', xy = (-1.5, 22.5), annotation_clip = False)


# save fig
if save_figs:
    plt.savefig(
        f'{dir_figs}/Heat_AnthVars_{part_in}_{clust_meth_in}_{model_in}.png',  # Heat_AnthVars_monthly_Horz.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )
    
else:
    plt.show()






##############################################################
#############################################################
##############################################################



# %%
# calculate total mean SHAP for each variable type




# define features that are climate
clim_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Climate', 'Features'
].reset_index(drop = True)
clim_feats = clim_feats.str.replace('TS_', '')

# define features that are Physiography features
phys_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Physiography', 'Features'
].reset_index(drop = True)
phys_feats = phys_feats.str.replace('TS_', '')

# define features that are Anthro_Hydro
anhyd_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Anthro_Hydro', 'Features'
].reset_index(drop = True)
anhyd_feats = anhyd_feats.str.replace('TS_', '')

# define features that are Anthro_Land features
anland_feats = feat_cats.loc[
    feat_cats['Coarse_Cat'] == 'Anthro_Land', 'Features'
].reset_index(drop = True)
anland_feats = anhyd_feats.str.replace('TS_', '')

# mean annual

df_shap_mannual.columns = df_shap_mannual.columns.str.replace('TS_', '')
df_shap_mannual = df_shap_mannual[df_shap_mannual['clust_meth'].isin(clust_meth_in)]

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
df_shap_annual = df_shap_annual[df_shap_annual['clust_meth'].isin(clust_meth_in)]

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
df_shap_monthly = df_shap_monthly[df_shap_monthly['clust_meth'].isin(clust_meth_in)]

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
if not df_shaptotout_mannual.index.name == 'Region':
    df_shaptotout_mannual.index = df_shaptotout_mannual['Region']
    df_shaptotout_mannual.drop('Region', axis = 1, inplace = True)


df_inmannual = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)

# # normalize data
df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']/df_shaptotout_mannual.sum(axis = 1)
df_inmannual['Physiography'] = df_shaptotout_mannual['Physiography_Impact']/df_shaptotout_mannual.sum(axis = 1)
df_inmannual['AnthroHydro'] = df_shaptotout_mannual['AnthroHydro_Impact']/df_shaptotout_mannual.sum(axis = 1)
df_inmannual['AnthroLand'] = df_shaptotout_mannual['AnthroLand_Impact']/df_shaptotout_mannual.sum(axis = 1)
# unnormalized data
# df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']
# df_inmannual['Physiography'] = df_shaptotout_mannual['Physiography_Impact']
# df_inmannual['AnthroHydro'] = df_shaptotout_mannual['AnthroHydro_Impact']
# df_inmannual['AnthroLand'] = df_shaptotout_mannual['AnthroLand_Impact']


# annual
if not df_shaptotout_annual.index.name == 'Region':
    df_shaptotout_annual.index = df_shaptotout_annual['Region']
    df_shaptotout_annual.drop('Region', axis = 1, inplace = True)

df_inannual = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)
# # Normalize data
df_inannual['Climate'] = df_shaptotout_annual['Climate_Impact']/df_shaptotout_annual.sum(axis = 1)
df_inannual['Physiography'] = df_shaptotout_annual['Physiography_Impact']/df_shaptotout_annual.sum(axis = 1)
df_inannual['AnthroHydro'] = df_shaptotout_annual['AnthroHydro_Impact']/df_shaptotout_annual.sum(axis = 1)
df_inannual['AnthroLand'] = df_shaptotout_annual['AnthroLand_Impact']/df_shaptotout_annual.sum(axis = 1)
# unormalized data
# df_inannual['Climate'] = df_shaptotout_annual['Climate_Impact']
# df_inannual['Physiography'] = df_shaptotout_annual['Physiography_Impact']
# df_inannual['AnthroHydro'] = df_shaptotout_annual['AnthroHydro_Impact']
# df_inannual['AnthroLand'] = df_shaptotout_annual['AnthroLand_Impact']


# monthly
if not df_shaptotout_monthly.index.name == 'Region':
    df_shaptotout_monthly.index = df_shaptotout_monthly['Region']
    df_shaptotout_monthly.drop('Region', axis = 1, inplace = True)

df_inmonthly = pd.DataFrame(
    columns = ['Climate', 'Physiography', 'AnthroHydro', 'AnthroLand']
)

# # Normalize data
df_inmonthly['Climate'] = df_shaptotout_monthly['Climate_Impact']/df_shaptotout_monthly.sum(axis = 1)
df_inmonthly['Physiography'] = df_shaptotout_monthly['Physiography_Impact']/df_shaptotout_monthly.sum(axis = 1)
df_inmonthly['AnthroHydro'] = df_shaptotout_monthly['AnthroHydro_Impact']/df_shaptotout_monthly.sum(axis = 1)
df_inmonthly['AnthroLand'] = df_shaptotout_monthly['AnthroLand_Impact']/df_shaptotout_monthly.sum(axis = 1)

# unnormalize data
# df_inmonthly['Climate'] = df_shaptotout_monthly['Climate_Impact']
# df_inmonthly['Physiography'] = df_shaptotout_monthly['Physiography_Impact']
# df_inmonthly['AnthroHydro'] = df_shaptotout_monthly['AnthroHydro_Impact']
# df_inmonthly['AnthroLand'] = df_shaptotout_monthly['AnthroLand_Impact']


# df_inmannual = df_inmannual.reindex(df_inannual.columns, axis = 1)

# plot
color = ['blue', 'saddlebrown', 'red', 'black']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True)

ax1.title.set_text('Mean Annual')
df_inmannual.plot(
    kind = 'bar', stacked = True, ax = ax1,
    color = color, width = 0.8
)
ax1.set(ylabel = 'Relative contribution of each variable type')
ax1.annotate('(a)', xy = (10.51, 1.01))


ax2.title.set_text('Annual')
df_inannual.plot(
    kind = 'bar', stacked = True, ax = ax2, legend = False,
    color = color, width = 0.8
)
ax2.annotate('(b)', xy = (10.51, 1.01))

ax3.title.set_text('Monthly')
df_inmonthly.plot(
    kind = 'bar', stacked = True, ax = ax3, legend = False,
    color = color, width = 0.8
)
ax3.annotate('(c)', xy = (10.51, 1.01))

# save fig
if save_figs:
    plt.savefig(
        f'{dir_figs}/RelativeContributions_{part_in}_{clust_meth_in}_{model_in}.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )
else:
    plt.show()


# %%
# df_inmannual['Climate'] = df_shaptotout_mannual['Climate_Impact']/df_shaptotout_mannual.sum(axis = 1)
if not df_shaptotout_mannual.index.name == 'Region':
    df_shaptotout_mannual.index = df_shaptotout_mannual['Region']
    df_shaptotout_mannual.drop('Region', axis = 1, inplace = True)

df_inmannual = pd.DataFrame(
    columns = ['Physiography', 'AnthroHydro', 'AnthroLand']
)

# define temp shaptout dataframe to work with
df_shaptout_temp = df_shaptotout_mannual.copy()
if 'Climate_Impact' in df_shaptout_temp.columns:
    df_shaptout_temp = df_shaptout_temp.drop('Climate_Impact', axis = 1)
# df_shaptout_temp.drop('Climate_Impact', axis = 1, inplace = True)
df_inmannual['Physiography'] = df_shaptout_temp['Physiography_Impact']/df_shaptout_temp.sum(axis = 1)
df_inmannual['AnthroHydro'] = df_shaptout_temp['AnthroHydro_Impact']/df_shaptout_temp.sum(axis = 1)
df_inmannual['AnthroLand'] = df_shaptout_temp['AnthroLand_Impact']/df_shaptout_temp.sum(axis = 1)

# annual
# df_shaptotout_annual.drop('Climate_Impact', axis = 1, inplace = True)
if not df_shaptotout_annual.index.name == 'Region':
    df_shaptotout_annual.index = df_shaptotout_annual['Region']
    df_shaptotout_annual.drop('Region', axis = 1, inplace = True)

df_inannual = pd.DataFrame(
    columns = ['Physiography', 'AnthroHydro', 'AnthroLand']
)

df_shaptout_temp = df_shaptotout_annual.copy()
if 'Climate_Impact' in df_shaptout_temp.columns:
    df_shaptout_temp = df_shaptout_temp.drop('Climate_Impact', axis = 1)
df_inannual['Physiography'] = df_shaptout_temp['Physiography_Impact']/df_shaptout_temp.sum(axis = 1)
df_inannual['AnthroHydro'] = df_shaptout_temp['AnthroHydro_Impact']/df_shaptout_temp.sum(axis = 1)
df_inannual['AnthroLand'] = df_shaptout_temp['AnthroLand_Impact']/df_shaptout_temp.sum(axis = 1)

# monthly
# df_shaptout_temp.drop('Climate_Impact', axis = 1, inplace = True)
if not df_shaptotout_mannual.index.name == 'Region':
    df_shaptout_temp.index = df_shaptout_temp['Region']
    df_shaptout_temp.drop('Region', axis = 1, inplace = True)

df_inmonthly = pd.DataFrame(
    columns = ['Physiography', 'AnthroHydro', 'AnthroLand']
)
df_shaptout_temp = df_shaptotout_monthly.copy()
if 'Climate_Impact' in df_shaptout_temp.columns:
    df_shaptout_temp = df_shaptout_temp.drop('Climate_Impact', axis = 1)
df_inmonthly['Physiography'] = df_shaptout_temp['Physiography_Impact']/df_shaptout_temp.sum(axis = 1)
df_inmonthly['AnthroHydro'] = df_shaptout_temp['AnthroHydro_Impact']/df_shaptout_temp.sum(axis = 1)
df_inmonthly['AnthroLand'] = df_shaptout_temp['AnthroLand_Impact']/df_shaptout_temp.sum(axis = 1)

# df_inannual.drop('Climate', axis = 1, inplace = True)
# df_inmonthly.drop('Climate', axis = 1, inplace = True)


# plot
color = ['saddlebrown', 'red', 'black']

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True)

ax1.title.set_text('Mean Annual')
df_inmannual.plot(
    kind = 'bar', stacked = True, ax = ax1,
    color = color, width = 0.8
)
ax1.set(ylabel = 'Relative contribution of each variable type')
ax1.annotate('(a)', xy = (10.51, 1.01))

ax2.title.set_text('Annual')
df_inannual.plot(
    kind = 'bar', stacked = True, ax = ax2, legend = False,
    color = color, width = 0.8
)
ax2.annotate('(b)', xy = (10.51, 1.01))



ax3.title.set_text('Monthly')
df_inmonthly.plot(
    kind = 'bar', stacked = True, ax = ax3, legend = False,
    color = color, width = 0.8
)
ax3.annotate('(c)', xy = (10.51, 1.01))

# save fig
if save_figs:
    plt.savefig(
        f'{dir_figs}/RelativeContributions_WO_Climate_{part_in}_{clust_meth_in}_{model_in}.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )
else:
    plt.show()
# %%
