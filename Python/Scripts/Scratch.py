import pandas as pd

df = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/PerfRankMonthAnnual_ByClustMethModel.csv')
df = df.query("clust_method in ['AggEcoregion', 'None', 'Class'] and train_val == 'valnit'")
df.sort_values(by = 'NSE_qmean', ascending=False).head(20)


############################

# %% plots of gains compared to xgboost
################################


import pandas as pd
from functools import reduce

# define function to calc gains

def getDiff(df_sum, regions_column):
    '''
    Take input dataframe and column name with regions in it and
    calculate gain in NSEm when using xgboost compared to other
    models. e.g., XGBoost - MLR

    Outputs dataframe holding gains
    '''
    # regions_column: column holding regions

    regions = []
    regr = []
    pcamlr = []
    mlr = []


    for region in df_sum[regions_column].unique():

        regions.extend([region])

        for model_str in ['regr_precip', 'strd_PCA_mlr', 'strd_mlr']:
            gain = df_sum.loc[
                (df_sum[regions_column] == region) & \
                    (df_sum['model'] == 'XGBoost'), 'NSE'].values - \
                        df_sum.loc[(df_sum[regions_column] == region) & \
                                    (df_sum['model'] == model_str), 'NSE'].values
            gain = gain.round(2)
            # print(region)

            if model_str == 'regr_precip':
                regr.extend(gain)
            elif model_str == 'strd_PCA_mlr':
                pcamlr.extend(gain)
            elif model_str == 'strd_mlr':
                mlr.extend(gain)
            else:
                raise NameError('model name is missing or not in dataframe')
            # print(len(regions), len(regr), len(pcamlr), len(mlr))
        
    df_out = pd.DataFrame({
        'region': regions,
        'SLR': regr,
        'PCA_MLR': pcamlr,
        'MLR': mlr
    }).sort_values(by = 'SLR')

    df_out = pd.melt(
        df_out, id_vars = [regions_column], var_name = 'Model',
        value_vars = ['SLR', 'PCA_MLR', 'MLR'],
        value_name = 'Gain' # r'Gain in $NSE_m$'
        )

    df_out = df_out.rename(columns={'region': 'Region'})

    return df_out
    

df_in = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv',
                 dtype={'STAID': 'string'})

df = df_in.query("clust_method in ['AggEcoregion', 'None', 'Class'] and train_val == 'valnit'")
df_ann = df.query('time_scale == "annual"') #  & model == "XGBoost"')
df_month = df.query('time_scale == "monthly" & model == "XGBoost"')

df_ann = df_ann.groupby(['clust_method', 'region', 'model', 'time_scale']).quantile(
    # [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    0.5
).reset_index().sort_values(by = "NSE")


df_month = df_month.groupby(['clust_method', 'region', 'model', 'time_scale']).quantile(
    # [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    0.5
).reset_index().sort_values(by = "NSE")





df_in.head()



##########################################
# %%
'''
Make plot;
- Bar plot
- x-axis: models(~XGBoost) within Regions/Groups
- y-axis: XGBoost-x_models performance (NSEm)
'''

part_in = ['train', 'valnit']
# part_in = ['train']
# part_in = ['valnit']


df = df_in.query("clust_method in ['AggEcoregion', 'None', 'Class']")
# " and train_val in @part_in")

# training                 
df_annTrain = df.query('time_scale == "annual"'\
                  " and train_val in ['train']")
df_monthTrain = df.query('time_scale == "monthly"'\
                    " and train_val in ['train']")
# quaniles
df_ansumTrain = df_annTrain.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()
df_mosumTrain = df_monthTrain.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()

# get gains
df_outanTrain = getDiff(df_ansumTrain, 'region')
df_outmoTrain = getDiff(df_mosumTrain, 'region')

# testing               
df_annTest = df.query('time_scale == "annual"'\
                  " and train_val in ['valnit']")
df_monthTest = df.query('time_scale == "monthly"'\
                    " and train_val in ['valnit']")
# quaniles
df_ansumTest = df_annTest.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()
df_mosumTest = df_monthTest.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()

# get gains
df_outanTest = getDiff(df_ansumTest, 'region')
df_outmoTest = getDiff(df_mosumTest, 'region')


x_suf = ' in $NSE_m$ (annual)'
y_suf = ' in $NSE_m$ (monthly)'

data_frames = [df_outanTrain, df_outanTest, df_outmoTrain, df_outmoTest]

df_plot = reduce(lambda  left,right: pd.merge(left, right, 
                                            on = ['Region', 'Model']), 
                                            data_frames)

df_plot.columns = ['Region', 'Model', 'Gain in $NSE_m$ (annual-train)',
                     'Gain in $NSE_m$ (annual-test)',
                     'Gain in $NSE_m$ (monthly-train)',
                     'Gain in $NSE_m$ (monthly-test)']
                    #  annualTest', 'Gain_monthTrain', 'Gain_monthTest']
# df_plot = pd.merge(df_outan, df_outmo, on = ['Region', 'Model'],
#                    suffixes=[x_suf, y_suf])

#####
# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rc('text', usetex=True)

fig, axs = plt.subplots(2, 2, figsize = (8, 6), sharex = True)

axsf = axs.flatten()
sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (annual-train)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[0]
)

# plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[0].annotate("(a)", [-0.3, 1.5])
axsf[0].set(xlabel='')
axsf[0].set_yticks([0, 1, 2, 3, 4, 5])
axsf[0].grid(True, axis='y')
# axsf[0].set_title('Annual Training')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (annual-test)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[1]
)

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[1].annotate("(b)", [-0.3, 1.5])
axsf[1].grid(True, axis='y')
axsf[1].legend().set_visible(False)
axsf[1].set_yticks([0, 1, 2, 3, 4, 5])
axsf[1].set(xlabel='')
# axsf[1].set_title('Annual Testing')


sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (monthly-train)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[2]
)

# axsf[2].tick_params(axis = 'x', 
#                     rotation = 45, 
#                     ha = 'right', 
#                     rotation_mode = 'anchor')
axsf[2].set_xticklabels(df_plot['Region'].unique(), 
                        rotation=45, ha='right', 
                        rotation_mode = 'anchor')
axsf[2].annotate("(c)", [-0.3, 1.5])
axsf[2].set(xlabel='Region')
axsf[2].set_yticks([0, 1, 2, 3, 4, 5])
axsf[2].grid(True, axis='y')
axsf[2].legend().set_visible(False)
# axsf[2].set_title('Monthly Training')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (monthly-test)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[3]
)

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[3].annotate("(d)", [-0.3, 1.5])
axsf[3].set(xlabel='Region')
axsf[3].grid(True, axis='y')
axsf[3].legend().set_visible(False)
axsf[3].set_yticks([0, 1, 2, 3, 4, 5])
# axsf[3].set_title('Monthly Testing')



part_name = '_'.join(part_in)

plt.savefig(
    # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    dpi = 300, bbox_inches = 'tight'
)