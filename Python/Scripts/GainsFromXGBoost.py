############################

# %% plots of gains compared to xgboost
################################


import pandas as pd
from functools import reduce

# define function to calc gains

def getDiff(df_sum, regions_column, metric_in):
    '''
    Take input dataframe and column name with regions in it and
    calculate gain in NSEm when using xgboost compared to other
    models. e.g., XGBoost - MLR

    metric_in str: 'NSE', 'KGE', 'Residuals'
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
                    (df_sum['model'] == 'XGBoost'), metric_in].values - \
                        df_sum.loc[(df_sum[regions_column] == region) & \
                                    (df_sum['model'] == model_str), metric_in].values
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
df_inma = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/PerfMetrics_MeanAnnual.csv',
                 dtype={'STAID': 'string'})

# subset df_in by timescale
df = df_in.query("clust_method in ['AggEcoregion', 'None', 'Class'] and train_val == 'valnit'")
df_ann = df.query('time_scale == "annual"') #  & model == "XGBoost"')
df_month = df.query('time_scale == "monthly" & model == "XGBoost"')
# handle mean annual
df_ma = df_inma.query("clust_method in ['AggEcoregion', 'None', 'Class'] "\
                      "and train_val == 'valnit'")

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
df_ma = df_inma.query("clust_method in ['AggEcoregion', 'None', 'Class']")
# " and train_val in @part_in")

# training                 
df_mannTrain = df_ma.query('time_scale == "mean_annual"'\
                  " and train_val in ['train']")
df_annTrain = df.query('time_scale == "annual"'\
                  " and train_val in ['train']")
df_monthTrain = df.query('time_scale == "monthly"'\
                    " and train_val in ['train']")
# quantiles
df_mansumTrain = df_mannTrain.groupby(
    ['region', 'model']
)['residuals'].quantile(0.5).reset_index()
df_ansumTrain = df_annTrain.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()
df_mosumTrain = df_monthTrain.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()


# get gains
df_mansumTrain['residuals'] = df_mansumTrain['residuals'].abs()
df_outmanTrain = getDiff(df_mansumTrain, 'region', 'residuals')
df_outanTrain = getDiff(df_ansumTrain, 'region', 'NSE')
df_outmoTrain = getDiff(df_mosumTrain, 'region', 'NSE')


# testing         
df_mannTest = df_ma.query('time_scale == "mean_annual"'\
                  " and train_val in ['valnit']")      
df_annTest = df.query('time_scale == "annual"'\
                  " and train_val in ['valnit']")
df_monthTest = df.query('time_scale == "monthly"'\
                    " and train_val in ['valnit']")
# quaniles
df_mansumTest = df_mannTest.groupby(
    ['region', 'model']
)['residuals'].quantile(0.5).reset_index()
df_ansumTest = df_annTest.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()
df_mosumTest = df_monthTest.groupby(
    ['region', 'model']
)['NSE'].quantile(0.5).reset_index()

# get gains
# take absolute value of residuals to get magnitude of difference
df_mansumTest['residuals'] = df_mansumTest['residuals'].abs()
df_outmanTest = getDiff(df_mansumTest, 'region', 'residuals')
df_outanTest = getDiff(df_ansumTest, 'region', 'NSE')
df_outmoTest = getDiff(df_mosumTest, 'region', 'NSE')


# x_suf = ' in $NSE_m$ (annual)'
# y_suf = ' in $NSE_m$ (monthly)'

# monthly and annual
data_frames = [df_outmanTrain, df_outmanTest, 
               df_outanTrain, df_outanTest, 
               df_outmoTrain, df_outmoTest]

df_plot = reduce(lambda  left,right: pd.merge(left, right, 
                                            on = ['Region', 'Model']), 
                                            data_frames)

df_plot.columns = ['Region', 'Model',  
                     'Gain in $Residuals_m$ (mean annual-train)',
                     'Gain in $Residuals_m$ (mean annual-test)',
                     'Gain in $NSE_m$ (annual-train)',
                     'Gain in $NSE_m$ (annual-test)',
                     'Gain in $NSE_m$ (monthly-train)',
                     'Gain in $NSE_m$ (monthly-test)']

df_plot = df_plot.sort_values(by = ['Model', 'Gain in $NSE_m$ (monthly-train)'],
                              ascending=[False, True])

                    #  annualTest', 'Gain_monthTrain', 'Gain_monthTest']
# df_plot = pd.merge(df_outan, df_outmo, on = ['Region', 'Model'],
#                    suffixes=[x_suf, y_suf])

# mean annual
# data_frames = [df_outmanTrain, df_outmanTest]

# df_plotma = pd.merge(data_frames[0], data_frames[1],
#                      on = ['Region', 'Model'])
# df_plotma.columns = ['Region', 'Model', 
#                      'Gain in $Residuals_m$ (mean annual-train)',
#                      'Gain in $Residuals_m$ (mean annual-test)']

#####
# %%

import matplotlib.pyplot as plt
import seaborn as sns

# Enable LaTeX rendering
# plt.rcParams['text.usetex'] = True
# plt.rc('text', usetex=True)

fig, axs = plt.subplots(3, 2, figsize = (8, 8), sharex = True)

axsf = axs.flatten()
sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $Residuals_m$ (mean annual-train)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[0]
)

# plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[0].annotate("(a)", [-0.3, 1.5])
axsf[0].set(xlabel='', ylabel='Mean Annual\nGains from XGBoost\n(cm)',
            title = 'Training')
axsf[0].set_yticks([-3, -2, -1, 0, 1, 2])
axsf[0].grid(True, axis='y')
axsf[0].legend().set_visible(False)

# axsf[0].set_title('Annual Training')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $Residuals_m$ (mean annual-test)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[1]
)

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[1].annotate("(b)", [-0.3, 1.5])
axsf[1].grid(True, axis='y')
axsf[1].legend().set_visible(False)
axsf[1].set_yticks([-3, -2, -1, 0, 1, 2])
axsf[1].set(xlabel='', ylabel='',
            title = 'Testing')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (annual-train)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[2]
)

# plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[2].annotate("(c)", [-0.3, 1.5])
axsf[2].set(xlabel='', ylabel='Annual\nGains from XGBoost\n(NSE)')
axsf[2].set_yticks([0, 1, 2, 3, 4, 5])
axsf[2].grid(True, axis='y')
axsf[2].legend(loc='best')
# axsf[0].set_title('Annual Training')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (annual-test)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[3]
)

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[3].annotate("(d)", [-0.3, 1.5])
axsf[3].grid(True, axis='y')
axsf[3].legend().set_visible(False)
axsf[3].set_yticks([0, 1, 2, 3, 4, 5])
axsf[3].set(xlabel='', ylabel='')
# axsf[1].set_title('Annual Testing')


sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (monthly-train)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[4]
)

# axsf[2].tick_params(axis = 'x', 
#                     rotation = 45, 
#                     ha = 'right', 
#                     rotation_mode = 'anchor')
axsf[4].set_xticklabels(df_plot['Region'].unique(), 
                        rotation=45, ha='right', 
                        rotation_mode = 'anchor')
axsf[4].annotate("(e)", [-0.3, 1.5])
axsf[4].set(xlabel='Region', ylabel = 'Monthly\nGains From XGBoost\n(NSE)')
axsf[4].set_yticks([0, 1, 2, 3, 4, 5])
axsf[4].grid(True, axis='y')
axsf[4].legend().set_visible(False)
# axsf[2].set_title('Monthly Training')

sns.barplot(
    data = df_plot,
    x = 'Region',
    y = f'Gain in $NSE_m$ (monthly-test)', # r'Gain in $NSE_m$',
    hue = 'Model',
    ax = axsf[5]
)

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
axsf[5].annotate("(f)", [-0.3, 1.5])
axsf[5].set(xlabel='Region', ylabel='')
axsf[5].grid(True, axis='y')
axsf[5].legend().set_visible(False)
axsf[5].set_yticks([0, 1, 2, 3, 4, 5])
# axsf[3].set_title('Monthly Testing')



part_name = '_'.join(part_in)

plt.savefig(
    # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest_AllScales.png',
    dpi = 300, bbox_inches = 'tight'
)