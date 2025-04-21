############################

# %% plots of gains compared to xgboost
################################


import pandas as pd
from functools import reduce

# define function to calc gains by region
def getDiff(df_sum, regions_column, metric_in):
    '''
    Take input dataframe and column name with regions in it and
    calculate gain in NSEm when using xgboost compared to other
    models. e.g., XGBoost - MLR

    df_sum (df): requires 3 columns. One with regions, one with models, and one with
        a metric_in (e.g., NSE) column with a single valuefor each region-model combo

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



# define function to calc gains by catchment
def getDiffByCatch(df_sum, regions_column, ID_column, metric_in, normalized=False):
    '''
    Take input dataframe and column name with regions in it and
    calculate gain in NSEm when using xgboost compared to other
    models. e.g., XGBoost - MLR

    df_sum (df): requires 4 columns. One with regions, one with models, and one with
        a metric_in (e.g., NSE) column with a single value for each catchment-model combo
        a ID_column with unique identifiers for each catchment

    metric_in str: 'NSE', 'KGE', 'Residuals'

    normalized (boolean): if True, normalizes using 2/(1-metric_in).
        Meant to be used with NNSE.
    
    Output
    ________________
    dataframe: 
        col1: staids
        col2: region
        col3: model
        col4: Diff (gain)


    '''
    # regions_column: column holding regions

    import numpy as np
    staids = []
    regions = []
    models = []
    gains = []
    # hru_ids = []
    # regr = []
    # pcamlr = []
    # mlr = [] 




    for region in df_sum[regions_column].unique():

        # regions.extend([region])

        for model_str in ['regr_precip', 'strd_PCA_mlr', 'strd_mlr']:

            df_in = df_sum[df_sum[regions_column] == region]
            
            df1 = df_in.query("model == 'XGBoost'").sort_values(by = [ID_column, regions_column])
            df2 = df_in.query("model == @model_str").sort_values(by = [ID_column, regions_column])
            if normalized:
                df1[metric_in] = 1/(2-df1[metric_in])
                df2[metric_in] = 1/(2-df2[metric_in])
            gain = df1[metric_in].values - df2[metric_in].values
            gain = gain.round(2)
            # print(region)
            gains.extend(gain)
            regions.extend(np.repeat(region, df1.shape[0]))
            models.extend(np.repeat(model_str, df1.shape[0]))
            staids.extend(df1[ID_column])

    df_out = pd.DataFrame({
        'STAID': staids,
        'Region': regions,
        'model': models,
        'Gain': gains
        
    }) # .sort_values(by = 'SLR')

    # df_out = pd.melt(
    #     df_out, id_vars = [regions_column], var_name = 'Model',
    #     value_vars = ['SLR', 'PCA_MLR', 'MLR'],
    #     value_name = 'Gain' # r'Gain in $NSE_m$'
    #     )

    # df_out = df_out.rename(columns={'region': 'Region'})

    return df_out



if __name__ == '__main__':
        
    # define working directory
    dir_res = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'
    # df_in = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv',
    df_in = pd.read_csv(f'{dir_res}/NSEComponents_KGE.csv',
                    dtype={'STAID': 'string'})
    # df_inma = pd.read_csv('D:/Projects/GAGESii_ANNstuff/Data_Out/Results/PerfMetrics_MeanAnnual.csv',
    df_inma = pd.read_csv(f'{dir_res}/PerfMetrics_MeanAnnual.csv',
                    dtype={'STAID': 'string'})

    # subset df_in by timescale
    # df = df_in.query("clust_method in ['AggEcoregion', 'None', 'Class'] and train_val == 'valnit'")
    # df_ann = df.query('time_scale == "annual"') #  & model == "XGBoost"')
    # df_month = df.query('time_scale == "monthly" & model == "XGBoost"')
    # # handle mean annual
    # df_ma = df_inma.query("clust_method in ['AggEcoregion', 'None', 'Class'] "\
    #                     "and train_val == 'valnit'")

    # df_ann = df_ann.groupby(['clust_method', 'region', 'model', 'time_scale']).quantile(
    #     # [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    #     0.5
    # ).reset_index().sort_values(by = "NSE")

    # df_month = df_month.groupby(['clust_method', 'region', 'model', 'time_scale']).quantile(
    #     # [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    #     0.5
    # ).reset_index().sort_values(by = "NSE")





    # df_in.head()



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
    df_ma = df_inma.query("clust_method in ['AggEcoregion', 'None', 'Class']") #\ 
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

    # catchment-wise
    STAID_Gains = getDiffByCatch(df_annTrain, 'region', 'STAID', 'NSE', normalized=True)
    # testing
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(STAID_Gains, y = 'Gain', x = 'Region', hue = 'model')
    # plt.grid()
    # plt.yscale('log')


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

    # barplots showning only difference of median NSEs
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Enable LaTeX rendering
    # plt.rcParams['text.usetex'] = True
    # plt.rc('text', usetex=True)
    xaxis_order = ['CntlPlains', 'EastHghlnds', 
                'MxWdShld', 'NorthEast',
                'SECstPlain', 'SEPlains',
                'WestMnts', 'WestPlains',
                'WestXeric', 'Non-ref',
                'Ref', 'All']
    fig, axs = plt.subplots(3, 2, figsize = (8, 8), sharex = True)

    axsf = axs.flatten()
    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $Residuals_m$ (mean annual-train)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[0],
        # order = xaxis_order
    )

    # plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    axsf[0].annotate("(a)", [-0.3, 1.5])
    axsf[0].set(xlabel='', ylabel='Mean Annual\nGains from XGBoost\n(cm)',
                title = 'Training')
    axsf[0].set_yticks([-3, -2, -1, 0, 1, 2])
    axsf[0].grid(True, axis='both')
    axsf[0].legend().set_visible(False)

    # axsf[0].set_title('Annual Training')

    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $Residuals_m$ (mean annual-test)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[1],
        # order = xaxis_order
    )

    plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    axsf[1].annotate("(b)", [-0.3, 1.5])
    axsf[1].grid(True, axis='both')
    axsf[1].legend().set_visible(False)
    axsf[1].set_yticks([-3, -2, -1, 0, 1, 2])
    axsf[1].set(xlabel='', ylabel='',
                title = 'Testing')

    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $NSE_m$ (annual-train)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[2],
        # order = xaxis_order
    )

    # plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    axsf[2].annotate("(c)", [-0.3, 1.5])
    axsf[2].set(xlabel='', ylabel='Annual\nGains from XGBoost\n(NSE)')
    axsf[2].set_yticks([0, 1, 2, 3, 4, 5])
    axsf[2].grid(True, axis='both')
    axsf[2].legend(loc='best')
    # axsf[0].set_title('Annual Training')

    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $NSE_m$ (annual-test)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[3],
        # order = xaxis_order
    )

    plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    axsf[3].annotate("(d)", [-0.3, 1.5])
    axsf[3].grid(True, axis='both')
    axsf[3].legend().set_visible(False)
    axsf[3].set_yticks([0, 1, 2, 3, 4, 5])
    axsf[3].set(xlabel='', ylabel='')
    # axsf[1].set_title('Annual Testing')


    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $NSE_m$ (monthly-train)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[4],
        # order = xaxis_order
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
    axsf[4].grid(True, axis='both')
    axsf[4].legend().set_visible(False)
    # axsf[2].set_title('Monthly Training')

    sns.barplot(
        data = df_plot,
        x = 'Region',
        y = f'Gain in $NSE_m$ (monthly-test)', # r'Gain in $NSE_m$',
        hue = 'Model',
        ax = axsf[5],
        # order = xaxis_order
    )

    plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
    axsf[5].annotate("(f)", [-0.3, 1.5])
    axsf[5].set(xlabel='Region', ylabel='')
    axsf[5].grid(True, axis='both')
    axsf[5].legend().set_visible(False)
    axsf[5].set_yticks([0, 1, 2, 3, 4, 5])
    # axsf[3].set_title('Monthly Testing')



    part_name = '_'.join(part_in)

    # plt.savefig(
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    #     f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest_AllScales.png',
    #     dpi = 300, bbox_inches = 'tight'
    # )



     # %%

    metric_in = 'NSE'

      # get gains
    df_mannTrain['residuals'] = df_mannTrain['residuals'].abs()
    # catchment-wise
    df_outmanTrain = getDiffByCatch(df_mannTrain, 'region', 'STAID', 'residuals')
    df_outanTrain = getDiffByCatch(df_annTrain, 'region', 'STAID', metric_in, normalized=True)
    df_outmonthTrain = getDiffByCatch(df_monthTrain, 'region', 'STAID', metric_in, normalized=True)
    # testing
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.boxplot(df_outmanTrain, y = 'Gain', x = 'Region', hue = 'model', showfliers=False)
    # plt.grid()
    # plt.yscale('log')

    # get gains
    # take absolute value of residuals to get magnitude of difference
    df_mannTest['residuals'] = df_mannTest['residuals'].abs()
    # catchment-wise
    df_outmanTest = getDiffByCatch(df_mannTest, 'region', 'STAID', 'residuals')
    df_outanTest = getDiffByCatch(df_annTest, 'region', 'STAID', metric_in, normalized=True)
    df_outmonthTest = getDiffByCatch(df_monthTest, 'region', 'STAID', metric_in, normalized=True)

    # monthly and annual
    data_frames = [df_outmanTrain, df_outmanTest, 
                df_outanTrain, df_outanTest, 
                df_outmonthTrain, df_outmonthTest]
    for df in data_frames:
        df['model'] = df['model'].replace('regr_precip', 'SLR')
        df['model'] = df['model'].replace('strd_mlr', 'MLR')
        df['model'] = df['model'].replace('strd_PCA_mlr', 'PCA_MLR')

    # df_plot = reduce(lambda  left,right: pd.merge(left, right, 
    #                                             on = ['STAID', 'Region', 'model']), 
    #                                             data_frames)


    # BEN FIX THIS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # columns_in = ['STAID', 'Region', 'Model',  
    ylabs_in = ['Gain in $Residuals_m$\n(mean annual-train)',
                        'Gain in $Residuals_m$ (mean annual-test)',
                        'Gain in $NSE_m$ (annual-train)',
                        'Gain in $NSE_m$ (annual-test)',
                        'Gain in $NSE_m$ (monthly-train)',
                        'Gain in $NSE_m$ (monthly-test)']
    
    df_names_in = ['MeanAnnual_Train', 'MeanAnnual_Test',
                   'Annual_Train', 'Annual_Test',
                   'Monthly_Train', 'Monthly_test']
    


    # df_plot = df_plot.sort_values(by = ['Model', 'Gain in $NSE_m$ (monthly-train)'],
    #                             ascending=[False, True])
    
    dfs_plot = {
        df_names_in[i]: data_frames[i] for i in range(len(df_names_in))
    }



     # %%

    # barplots showning only difference of median NSEs
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Enable LaTeX rendering
    # plt.rcParams['text.usetex'] = True
    # plt.rc('text', usetex=True)
    xaxis_order = ['CntlPlains', 'EastHghlnds', 
                'MxWdShld', 'NorthEast',
                'SECstPlain', 'SEPlains',
                'WestMnts', 'WestPlains',
                'WestXeric', 'Non-ref',
                'Ref', 'All']
    annots = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axs = plt.subplots(3, 2, figsize = (8, 8), sharex = True)

    for i, ax in enumerate(axs.flatten()):
        # print(i)
        # print(ax)
        if i in [0, 1]:
            ylim_in = (-55, 25)
            annot_loc = [0.2, -50]
            ytick_in = [-45, -30, -15, 0, 15]
        else:
            ylim_in = (-0.5, 1)
            annot_loc = [0.2, -0.45]
            ytick_in = [-0.5, 0, 0.5, 1]
        # if i in [1, 3, 5]:
        #     ytick_in = []

        if i == 0:
            ylab_in = 'Mean Annual\nGains from XGBoost\n(cm)'
        elif i == 2:
            ylab_in = 'Annual\nGains from XGBoost\n(NNSE)'
        elif i == 4:
            ylab_in = 'Monthly\nGains From XGBoost\n(NNSE)'
        else:
            ylab_in = ''
        
        if i == 0:
            title_in = 'Training'
        elif i == 1:
            title_in = 'Testing'
        else:
            ax.title.set_visible(False) # set on or off

        sns.boxplot(
            data = dfs_plot[list(dfs_plot.keys())[i]],
            x = 'Region',
            y = f'Gain', # r'Gain in $NSE_m$',
            hue = 'model',
            ax = ax, # axsf[0],
            showfliers = False,
            zorder = 2
            # order = xaxis_order
        )

        ax.set(ylabel = ylabs_in[i], ylim=ylim_in)

        # plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
        ax.annotate(annots[i], annot_loc)
        ax.set(xlabel='', ylabel=ylab_in,
                    title = title_in)
        ax.set_yticks(ytick_in)
        # ax.set_yticklabels(ylabs_in)
        ax.grid(True, axis='both')
        ax.axhline(0, ls='-', linewidth=0.9, color='red', zorder=3)
        ax.legend().set_visible(False)

        if i == (len(axs.flatten())-1):
            # add legend 
            # ax.legend(loc='lower center', ncol = 3)
            # Add legend outside of the subplots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor = [0.5, -0.02])

            # Adjust the layout to make space for the legend
            plt.subplots_adjust(bottom=0.15)

        if i in [4, 5]:
            plt.sca(ax)  # Set current axis to ax
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
            ax.set(xlabel = "Region")
        
        if i in [1, 3, 5]:
            ax.set_yticklabels([])
            
        

    fig.subplots_adjust(hspace=0.09, wspace=0.1) 

    part_name = '_'.join(part_in)

    # plt.savefig(
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     dpi = 300, bbox_inches = 'tight'
    # )














    # Enable LaTeX rendering
    # plt.rcParams['text.usetex'] = True
    # plt.rc('text', usetex=True)
    xaxis_order = ['CntlPlains', 'EastHghlnds', 
                'MxWdShld', 'NorthEast',
                'SECstPlain', 'SEPlains',
                'WestMnts', 'WestPlains',
                'WestXeric', 'Non-ref',
                'Ref', 'All']
    annots = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    fig, axs = plt.subplots(3, 2, figsize = (8, 8), sharex = True)

    for i, ax in enumerate(axs.flatten()):
        # print(i)
        # print(ax)
        if i in [0, 1]:
            ylim_in = (-55, 25)
            annot_loc = [0.2, -50]
            ytick_in = [-45, -30, -15, 0, 15]
        else:
            ylim_in = (-0.5, 1)
            annot_loc = [0.2, -0.45]
            ytick_in = [-0.5, 0, 0.5, 1]
        # if i in [1, 3, 5]:
        #     ytick_in = []

        if i == 0:
            ylab_in = 'Mean Annual\nGains from XGBoost\n(cm)'
        elif i == 2:
            ylab_in = 'Annual\nGains from XGBoost\n(NNSE)'
        elif i == 4:
            ylab_in = 'Monthly\nGains From XGBoost\n(NNSE)'
        else:
            ylab_in = ''
        
        if i == 0:
            title_in = 'Training'
        elif i == 1:
            title_in = 'Testing'
        else:
            ax.title.set_visible(False) # set on or off

        sns.boxplot(
            data = dfs_plot[list(dfs_plot.keys())[i]],
            x = 'Region',
            y = f'Gain', # r'Gain in $NSE_m$',
            hue = 'model',
            ax = ax, # axsf[0],
            showfliers = False,
            zorder = 2
            # order = xaxis_order
        )

        ax.set(ylabel = ylabs_in[i], ylim=ylim_in)

        # plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
        ax.annotate(annots[i], annot_loc)
        ax.set(xlabel='', ylabel=ylab_in,
                    title = title_in)
        ax.set_yticks(ytick_in)
        # ax.set_yticklabels(ylabs_in)
        ax.grid(True, axis='both')
        ax.axhline(0, ls='-', linewidth=0.9, color='red', zorder=3)
        ax.legend().set_visible(False)

        if i == (len(axs.flatten())-1):
            # add legend 
            # ax.legend(loc='lower center', ncol = 3)
            # Add legend outside of the subplots
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor = [0.5, -0.02])

            # Adjust the layout to make space for the legend
            plt.subplots_adjust(bottom=0.15)

        if i in [4, 5]:
            plt.sca(ax)  # Set current axis to ax
            plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
            ax.set(xlabel = "Region")
        
        if i in [1, 3, 5]:
            ax.set_yticklabels([])
            
        

    fig.subplots_adjust(hspace=0.09, wspace=0.1) 

    part_name = '_'.join(part_in)

    # plt.savefig(
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     dpi = 300, bbox_inches = 'tight'
    # )