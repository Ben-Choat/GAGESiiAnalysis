############################
# %%
# define function to calc gains by catchment
def getDiffByCatch(df_sum, regions_column, ID_column, metric_in, normalized=False):
    '''
    Take input dataframe and column name with regions in it and
    calculate gain in NSEm when using xgboost compared to other
    models. e.g., XGBoost - MLR

    df_sum (df): requires 4 columns. One with cluster methods, one with models, 
        and one with
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
    clust_meths = []
    gains = []
    # hru_ids = []
    # regr = []
    # pcamlr = []
    # mlr = [] 




    for region in df_sum[regions_column].unique():
        print(f'Processing region: {region}')

        # regions.extend([region])

        for clust_str in ['Class', 'None']:
            print(f'Processing clust method: {clust_str}')

            df_in = df_sum[df_sum[regions_column] == region]
            
            df1 = df_in.query("clust_method == 'AggEcoregion'").sort_values(by = [ID_column, regions_column])
            df2 = df_in.query("clust_method == @clust_str").sort_values(by = [ID_column, regions_column])
            if normalized:
                df1[metric_in] = 1/(2-df1[metric_in])
                df2[metric_in] = 1/(2-df2[metric_in])
            gain = df1[metric_in].values - df2[metric_in].values
            gain = gain.round(2)
            # print(region)
            gains.extend(gain)
            regions.extend(np.repeat(region, df1.shape[0]))
            clust_meths.extend(np.repeat(clust_str, df1.shape[0]))
            staids.extend(df1[ID_column])

    df_out = pd.DataFrame({
        'STAID': staids,
        'Region': regions,
        'clust_method': clust_meths,
        'Gain': gains
        
    }) # .sort_values(by = 'SLR')

    # df_out = pd.melt(
    #     df_out, id_vars = [regions_column], var_name = 'Model',
    #     value_vars = ['SLR', 'PCA_MLR', 'MLR'],
    #     value_name = 'Gain' # r'Gain in $NSE_m$'
    #     )

    # df_out = df_out.rename(columns={'region': 'Region'})

    return df_out



# %%

if __name__ == '__main__':

    import pandas as pd
    import matplotlib.pyplot as plt

        
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
    model_in = 'XGBoost'

    # getDiffByCatch(df_inma, 'region', 'STAID', 'residuals')

    # regionanlization schemes to consider
    rgns_schemes = ['AggEcoregion', 'Class', 'None']

    # subset working dataframes to regionalization schemes of interest
    df_in = df_in.query("clust_method in @rgns_schemes & model == @model_in")
    df_inma = df_inma.query("clust_method in @rgns_schemes & model == @model_in")


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
    # df_ma.columns = df_ma.columns.str.replace('level_0', 'Region')


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

    # calculate summary dataframes
    # first get absoluate value of residuals
    df_ma['res_abs'] = df_ma.residuals.abs()
    
    # mean annual training
    df_tempIn = df_ma.query("train_val == 'train'")
    summary_matrain = getDiffByCatch(df_tempIn, 'region', 'STAID', 'res_abs')
    # mean annual testing
    df_tempIn = df_ma.query("train_val == 'valnit'")
    summary_matest = getDiffByCatch(df_tempIn, 'region', 'STAID', 'res_abs')

    # annual training
    df_tempIn = df_moan.query("time_scale == 'annual' & train_val == 'train'")
    summary_antrain = getDiffByCatch(df_tempIn, 'region', 'STAID', 'NSE', normalized = True)
    # annual testing
    df_tempIn = df_moan.query("time_scale == 'annual' & train_val == 'valnit'")
    summary_antest = getDiffByCatch(df_tempIn, 'region', 'STAID', 'NSE', normalized = True)
    
    
    # annual training
    df_tempIn = df_moan.query("time_scale == 'monthly' & train_val == 'train'")
    summary_motrain = getDiffByCatch(df_tempIn, 'region', 'STAID', 'NSE', normalized = True)
    # annual testing
    df_tempIn = df_moan.query("time_scale == 'monthly' & train_val == 'valnit'")
    summary_motest = getDiffByCatch(df_tempIn, 'region', 'STAID', 'NSE', normalized = True)


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

    ylabs_in = ['Mean Annual\nGains from Regionlization\n(cm)',
                        '',
                        'Annual\nGains from Regionlization\n(NNSE)',
                        '',
                        'Monthly\nGains from Regionlization\n(NNSE)',
                        '']

    annots = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    

    # %%

    # barplots showning only difference of median NSEs
    import matplotlib.pyplot as plt
    import seaborn as sns

    # define ecoregion names
    ecos_in = ['CntlPlains', 'EastHghlnds', 
                'MxWdShld', 'NorthEast',
                'SECstPlain', 'SEPlains',
                'WestMnts', 'WestPlains',
                'WestXeric']

    # for rgn in ecos_in:
    #     staid_temp = df_in.loc[df_in['region'] == rgn, 'STAID']
    #     df_work = df_in.query("STAID in @staid_temp")
    #     df_workma = df_inma.query("STAID in @staid_temp")

    metric_in = 'NSE'
    # sns.boxplot(df_workma,)
    fig, axs = plt.subplots(figsize=(8, 8), nrows=3, ncols=2, sharex=True)
    for i, ax in enumerate(axs.flatten()):

        if i == 4:
            df_plot = summary_motrain
        elif i == 5:
            df_plot = summary_motest
        elif i == 2:
            df_plot = summary_antrain
        elif i == 3:
            df_plot = summary_antest
        elif i == 0:
            df_plot = summary_matrain
        elif i == 1:
            df_plot = summary_matest    

        partition = 'train' if i in [0, 2, 4] else 'valnit'
        if i in [4, 5]:
            scale = 'monthly'
            ylim_in = [-0.25, 0.26]
            met_in = metric_in
            ytick_in = [-0.25, 0, 0.25]
            annot_loc = [0.2, -0.23]
            # df_plot = df_moan.copy()
        elif i in [2, 3]:
            scale = 'annual'
            ylim_in = [-0.5, 1]
            met_in = metric_in
            ytick_in = [-1, -0.5, 0, 0.5, 1]
            annot_loc = [0.2, -0.9]
            # df_plot = df_moan.copy()
        else:
            scale = 'mean_annual'
            ylim_in = [-15, 15]
            # ylim = [-20, 20]
            # met_in = 'residuals'
            met_in = 'res_abs'
            ytick_in = [-15, -10, -5, 0, 5, 10, 15]
            # df_plot = df_ma.copy()
            # df_plot['abs_residuals'] = df_plot['residuals'].abs()
            # ylim_in = (-55, 25)
            annot_loc = [0.2, -14]
            
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
                x = 'Region', 
                y = 'Gain',
                hue = 'clust_method',
                # legend = leg_in,
                showfliers = False,
                ax=ax)
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

        if i in [1, 3, 5]:
            ax.set_yticklabels([])

        if i in [4, 5]:
            ax.set(xlabel = "Ecoregion")
        

    fig.subplots_adjust(hspace=0.09, wspace=0.1) 

    # plt.savefig(
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromRegionalizationVsModel_Boxplot_trainAndtest_AllScales.png',
    #     dpi = 300, bbox_inches = 'tight'
    # )


# %% plot gains for all regions together 
   
    


    # define ecoregion names
    ecos_in = ['CntlPlains', 'EastHghlnds', 
                'MxWdShld', 'NorthEast',
                'SECstPlain', 'SEPlains',
                'WestMnts', 'WestPlains',
                'WestXeric']
    
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
    
    


    # for rgn in ecos_in:
    #     staid_temp = df_in.loc[df_in['region'] == rgn, 'STAID']
    #     df_work = df_in.query("STAID in @staid_temp")
    #     df_workma = df_inma.query("STAID in @staid_temp")

    
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
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_{part_name}.png',
    # #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_trainAndtest.png',
    #     # f'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/GainsFromXGBoostVsModel_Boxplot_trainAndtest_AllScales.png',
    #     'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Figures/GainsFromRegionalizationVsModelClumped_Boxplot_trainAndtest_AllScales.png',
    #     dpi = 300, bbox_inches = 'tight'
    # )


# %%
