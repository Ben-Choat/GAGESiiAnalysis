'''
BChoat 2023/05/05

Notes
'''


'''
BChoat 2022/10/05
Script to plot timeseries results
'''


# %% 
# Import Libraries
###########

import pandas as pd
import numpy as np
from Load_Data import load_data_fun
import glob
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Lasso
import xgboost as xgb
# import matplotlib.pyplot as plt
# import plotnine as p9
# import seaborn as sns
from Regression_PerformanceMetrics_Functs import *
from NSE_KGE_timeseries import NSE_KGE_Apply
# from statsmodels.distributions.empirical_distribution import ECDF
# from sklearn.preprocessing import StandardScaler
from GAGESii_Class import *






# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well aA
# naming the combined file
# clust_meth = 'None' # 
clust_meths = ['Class', 'None', 'AggEcoregion'  'CAMELS', 'HLR', 
                'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1',
                'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4']

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
# regions = np.linspace(-1, 26, 28).astype(int)
# regions = np.append(regions, ['CntlPlains', 'EastHghlnds', 'MxWdShld', 'NorthEast', 
#                          'SECstPlain', 'SEPlains', 'WestMnts', 'WestPlains', 
#                          'WestXeric', 'All', 'Ref', 'Non-ref'])
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
# time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly', 'daily'
time_scales = ['annual', 'monthly'] # 'mean_annual', 

# define which model you want to work with
model_works = ['regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost']
# model_works = ['XGBoost', 'strd_mlr'] # , 
# 'regr_precip', 'strd_mlr', 'strd_PCA_mlr', 'XGBoost'
# which data to plot/work with
# train_val = 'valnit' # 'train', 'valnit'
train_vals = ['train', 'valnit']

# specifiy whether or not you want the explanatory vars to be standardized
# strd_in =   False # True #

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# read in id file to get regions associated with clust-meths
df_id = pd.read_csv(
    f'{dir_work}/data_work/GAGESiiVariables/ID_train.csv',
    dtype = {'STAID': 'string'}
)

# # directory where to place outputs
# dir_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/TEST_RESULTS'

# delete df_nse_kge if already read in, to avoid duplicates
if 'df_nse_kge' in globals():
    del(df_nse_kge)

for time_scale in time_scales:
    for clust_meth in clust_meths:
        if clust_meth == 'None':
            regions = ['All']
        else:
            regions = df_id[clust_meth].unique()

        for model_work in model_works:
            for train_val in train_vals:
                for region in regions:

                

                    # print update
                    print('\n\n\n',\
                          time_scale, clust_meth, region, model_work, train_val,\
                          '\n\n\n')
                    
                    if model_work == 'XGBoost':
                        strd_in = False
                    else:
                        strd_in = True




                    # %%
                    # Load data
                    ###########

                    try:
                        # load train data (explanatory, water yield, ID)
                        df_trainexpl, df_trainWY, df_trainID = load_data_fun(
                            dir_work = dir_work, 
                            time_scale = time_scale,
                            train_val = 'train',
                            clust_meth = clust_meth,
                            region = region,
                            standardize = strd_in # whether or not to standardize data
                        )
                    except:
                        f'{time_scale}-{clust_meth}-{region}: Not available, continue'
                        continue

                    if df_trainexpl.shape[0] == 0:
                        continue

                    # load valnit data (explanatory, water yield, ID)
                    df_valnitexpl, df_valnitWY, df_valnitID = load_data_fun(
                        dir_work = dir_work, 
                        time_scale = time_scale,
                        train_val = 'valnit',
                        clust_meth = clust_meth,
                        region = region,
                        standardize = strd_in # whether or not to standardize data
                    )

                    # read in columns that were previously removed due to high VIF
                    file = glob.glob(
                        f'{dir_work}/data_out/{time_scale}/VIF_Removed/*{clust_meth}_{region}.csv'
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

                    # drop columns that were removed due to high VIF
                    df_trainexpl.drop(vif_removed, axis = 1, inplace = True)
                    # df_testinexpl.drop(vif_removed, axis = 1, inplace = True)
                    df_valnitexpl.drop(vif_removed, axis = 1, inplace = True)


                    # store staid's and date/year/month
                    # define columns to keep if present in STAID
                    col_keep = ['STAID', 'year', 'month', 'day', 'date']
                    STAIDtrain = df_trainexpl[df_trainexpl.columns.intersection(col_keep)]
                    # STAIDtestin = df_testinexpl[df_testinexpl.columns.intersection(col_keep)]
                    STAIDvalnit = df_valnitexpl[df_valnitexpl.columns.intersection(col_keep)] 

                    # remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
                    # subset WY to version desired (ft)

                    if(time_scale == 'mean_annual'):
                        
                        df_trainexpl.drop('STAID', axis = 1, inplace = True)
                        df_trainWY = df_trainWY['Ann_WY_cm']
                        # df_testinexpl.drop('STAID', axis = 1, inplace = True)
                        # df_testinWY = df_testinWY['Ann_WY_cm']
                        df_valnitexpl.drop('STAID', axis = 1, inplace = True)
                        df_valnitWY = df_valnitWY['Ann_WY_cm']

                    if(time_scale == 'annual'):

                        df_trainexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
                        df_trainWY = df_trainWY['Ann_WY_cm']
                        # df_testinexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
                        # df_testinWY = df_testinWY['Ann_WY_cm']
                        df_valnitexpl.drop(['STAID', 'year'], axis = 1, inplace = True)
                        df_valnitWY = df_valnitWY['Ann_WY_cm']

                    if(time_scale == 'monthly'):

                        df_trainexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)
                        df_trainWY = df_trainWY['Mnth_WY_cm']
                        # df_testinexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace = True)
                        # df_testinWY = df_testinWY['Mnth_WY_cm']
                        df_valnitexpl.drop(['STAID', 'year', 'month'], axis = 1, inplace = True)
                        df_valnitWY = df_valnitWY['Mnth_WY_cm']



                    # %%
                    # read in names of VIF_df csvs to see what models were used
                    ############
                    # models_list = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_*')
                    # print(models_list)

                    # read in results for the time_scale being worked with
                    results_summ = pd.read_pickle(
                        f'{dir_work}/data_out/{time_scale}/combined/All_SummaryResults_{time_scale}.pkl'
                    )
                    # subset to clust_meth and region of interest
                    results_summ = results_summ[
                        (results_summ['clust_method'] == clust_meth) & (results_summ['region'] == region)
                        ]

                    # repeat for individual catchments
                    results_ind = pd.read_pickle(
                        f'{dir_work}/data_out/{time_scale}/combined/All_IndResults_{time_scale}.pkl'
                    )
                    # subset to clust_meth and region of interest
                    results_ind = results_ind[
                        (results_ind['clust_method'] == clust_meth) & (results_ind['region'] == region)
                        ]

                    # results_summ

                    # results_ind


                    # %%
                    # precipitation
                    ###########################################################

                    if model_work == 'regr_precip':
                        expl_in = np.array(df_trainexpl['prcp']).reshape(-1,1)
                        resp_in = df_trainWY
            
                        model = LinearRegression().fit(expl_in, resp_in)
            
                        # write model coef and int to csv
                        temp = pd.DataFrame({
                            'features': ['prcp', 'intercept'],
                            'coef': [model.coef_, model.intercept_]
                            })
        
            
                        # y_predicted = model.predict(expl_in)



                    # %%
                    #######
                    # MLR
                    if model_work == 'strd_mlr':
                        file = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_strd_mlr*.csv')[0]
                        # get variables appearing in final model
                        vars_keep = pd.read_csv(
                            file
                        )['feature']

                        # DRAIN_SQKM stored as DRAIN_SQKM_x in some output by accident,
                        # so replace it
                        vars_keep = vars_keep.str.replace('SQKM_x', 'SQKM')

                        # subset data to variables used in model
                        X_in = df_trainexpl[vars_keep]


                        # define model and parameters
                        reg = LinearRegression()

                        # apply linear regression using all explanatory variables
                        # this object is used in feature selection just below and
                        # results from this will be used to calculate Mallows' Cp
                        model = reg.fit(X_in, df_trainWY)

                        

                        # create dataframe with regression coefficients and variables
                        vars_temp = vars_keep.append(pd.Series('intercept'))
                        coef_temp = np.append(model.coef_, model.intercept_)
                        df_coef = pd.DataFrame({
                            'features': vars_temp,
                            'coef': coef_temp
                        })



                    # %%
                    #######
                    # strd-PCA-MLR
                    if model_work == 'strd_PCA_mlr':
                        # raise(ValueError, 'PCA_strd_mlr is not programmed proper yet.')
                        file = glob.glob(f'{dir_work}/data_out/{time_scale}/VIF_dfs/{clust_meth}_{region}_strd_PCA_mlr*.csv')[0]
                        
                        from GAGESii_Class import Clusterer
                        # from GAGESii_Class import Regressor
                        
                        # get variables appearing in final model
                        vars_keep = pd.read_csv(
                            file
                        )['feature']

                        temp_dupl = vars_keep[vars_keep.duplicated()]

                        vars_keep = vars_keep.drop_duplicates()
                       
                        print('just before wirting duplicates')

                        # log duplicated column if exist
                        if len(temp_dupl) > 0:
                            with open('D:/Projects/GAGESii_ANNstuff/Data_Out/'\
                                    'Results/PCA_Duplicated.txt', 'a+') as file:
                                file.write(f'\n{time_scale}, {clust_meth},'\
                                        f' {region}, {model_work}, {train_val}'\
                                            f'\n\tduplicated: {temp_dupl.values}\n\n')
                                file.close()

                    
                        # create list of columns to drop
                        cols_drop = df_trainexpl.columns[df_trainexpl.columns.isin(col_keep)]

                        print('just defined cols_drop')
                        
                        # define cluster/reducer object
                        clust = Clusterer(clust_vars = df_trainexpl.drop(columns = cols_drop),# ['STAID', 'year', 'month']),
                            id_vars = STAIDtrain['STAID'])
                        
                        # perform PCA on training data and plot
                        clust.pca_reducer(
                            nc = None, # None option includes all components
                            color_in = STAIDtrain['STAID'], # 'blue'
                            plot_out = False # plot_out
                        )
                
                        
                        # regression on transformed explanatory variables
                
                        # subset to clusters of interest
                        expl_vars_in = clust.df_pca_embedding_[vars_keep]
                        
                        # subset data to variables used in model
                        X_in = expl_vars_in[vars_keep]


                        # define model and parameters
                        reg = LinearRegression()

                        model = reg.fit(
                            # df_train_mnexpl[features_in], df_train_mnanWY
                            X_in, df_trainWY
                            )
                    

                        # create dataframe with regression coefficients and variables
                        vars_temp = vars_keep.append(pd.Series('intercept'))
                        coef_temp = np.append(model.coef_, model.intercept_)
                        df_coef = pd.DataFrame({
                            'features': vars_temp,
                            'coef': coef_temp
                        })


                

                

                    # %%
                    # XGBOOST
                    ##########
                    if model_work == 'XGBoost':
                        # first define xgbreg object
                        model = xgb.XGBRegressor()

                        # define temp time_scale var since model names do not have '_' in them
                        temp_time = time_scale.replace('_', '')

                        # reload model into object
                        # try:
                        model.load_model(
                            f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
                            f'/Models/XGBoost_{temp_time}_{clust_meth}_{region}_model.json'
                            # f'/Models/xgbreg_meanannual_XGBoost_{clust_meth}_{region}_model.json'
                            )
                        # except:
                        #     model.load_model(
                        #         f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
                        #         # f'/Models/XGBoost_{temp_time}_{clust_meth}_{region}_model.json'
                        #         f'/Models/xgbreg_{temp_time}_XGBoost_{clust_meth}_{region}_model.json'
                        #         )

                        X_in = df_trainexpl



                    # %% 
                    # Plot
                    #############
                
                    if model_work == 'regr_precip':
                        if train_val == 'train':
                            X_in = np.array(df_trainexpl['prcp']).reshape(-1,1)
                        else:
                            X_in = np.array(df_valnitexpl['prcp']).reshape(-1,1)
                        
                    elif model_work == 'strd_PCA_mlr':
                        if train_val == 'train':
                            X_in = X_in
                        else:
                            x_tr = pd.DataFrame(
                                clust.pca_fit_.transform(df_valnitexpl)
                            )
                            x_tr.columns = [
                                f'Comp{i}' for i in np.arange(0, x_tr.shape[1], 1)
                                ]
                            X_in = x_tr[vars_keep]
                            
                    elif model_work == 'strd_mlr':
                        if train_val == 'train':
                            X_in = df_trainexpl[vars_keep] 
                        else:
                            X_in = df_valnitexpl[vars_keep]

                    else: # ADD DATA READ IN FOR XGBOOST HERE
                        if train_val == 'train':
                            X_in = df_trainexpl
                        else:
                            X_in = df_valnitexpl
                    
                    
                    if train_val == 'train':
                        # X_in = X_in
                        STAID_in = STAIDtrain
                        df_WY_in = df_trainWY    

                    if train_val == 'valnit':
                        # X_in = df_valnitexpl[vars_keep] if model_work == 'strd_mlr' else df_valnitexpl
                        STAID_in = STAIDvalnit
                        df_WY_in = df_valnitWY


                    # predict WY
                    y_pred = model.predict(X_in)

                    # define array to use as label of observed or predicted

                    lab_in = np.hstack([
                        np.repeat('Obs', len(y_pred)),
                        np.repeat('Pred', len(y_pred))
                    ])

                    # if time_scale == 'mean_annual':
                    #     # create dataframe for plotting
                    #     plot_data = pd.DataFrame({
                    #         'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
                    #         'year': np.hstack([STAID_in['year'], STAID_in['year']]),
                    #         'WY_cm': np.hstack([df_WY_in, y_pred]),
                    #         'label': lab_in
                    #     })
                    
                    # if time_scale == 'annual':
                    #     # create dataframe for plotting
                    #     plot_data = pd.DataFrame({
                    #         'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
                    #         'year': np.hstack([STAID_in['year'], STAID_in['year']]),
                    #         'WY_cm': np.hstack([df_WY_in, y_pred]),
                    #         'label': lab_in
                    #     })

                    # if time_scale == 'monthly':
                    #     # create dataframe for plotting
                    #     plot_data = pd.DataFrame({
                    #         'STAID': np.hstack([STAID_in['STAID'], STAID_in['STAID']]),
                    #         'year': np.hstack([STAID_in['year'], STAID_in['year']]),
                    #         'month': np.hstack([STAID_in['month'], STAID_in['month']]),
                    #         'WY_cm': np.hstack([df_WY_in, y_pred]),
                    #         'label': lab_in
                    #     })





    # NOW CALC COMPONENTS OF KGE

                    # %%
                    # calc NSE and KGE from all stations
                    df_temp = NSE_KGE_Apply(
                        pd.DataFrame({
                            'y_obs': df_WY_in,
                            'y_pred': y_pred,
                            'ID_in': STAID_in['STAID']
                            }),
                            return_comp = True
                        )

                    df_temp['clust_method'] = np.repeat(clust_meth, df_temp.shape[0])
                    df_temp['region'] = np.repeat(region, df_temp.shape[0])
                    df_temp['model'] = np.repeat(model_work, df_temp.shape[0])
                    df_temp['train_val'] = np.repeat(train_val, df_temp.shape[0])
                    df_temp['time_scale'] = np.repeat(time_scale, df_temp.shape[0])

                    if not 'df_nse_kge' in globals():
                        df_nse_kge = df_temp
                    else:
                        df_nse_kge = pd.concat([
                            df_nse_kge, df_temp
                        ])

                    # print(f'\n\n\n {time_scale}-{clust_meth}-{region}\n\n\n')


print('Printing df_nse_kge to csv')


df_nse_kge.to_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv',
    index = False)
# %%

# plt.hist(df_temp['NSE'], bins = 10000)
# plt.xlim(-1, 1)


# %%
# compare KGE report from summit with vals calculated here
############################

df_summit = pd.read_pickle(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/monthly/combined/All_SummaryResults_monthly.pkl')


df_nse_kge = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/NSEComponents_KGE.csv')


df_nse_kge = df_nse_kge[df_nse_kge['time_scale'] == 'monthly']

df_summit.shape
df_nse_kge.shape
# %%
# # ECDF of NSE and KGE

# # define which metric to use (e.g., NSE, or KGE)
# metr_in = 'NSE'

# cdf = ECDF(df_nse_kge[metr_in])

# # define lower limit for x (NSE)
# xlow = -1
# # remove vals from cdf lower than xlow
# cdf.y = cdf.y[cdf.x > xlow]
# cdf.x = cdf.x[cdf.x > xlow]


# p = (
#     p9.ggplot() +
#     p9.geom_line(p9.aes(
#         x = cdf.x,
#         y = cdf.y)) +
#         p9.xlim([-1, 1]) +
#         p9.ylim([0,1]) +
#         p9.theme_bw() +
#         p9.ggtitle(
#             f'eCDF of {metr_in} in {region}-train from {model_work}'
#         )
    
#     )


# print(p)

# print(f'median NSE: {np.median(df_nse_kge[metr_in])}')