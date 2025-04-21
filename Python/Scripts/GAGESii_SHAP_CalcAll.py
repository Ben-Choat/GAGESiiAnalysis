'''
BChoat 10/17/2022

Script to loop through all models and calculate SHAP values.
After being calculated shap values are appended to a dataframe,
which is eventually written to a csv. 


'''





# %%
# Import libraries
#################

import pandas as pd
import numpy as np
import shap
from Load_Data import load_data_fun
import glob
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import xgboost as xgb




# %% 
# Define list of clustering methods, regions/clusters, and timescales on
# which to work

# clustering methods
clust_meths = ['None', 'Class', 'AggEcoregion']

# clust_meths = ['None', 'Class', 'AggEcoregion', 
#         'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1', 
#         'CAMELS', 'HLR', 'Nat_0', 'Nat_1', 'Nat_2',
#         'Nat_3', 'Nat_4']

# clust_meths = ['Nat_3']

# clust_meth = ['Anth_0'] # ,'Anth_0', 'Nat_0']

# read in ID.csv file to get unique clusters under each method
df_ID = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/' \
        'GAGESiiVariables/ID_train.csv'
)

# time scales
time_scale = ['monthly', 'annual', 'mean_annual']

# partition in (train or valint (aka testing))
# part_in = 'train'
part_in = 'valnit'

# use 'NSE' or 'KGE'? '|residuals| always used for mean_annual
metric_in = 'KGE'

# drop noise? True or False
dropNoise = False

# Define directory variables
# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results' 

# another main location with output from HPC runs.
dir_workHPC = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/'
# # read in any data needed
# dir_in = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results'
# # mean annual
# df_mnan_results = pd.read_csv(
#                 f'{dir_in}/PerfMetrics_MeanAnnual.csv',
#                 dtype = {'STAID': 'string',
#                          'region': 'string'}
#             )

# # df_mnan_results['residuals'] = df_mnan_results.residuals.abs()

# df_mnan_results = df_mnan_results[[
#                 'STAID', 'residuals', 'clust_method', 'region',\
#                             'model', 'time_scale', 'train_val'
#             ]]

# # annual, montly
# df_anm_results = pd.read_csv(
#                 f'{dir_in}/NSEComponents_KGE.csv',
#                 dtype = {'STAID': 'string',
#                          'region': 'string'}
#             )
                  


# directory where to place SHAP outputs
dir_shapout = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'

# directory where to place PCA outputs
dir_pcaout = 'D:/Projects/GAGESii_ANNstuff/Data_Out/PCA_OUT'

# initialize dataframe to hold pca info
df_pca95 = pd.DataFrame(
    columns = ['ClusterMethod', 'Region', 'PCA95']
)

# name list of column names to drop when defining output dataframe of shap values
names_drop = ['STAID', 'year', 'month', 'day', 'date']


# %% define function to compute mean quantile metrics
#############################

def q_metr(df_work, perf_metric):
    '''
    inputs
    ---------------
    df_work: pandas dataframe with at least columns
        ['clust_method', 'model', 'train_val', 'time_scale']
    perf_metric: str, 'KGE', 'NSE', or '|residuals|'
    
    returns
    -----------------
    df_qntls: a dataframe with average quantile performance metrics by clust-method and region
    '''

    # define quantiles to use in average
    qnts_in = np.round(np.arange(0.05, 1.0, 0.05), 2)
    
    df_qntls = None
    
    # df_work = df_in.copy()
    # if drop_noise:
    #     df_work = df_work[df_work['region'] != '-1']

    for q in qnts_in:
        print(f'q: {q}')
        if df_qntls is None:
            # if first q, then create new dataframe
            df_qntls = df_work.groupby(
                    ['clust_method', 'model', 'train_val', 'time_scale']
                    )[perf_metric].quantile(q).reset_index()
            df_qntls = df_qntls.rename(columns = {f'{perf_metric}': f'{perf_metric}_q{q}'})
           
        else:
            # otherwise, update just created dataframe
            df_qntls[f'{perf_metric}_q{q}'] = df_work.groupby(
                    ['clust_method', 'model', 'train_val', 'time_scale']
                    )[perf_metric].quantile(q).reset_index()[perf_metric]
            
            df_qntlmean = df_qntls[['clust_method', 'model', 'train_val', 'time_scale']]

    meanqntls = df_qntls.drop(['clust_method', 'model', 'train_val', 'time_scale'], axis = 1)
    meanqntls = meanqntls.apply('mean', axis = 1)
    df_qntlmean[f'{perf_metric}_qmean'] = meanqntls
            
    return(df_qntlmean)


# %%
# Load data
###########

# define dict to hold cluster methods: cluster
df_regions = {}

# define regions for different clustering methods
for cl in clust_meths:
    if cl == 'None':
        df_regions[cl] = 'All'
    else:
        df_regions[cl] = np.sort(df_ID[cl].unique())


for timescale in time_scale:

    # load results file to get best model
    # read in results for the time_scale being worked with
    # results_summAll = pd.read_pickle(
    #     f'{dir_work}/data_out/{timescale}/combined/All_SummaryResults_{timescale}.pkl'
    # )

    if timescale == 'mean_annual':
        results_summAll = pd.read_csv(
                f'{dir_work}/PerfMetrics_MeanAnnual.csv',
                dtype = {'STAID': 'string',
                         'region': 'string'}
            )
        
        results_summAll['|residuals|'] = results_summAll.residuals.abs()

        results_summAll = results_summAll[[
                'STAID', 'residuals', '|residuals|', 'clust_method', 'region',\
                            'model', 'time_scale', 'train_val'
            ]]
        
        

    else:
        results_summAll = pd.read_csv(
                f'{dir_work}/NSEComponents_KGE.csv',
                dtype = {'STAID': 'string',
                        'region': 'string'}
            )
        
        results_summAll = results_summAll[
            results_summAll['time_scale'] == timescale
            ]
        
    results_summAll = results_summAll[
        results_summAll['train_val'] == part_in
        ]
    
    if timescale == 'mean_annual':
        metric_temp = '|residuals|'
    else:
        metric_temp = metric_in
    df_summTemp = q_metr(results_summAll, metric_temp)

    # read in results w/PCA and model parameter info
    resultsPCAparams = pd.read_pickle(
        f'{dir_workHPC}/{timescale}/'\
            f'combined/All_SummaryResults_{timescale}.pkl'
    )

    resultsPCAparams['region'] = resultsPCAparams['region'].astype(str)

    # load data to get colnames for output dataframe (all expl vars)
    df_expl, df_WY, df_ID = load_data_fun(
        dir_work = dir_work, 
        time_scale = timescale,
        train_val = part_in,
        clust_meth = 'AggEcoregion',
        region = 'MxWdShld',
        standardize = False # whether or not to standardize data
        )

    # define var holding all explanatory var columns names
    expl_names = df_expl.columns

    # define empty dataframe with all explanatory vars as columns, to append
    # results to
    df_shap_out = pd.DataFrame(
        columns = expl_names
    )
    # drop columns to not keep
    df_shap_out.drop(
        expl_names.intersection(names_drop), 
        axis = 1,
        inplace = True
    )

    del(df_expl, df_WY, df_ID)

    # initialize empty lists to hold cluster methods, regions, and best models
    # and best score (either best NSE or r2) which will be added to output 
    # dataframe at end
    methods_out = []
    regions_out = []
    best_models = []
    best_scores = []

    for method in clust_meths:

        # print update
        print(f'\n Processing scenario: {timescale}-{method} \n')

        # define regions for the clust_meth
        if isinstance(df_regions[method], str):
            region = [df_regions[method]]
        else: 
            region = df_regions[method]
        for cluster in region: 
            
            cluster = str(cluster)

            print(f'\n Cluster: {cluster}')

            # add clust_meth and region to output lists
            methods_out.append(method)
            regions_out.append(cluster)

            # subset to clust_meth and region of interest
            results_summ = results_summAll[
                (results_summAll['clust_method'] == method) & 
                (results_summAll['region'] == cluster)
                ]
            
            # subset to rows of interest in PCA and params dataframe
            temp = resultsPCAparams[
                (resultsPCAparams['train_val'] == 'train') & # will be same for train and test (valnit)
                (resultsPCAparams['clust_method'] == method) &
                (resultsPCAparams['region'] == cluster)
            ].reset_index(drop = True)


            # drop PCA models and extract the number of components it took to 
            # explain 95% of variation in explanatory variables

            # extract number of components
            temp = temp.loc[
                temp['model'].str.contains('PCA'), 'parameters'
                ].values[0]

            try:
                pca95 = int(temp[len(temp) - 2:])
            except:
                pca95 = int(temp[len(temp) - 1:])

            del(temp)

            results_summ = results_summ[
                ~results_summ['model'].str.contains('PCA')
                ].reset_index(drop = True)
            
            temp_df = pd.DataFrame({
                'ClusterMethod': [method],
                'Region': [cluster],
                'PCA95': [pca95]
            })

            df_pca95 = pd.concat(
                [df_pca95, temp_df], 
                ignore_index = True)

            # return the model and parameters for the model performing best on
            # valnit data
            # use NSE if time series data, or r2 if non-time series (e.g., mean annaul)
            # find max valnit value
           
            if timescale == 'mean_annual':
                best_score = np.min(df_summTemp.loc[
                    df_summTemp['train_val'] == part_in, f'{metric_temp}_qmean'
                ])
            else:
                best_score = np.max(df_summTemp.loc[
                    df_summTemp['train_val'] == part_in, f'{metric_temp}_qmean'
                ])

            # subset to best model based on max valnit NSE    
            best_model = df_summTemp.loc[
                (df_summTemp['train_val'] == part_in) &
                (df_summTemp[f'{metric_temp}_qmean'] == best_score),
                'model'
            ].reset_index(drop = True)
            # subset to best parameters based on max valnit metric
            best_params = resultsPCAparams.loc[
                resultsPCAparams['model'] == best_model[0], 'parameters'
                ]
            
            #     df_summTemp.loc[
            #     (df_summTemp['train_val'] == 'valnit') &
            #     (df_summTemp[f'{metric_temp}_qmean'] == max_valnit),
            #     'parameters'
            # ]
            # else:
            #     max_valnit = np.max(df_summTemp.loc[
            #         df_summTemp['train_val'] == 'valnit', 'NSE'
            #         ])
            #     # subset to best model based on max valnit NSE    
            #     best_model = df_summTemp.loc[
            #         (df_summTemp['train_val'] == 'valnit') &
            #         (df_summTemp['NSE'] == max_valnit),
            #         'model'
            #     ]
            #     # subset to best parameters based on max valnit NSE 
            #     best_params = df_summTemp.loc[
            #         (df_summTemp['train_val'] == 'valnit') &
            #         (df_summTemp['NSE'] == max_valnit),
            #         'parameters'
            #     ]



            # append best_model to list of best_models
            # and best score to list of best scores
            best_models.append(best_model)
            best_scores.append(best_score)

            


            # %% 
            # load data (explanatory, water yield, ID)
            #########

            # if xgboost then do not standardize data, otherwise, do
            stndz = best_model != 'XGBoost'

            # load data
            df_expl, df_WY, df_ID = load_data_fun(
                dir_work = dir_work, 
                time_scale = timescale,
                train_val = part_in,
                clust_meth = method,
                region = cluster,
                standardize = np.array(stndz) # whether or not to standardize data
                )

            # DRAIN_SQKM stored as DRAIN_SQKM_x in some output by accident,
            # so replace it
            if 'SQKM_x' in df_expl.columns.values:
                df_expl.columns = df_expl.columns.str.replace('SQKM_x', 'SQKM')


            # remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
            # subset WY to version desired (ft)
            # store staid's and date/year/month
            if(timescale == 'mean_annual'):
                STAID = df_expl['STAID']  
                df_expl.drop('STAID', axis = 1, inplace = True)
                df_WY = df_WY['Ann_WY_cm']
            if(timescale == 'annual'):
                STAID = df_expl[['STAID', 'year']]  
                df_expl.drop(['STAID', 'year'], axis = 1, inplace = True)
                df_WY = df_WY['Ann_WY_cm']
            if(timescale == 'monthly'):
                STAID = df_expl[['STAID', 'year', 'month']]   
                df_expl.drop(['STAID', 'year', 'month'], axis = 1, inplace =True)
                df_WY = df_WY['Mnth_WY_cm']
            if(timescale == 'daily'):
                STAID = df_expl[['STAID', 'date']] 
                df_expl.drop(['STAID', 'year', 'month', 'day', 'date'], axis = 1, inplace =True)
                df_WY = df_WY['dlyWY_cm']

        
            # read in columns that were previously removed due to high VIF
            file = glob.glob(
                f'{dir_workHPC}{timescale}/VIF_Removed/*{method}_{cluster}.csv'
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



            # %% 
            # build best model
            ############


            # %%
            # regression with only precip
            ############

            if best_model.values == 'regr_precip':
                model = LinearRegression()
                model.fit(np.array(df_expl['prcp']).reshape(-1,1), df_WY)

            X_in = df_expl['prcp']


            # %% MLR
            #########################
            # read in regression variables for mlr if best_model is strd_mlr
            if best_model.values == 'strd_mlr':
                # MLR
                file = glob.glob(
                    f'{dir_work}/data_out/{timescale}/VIF_dfs/'
                    f'{method}_{cluster}_strd_mlr*.csv')[0]
                # get variables appearing in final model
                vars_keep = pd.read_csv(
                    file
                )['feature']

                if 'DRAIN_SQKM_x' in vars_keep.values:
                    vars_keep = vars_keep.str.replace('DRAIN_SQKM_x', 'DRAIN_SQKM')

                # subset data to variables used in model
                X_in = df_expl[vars_keep]

                # define model and parameters
                reg = LinearRegression()

                # apply linear regression using all explanatory variables
                # this object is used in feature selection just below and
                # results from this will be used to calculate Mallows' Cp
                model = reg.fit(X_in, df_WY)

            # %% xgboost
            ###################
            # read in xgboost if best model is xgboost
            if best_model.values == 'XGBoost':
                # first define xgbreg object
                model = xgb.XGBRegressor()

                # define temp time_scale var since model names do not have '_' in them
                temp_time = timescale.replace('_', '')

                # reload model into object
                model.load_model(
                    f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{timescale}'
                    f'/Models/XGBoost_{temp_time}_{method}_{cluster}_model.json'
                    )
                X_in = df_expl



            # %%
            # calculate shap values depending on model type
            ##############


            # %%
            # regr_precip
            ###############
            if best_model.values == 'regr_precip':
                df_shapmean = pd.DataFrame(
                    model.coef_,
                    columns = ['prcp']
                )

                df_shap_out = pd.concat(
                    [df_shap_out, df_shapmean],
                    ignore_index = True
                )

            # lasso and stepwise regression
            if best_model.values in ['strd_lasso', 'strd_mlr']:

                # take subsample of specified % to use as background distribution
                # % to subsample
                ratio_sample = 0.50
                Xsubset = shap.utils.sample(X_in, int(np.floor(X_in.shape[0] * ratio_sample)))

                # define explainer
                explainer = shap.LinearExplainer(model, Xsubset)
                
                # calc shap values
                shap_values = explainer(X_in) # df_expl)

                # define shapvalue dataframes
                df_shap_valout = pd.DataFrame(
                    shap_values.values,
                    columns = X_in.columns
                )

                # df_shap_baseout = pd.DataFrame(
                #     shap_values.base_values,
                #     columns = ['base_value']
                # )

                # get temp list, where = -1 if coef (-) or =1 if coef (+)
                temp_coef = model.coef_.copy()
                temp_coef[temp_coef < 0] = -1
                temp_coef[temp_coef >= 0] = 1

                # take mean of shap values, and give direction by 
                # multiplying by temp_coef
                df_shapmean = pd.DataFrame(
                    df_shap_valout.abs().mean() * temp_coef,
                ).T

                # add current df_shapmean to df_shap_out
                df_shap_out = pd.concat(
                    [df_shap_out, df_shapmean],
                    ignore_index = True
                )




            if best_model.values == 'XGBoost':
                # define explainer
                explainer = shap.TreeExplainer(model)
                # calc shap values
                # check_additivity = False because one model
                shap_values = explainer(X_in, check_additivity = False) # df_expl)
                
                df_shap_valout = pd.DataFrame(
                    shap_values.values,
                    columns = X_in.columns
                )

                # df_shap_baseout = pd.DataFrame(
                #     shap_values.base_values
                # )

                # standardaize the explanatory vars for getting slope direction
                # using linear regression
                scaler = StandardScaler()
                scaler.fit(X_in)

                df_explstand = pd.DataFrame(
                    scaler.transform(X_in),
                    columns = X_in.columns
                    )
                
                # define empty list to hold multipliers for direction of
                # shap relationship
                shap_dirxn = []
                # loop through all vars and get direction of relationship
                for colname in X_in.columns:
                    x = np.array(df_explstand[colname]).reshape(-1,1)
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

                df_shapmean = df_shapmean/df_WY.mean()

            # add current df_shapmean to df_shap_out
            df_shap_out = pd.concat(
                [df_shap_out, df_shapmean], 
                ignore_index = True
            )
    
    # df_shap_out = df_shap_out.drop_duplicates().reset_index(drop = True)

    # convert best_models to array, so dtype and name aren't written with each
    best_models = np.concatenate(
        [x.astype(str).values.tolist() for x in best_models]
    )

    # add cluster method and region to output shap dataframe and pca dataframe
    df_shap_out['clust_meth'] = methods_out
    df_shap_out['region'] = regions_out
    df_shap_out['best_model'] = best_models
    df_shap_out['best_score'] = best_scores

    df_shap_out = df_shap_out.drop_duplicates().reset_index(drop = True)

    # write df_shap_out to csv
    df_shap_out.to_csv(
        f'{dir_shapout}/MeanShap_{part_in}_{timescale}_normQ.csv',
        index = False,
        mode = 'a'
    )

    # write pcas to csv
    df_pca95.to_csv(
        f'{dir_pcaout}/PCA95_{part_in}_{timescale}.csv',
        index = False,
        mode = 'a'
    )

print("\n\n---------------COMPLETE----------------\n\n")