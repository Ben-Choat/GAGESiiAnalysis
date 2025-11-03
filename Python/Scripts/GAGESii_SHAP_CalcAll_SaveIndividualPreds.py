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
import os




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
    # 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/' \
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/HPC_Files/'\
        'GAGES_Work/data_work/GAGESiiVariables/ID_train.csv'
)

# time scales
time_scale = ['monthly', 'annual', 'mean_annual']
# time_scale = ['mean_annual']

# cluster methods to include in analysis
clust_meth_in = ['None', 'Class', 'AggEcoregion']

# models to consider in analysis
# models_in = ['regr_precip', 'strd_mlr', 'XGBoost']
models_in = ['XGBoost']

# partition in (train or valint (aka testing))
# part_in = 'train'
part_in = ['train', 'valnit']


# use 'NSE' or 'KGE'? '|residuals| always used for mean_annual
metric_in = 'NSE' # 'KGE'

# drop noise? True or False
dropNoise = False

# Define directory variables
# directory with data to work with
# dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results' 
dir_work = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results'

# another main location with output from HPC runs.
# dir_workHPC = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/'
dir_workHPC = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                'HPC_Files/GAGES_Work/data_out/'
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
# dir_shapout = 'D:/Projects/GAGESii_ANNstuff/Data_Out/SHAP_OUT'
dir_shapout = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                    'Data_Out/SHAP_OUT'\


# name list of column names to drop when defining output dataframe of shap values
names_drop = ['STAID', 'year', 'month', 'day', 'date']


# %%
# Load data
###########

# define dict to hold cluster methods: cluster
df_regions = {}

# define regions for different clustering methods
for cl in clust_meths:
    if cl == 'None':
        df_regions[cl] = ['All']
    else:
        df_regions[cl] = np.sort(df_ID[cl].unique())


for timescale in time_scale[1:2]:
    print(timescale)
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
        results_summAll['train_val'].isin(part_in)
        ]
    
    if timescale == 'mean_annual':
        metric_temp = '|residuals|'
    else:
        metric_temp = metric_in
    # df_summTemp = q_metr(results_summAll, metric_temp)


    results_summAll = results_summAll.query(
        "clust_method in @clust_meth_in &"\
            "model in @models_in &" \
                "train_val in @part_in"
    )


    # subset to only best score
    results_summAll = results_summAll.sort_values(
            by = ['STAID', metric_temp]
    )

    # keep best for each STAID
    if timescale == 'mean_annual':
        results_summAll = results_summAll.drop_duplicates(
            subset = 'STAID', keep = 'first'
        )
    else:
        results_summAll = results_summAll.drop_duplicates(
            subset = 'STAID', keep = 'last'
        )

    # troubleshooting
    # results_summAll = results_summAll.drop_duplicates(subset=['model', 'time_scale'])
    # results_summAll = results_summAll.query("STAID == '05517530'")

    # load data to get colnames for output dataframe (all expl vars)
    df_expl, df_WY, df_ID = load_data_fun(
        dir_work = dir_work, 
        time_scale = timescale,
        train_val = part_in[0],
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

    # create variables to track clust method, region, and model working
    # with to see if need to load new data or change variables
    last_clust = None
    last_trainval = None
    last_region = None
    last_model = None

    # sort so can load data as few times as possible
    results_summAll = results_summAll.sort_values(
                by = ['train_val', 'clust_method', 'region', 'model']
                )
    
    print(results_summAll.head())
    
    # # create list to hold output information
    # staid_out = []
    # clust_meth_out = []
    # region_out = []
    # time_scale_out = []


    for i, row in results_summAll.reset_index(drop=True).iterrows():
        # if row['STAID'] == '03354000': break # troubleshooting
        print(f'\ncatchment-{i}')
        print(f"STAID: {row['STAID']}")
        print(f'time scale: {timescale}')
        # print(row)
        # print(row[['STAID', 'NSE', 'KGE', 'clust_method', 'region', '']])
    


        if (last_clust != row['clust_method']) | \
            (last_trainval != row['train_val']) | \
                (last_region != row['region']) | \
                    (last_model != row['model']):
            
            print(list(row[['clust_method', 'region', 'model']]))
            # define if data should be standardized or not
            if row['model'] != 'XGBoost':
                stndz = True
            else:
                stndz = False
            
            # load data (explanatory, water yield, ID)
            #########
            # load data
            df_expl, df_WY, df_ID = load_data_fun(
                dir_work = dir_work, 
                time_scale = timescale,
                train_val = row['train_val'],
                clust_meth = row['clust_method'],
                region = row['region'],
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
                STAID = df_expl[['STAID']]  
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
                f'{dir_workHPC}{timescale}/VIF_Removed/*{row["clust_method"]}'\
                    f'_{row["region"]}.csv'
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

        last_clust = row['clust_method']
        last_trainval = row['train_val']
        last_region = row['region']
        last_model = row['model']

        df_expl_in = df_expl[STAID['STAID'] == row['STAID']].reset_index(drop = True)
        df_WY_in = df_WY[STAID['STAID'] == row['STAID']].reset_index(drop = True)

            
            

         # %% 
        # build best model
        ############


        # %%
        # regression with only precip
        ############

        if row['model'] == 'regr_precip':
            X_in = df_expl_in['prcp']
            model = LinearRegression()
            model.fit(np.array(X_in).reshape(-1,1), df_WY_in)

            df_shapmean = pd.DataFrame(
                model.coef_,
                columns = ['prcp']
            )

            # df_shap_out = pd.concat(
            #     [df_shap_out, df_shapmean],
            #     ignore_index = True
            # )



        # %% MLR
        #########################
        # read in regression variables for mlr if best_model is strd_mlr
        if row['model'] in ['strd_lasso', 'strd_mlr']:
            # MLR
            ##############BEN START HERE
            file = glob.glob(
                f'{dir_workHPC}/{timescale}/VIF_dfs/'\
                f'{row["clust_method"]}_{row["region"]}_strd_mlr*.csv')[0]
            # get variables appearing in final model
            vars_keep = pd.read_csv(
                file
            )['feature']

            if 'DRAIN_SQKM_x' in vars_keep.values:
                vars_keep = vars_keep.str.replace('DRAIN_SQKM_x', 'DRAIN_SQKM')

            # subset data to variables used in model
            X_in = df_expl_in[vars_keep]

            # define model and parameters
            reg = LinearRegression()

            # apply linear regression using all explanatory variables
            # this object is used in feature selection just below and
            # results from this will be used to calculate Mallows' Cp
            model = reg.fit(X_in, df_WY_in)
            
            # take subsample of specified % to use as background distribution
            # % to subsample
            ratio_sample = 1
            Xsubset = shap.utils.sample(X_in, int(np.floor(X_in.shape[0] * ratio_sample)))

            # define explainer
            explainer = shap.LinearExplainer(model, Xsubset)
            
            # calc shap values
            shap_values = explainer(X_in) # df_expl_in)

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
            # df_shap_out = pd.concat(
            #     [df_shap_out, df_shapmean],
            #     ignore_index = True
            # )





        # %% xgboost
        ###################
        # read in xgboost if best model is xgboost
        if row['model'] == 'XGBoost':
            # first define xgbreg object
            model = xgb.XGBRegressor()

            # define temp time_scale var since model names do not have '_' in them
            temp_time = timescale.replace('_', '')

            # reload model into object
            model.load_model(
                # f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{timescale}'
                'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                    f'HPC_Files/GAGES_Work/data_out/{timescale}/Models/'\
                        f'XGBoost_{temp_time}_'\
                    f'{row["clust_method"]}_{row["region"]}_model.json'
                )
            X_in = df_expl_in

             # define explainer
            explainer = shap.TreeExplainer(model)
            # calc shap values
            # check_additivity = False because one model
            shap_values = explainer(X_in, check_additivity = False) # df_expl_in)
            
            df_shap_valout = pd.DataFrame(
                shap_values.values,
                columns = X_in.columns
            )           

            # summary plot
            # shap.summary_plot(shap_values, X_in)
            # Force Plot
            # shap.initjs()
            # shap.plots.force(
            #     explainer.expected_value,
            #     explainer.shap_values(X_in)[0],
            #     X_in.iloc[0]
            # )
            # dependence plot
            # shap.dependence_plot(
            #     "prcp",
            #     explainer.shap_values(X_in),
            #     X_in,
            #     interaction_index="WWTP_Effluent"
            # )
            # decision plot
            # shap.decision_plot(
            #     explainer.expected_value,
            #     explainer.shap_values(X_in),
            #     X_in.columns
            # )

            # add mean WY to output df
            df_shap_valout['WY_cm'] = df_WY_in

            # swe_1 instead of swe appeard in some mean annual models
            # so correct column name to swe
            df_shap_valout.columns = df_shap_valout.columns.str.replace('swe_1', 'swe')



        # # add current df_shapmean to df_shap_out
        df_shapmean = pd.concat(
            [df_shap_out, df_shap_valout], 
            axis=0
            # ignore_index=True
        )
       


        # df_out = pd.concat([results_summAll, df_shap_out])

        # results_summAll.reset_index(drop=True, inplace=True)
        results_out = row.to_frame().T.reset_index(drop=True)
        # df_shap_out.reset_index(drop=True, inplace=True)

        # df_out = pd.concat([results_summAll, df_shap_out], axis = 1)
        # df_out = pd.concat([results_out, df_shap_out], axis = 1)
        df_out = pd.concat([results_out, df_shapmean], axis = 1)

        print(df_out['prcp'])

        # write df_shap_out to csv
        file_out = f'{dir_shapout}/MeanShap_BestGrouping_All_{timescale}_normQ.csv'

        if os.path.exists(file_out):
            header_in = False
        else:
            header_in = True

        df_out.to_csv(
            file_out,
            index = False, mode = 'a', header = header_in
        )
           


    # # df_shap_out = df_shap_out.drop_duplicates().reset_index(drop = True)

    # # convert best_models to array, so dtype and name aren't written with each
    # best_models = np.concatenate(
    #     [x.astype(str).values.tolist() for x in best_models]
    # )

    # # add cluster method and region to output shap dataframe and pca dataframe
    # df_shap_out['clust_meth'] = methods_out
    # df_shap_out['region'] = regions_out
    # df_shap_out['best_model'] = best_models
    # df_shap_out['best_score'] = best_scores

    # df_shap_out = df_shap_out.drop_duplicates().reset_index(drop = True)

    # write df_shap_out to csv
    # df_shap_out.to_csv(
    #     f'{dir_shapout}/MeanShap_BestModel_All_{part_in}_{timescale}_normQ.csv',
    #     index = False,
    #     mode = 'a'
    # )

    # # write pcas to csv
    # # df_pca95.to_csv(
    # #     f'{dir_pcaout}/PCA95_{part_in}_{timescale}.csv',
    # #     index = False,
    # #     mode = 'a'
    # # )

print("\n\n---------------COMPLETE----------------\n\n")