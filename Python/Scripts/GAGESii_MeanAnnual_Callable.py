# Ben Choat, 2022/07/06

# Script for executing analysis for dissertation on mean water yield and explanatory vars
# Ch4: Understanding drivers of water yield and how they vary across temporal scales

# Raw: Explanatory variables directly
# Rescale: Standardization or Normalization
# Reduce: PCA or UMAP
# Cluster: k-means, k-medoids, HDBSCAN, ecoregions
    # cluster on 1. all expl. vars, 2. physcial catchments vars, 3. anthropogenic vars,
    # 4. climate vars, 5. stack (2) and (3), 6. stack (2) and (4), 7. stack (2), (4), (3)

# Predict: OLS-MLR, Lasso, XGBOOST

# Raw -> Predict (done - excluding XGBOOST)

# Raw -> Rescale -> Predict (done - excluding XGBOOST)

# Raw -> Rescale -> Reduce -> Predict (done - excluding XGBOOST)

# Raw -> Rescale -> Cluster -> Predict

# Raw -> Rescale -> Reduce -> Cluster -> Predict

# Raw -> Rescale -> Cluster -> Cluster -> Predict


# %% Import classes and libraries
# from statistics import LinearRegression
from GAGESii_Class import Clusterer
from GAGESii_Class import Regressor
from Regression_PerformanceMetrics_Functs import *
import pandas as pd
# import plotnine as p9
# import plotly.express as px # easier interactive plots
# from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from os.path import exists
# from sklearn.linear_model import Lasso
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# import umap


# %% DEFINE FUNCTION TO APPLY ALL REGRESSION MODELS AND ASK FOR USER INPUT WHERE NEEDED

def regress_fun(df_train_expl, # training data explanatory variables. Expects STAID to be a column
                df_testin_expl, # validation data explanatory variables using same catchments that were trained on
                df_valnit_expl, # validation data explanatory variables using different catchments than were trained on
                train_resp, # training data response variables. NOTE: this should be a series, not a dataframe
                testin_resp, # validation data response variables using same catchments that were trained on
                valnit_resp, # validation data response variables using different catchments than were trained on
                train_ID, # training data id's (e.g., clusters or ecoregions)
                testin_ID, # validation data id's from catchments used in training (e.g., clusters or ecoregions)
                valnit_ID, # validation data id's from catchments not used in training (e.g., clusters or ecoregions)
                clust_meth, # the clustering method used. This variable is used for naming models (e.g., AggEcoregion)
                reg_in, # region label, i.e., 'NorthEast'
                grid_in # dict with XGBoost parameters
                ):

    # instantiate counter to know how many rows to edit with the current region label
    mdl_count = 0

    # %% Define variables
    # explantory var (and other data) directory
    dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

    # %% Load or define dataframe to append results to as models are generated
    # %% Define dataframe to append results to as models are generated
    try:
        df_results_temp = pd.read_csv(f'{dir_expl}/Results/Results_NonTimeSeries.csv')
    except:
        df_results_temp = pd.DataFrame({
            'model': [], # specify which model, e.g., 'raw_lasso'
            'train_val': [], # specify if results from training, validation (i.e., testin or testint)
            'parameters': [], # any hyperparameters (e.g., alpha penalty in lasso regression)
            'n_features': [], # number of explanatory variables (i.e., featuers)
            'ssr': [], # sum of squared residuals
            'r2': [], # unadjusted R2
            'r2adj': [], # adjusted R2
            'mae': [], # mean absolute error
            'rmse': [], # root mean square error
            'VIF': [], # variance inflation factor (vector of all included featuers and related VIFs)
            'percBias': [], # percent bias
            'region': [], # the region or cluster number being modeled
            'clust_method': [] # the clustering method used to generate region
        })



    # Define features not to transform
    not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
                'GEOL_REEDBUSH_DOM_quarternary', 
                'GEOL_REEDBUSH_DOM_sedimentary', 
                'GEOL_REEDBUSH_DOM_ultramafic', 
                'GEOL_REEDBUSH_DOM_volcanic']
                # 'GEOL_REEDBUSH_DOM_gneiss', 

    # subset not_tr_in to those columns remaining in dataframe
    # in case any were removed due to high VIFs
    not_tr_in = df_train_expl.columns.intersection(not_tr_in)







    #########################

    # %% Mean annual precipitation as only predictor ols
    print('resp vs. mean annual precipitation')

    model_name = 'regr_precip' # input('enter name for model (e.g., regr_precip):') # regr_precip'

    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):
        # update model count
        mdl_count = mdl_count + 1
        
        parameters = 'none'
        n_features = 1

        train_val = 'train'
        print(train_val)

        # define explanatory and response vars
        expl_in = np.array(df_train_expl['PPTAVG_BASIN']).reshape(-1,1)
        resp_in = train_resp

        model = LinearRegression().fit(expl_in, resp_in)
        y_predicted = model.predict(expl_in)

        # ssr
        ssr_out = ssr(y_predicted, resp_in)
        # r2
        r2_out = model.score(expl_in, resp_in)
        # adjr2
        r2adj_out = R2adj(len(resp_in), 1, r2_out)
        # MAE
        mae_out = mean_absolute_error(resp_in, y_predicted)
        # RMSE
        rmse_out = mean_squared_error(resp_in, y_predicted, squared = False)

        # % bias
        pbias_out = PercentBias(y_predicted, resp_in)

        to_append = pd.DataFrame({
            'model': model_name,
            'train_val': train_val,
            'parameters': parameters,
            'n_features': n_features,
            'ssr': ssr_out,
            'r2': r2_out,
            'r2adj': r2adj_out,
            'mae': mae_out,
            'rmse': rmse_out,
            'VIF': 1,
            'percBias': pbias_out
        }, index = [0])

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)
        
        # prepare dataframe for plotting performance
        df_in = pd.DataFrame({
            'observed': resp_in,
            'predicted': y_predicted,
            'ID': train_ID
        })

        print('training')

        # Plot predicted vs observed
        p = (
                p9.ggplot(data = df_in) +
                p9.geom_point(p9.aes(x = 'observed', 
                                        y = 'predicted', 
                                        color = 'ID')) +
                p9.geom_abline(slope = 1) +
                p9.theme_bw() +
                p9.theme(axis_text = p9.element_text(size = 14),
                            axis_title = p9.element_text(size = 14),
                            aspect_ratio = 1,
                            legend_text = p9.element_text(size = 14),
                            legend_title = p9.element_text(size = 14))
            )
        print(p)


        #####
        # testin data
        train_val = 'testin'
        print(train_val)

        # define explanatory and response vars
        expl_in = np.array(df_testin_expl['PPTAVG_BASIN']).reshape(-1,1)
        resp_in = testin_resp

        model = model
        y_predicted = model.predict(expl_in)

        # ssr
        ssr_out = ssr(y_predicted, resp_in)
        # r2
        r2_out = model.score(expl_in, resp_in)
        # adjr2
        r2adj_out = R2adj(len(resp_in), 1, r2_out)
        # MAE
        mae_out = mean_absolute_error(resp_in, y_predicted)
        # RMSE
        rmse_out = mean_squared_error(resp_in, y_predicted, squared = False)

        # % bias
        pbias_out = PercentBias(y_predicted, resp_in)

        to_append = pd.DataFrame({
            'model': model_name,
            'train_val': train_val,
            'parameters': parameters,
            'n_features': n_features,
            'ssr': ssr_out,
            'r2': r2_out,
            'r2adj': r2adj_out,
            'mae': mae_out,
            'rmse': rmse_out,
            'VIF': 1,
            'percBias': pbias_out
        }, index = [0])

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)
        
        
        # prepare dataframe for plotting performance
        df_in = pd.DataFrame({
            'observed': resp_in,
            'predicted': y_predicted,
            'ID': testin_ID
        })

        

        # Plot predicted vs observed
        p = (
                p9.ggplot(data = df_in) +
                p9.geom_point(p9.aes(x = 'observed', 
                                        y = 'predicted', 
                                        color = 'ID')) +
                p9.geom_abline(slope = 1) +
                p9.theme_bw() +
                p9.theme(axis_text = p9.element_text(size = 14),
                            axis_title = p9.element_text(size = 14),
                            aspect_ratio = 1,
                            legend_text = p9.element_text(size = 14),
                            legend_title = p9.element_text(size = 14))
            )
        print(p)

        ####
        # valnit

        train_val = 'valnit'
        print(train_val)

        # define explanatory and response vars
        expl_in = np.array(df_valnit_expl['PPTAVG_BASIN']).reshape(-1,1)
        resp_in = valnit_resp

        model = model
        y_predicted = model.predict(expl_in)

        # ssr
        ssr_out = ssr(y_predicted, resp_in)
        # r2
        r2_out = model.score(expl_in, resp_in)
        # adjr2
        r2adj_out = R2adj(len(resp_in), 1, r2_out)
        # MAE
        mae_out = mean_absolute_error(resp_in, y_predicted)
        # RMSE
        rmse_out = mean_squared_error(resp_in, y_predicted, squared = False)

        # % bias
        pbias_out = PercentBias(y_predicted, resp_in)

        to_append = pd.DataFrame({
            'model': model_name,
            'train_val': train_val,
            'parameters': parameters,
            'n_features': n_features,
            'ssr': ssr_out,
            'r2': r2_out,
            'r2adj': r2adj_out,
            'mae': mae_out,
            'rmse': rmse_out,
            'VIF': 1,
            'percBias': pbias_out
        }, index = [0])

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)



        # prepare dataframe for plotting performance
        df_in = pd.DataFrame({
            'observed': resp_in,
            'predicted': y_predicted,
            'ID': valnit_ID
        })

        # Plot predicted vs observed
        p = (
                p9.ggplot(data = df_in) +
                p9.geom_point(p9.aes(x = 'observed', 
                                        y = 'predicted', 
                                        color = 'ID')) +
                p9.geom_abline(slope = 1) +
                p9.theme_bw() +
                p9.theme(axis_text = p9.element_text(size = 14),
                            axis_title = p9.element_text(size = 14),
                            aspect_ratio = 1,
                            legend_text = p9.element_text(size = 14),
                            legend_title = p9.element_text(size = 14))
            )
        print(p)











    ########################
    #########################
    # NEXT MODELING PATH
    ########################
    #########################








    # %% 
    # Raw -> Predict
    ###################

    # regression on untransformed explanatory variables
    # Instantiate a Regressor object 
    regr = Regressor(expl_vars = df_train_expl.drop(columns = ['STAID']),
            resp_var = train_resp)

    # model name for saving in csv
    model_name = 'raw_lasso' # input('enter name for model (e.g., raw_lasso):') 
    
    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1



        # %%
        ##### Lasso regression
        print('raw -> lasso')
        # search via cross validation
        regr.lasso_regression(
            alpha_in = list(np.arange(0.01, 1.01, 0.01)), # must be single integer or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100,
            n_jobs_in = -1
        )
        # print all results from CV
        pd.DataFrame(regr.lassoCV_results.cv_results_)

        print('top 10 results in order for mae, rmse, and r2')
        print(regr.df_lassoCV_mae[0:10])
        print(regr.df_lassoCV_rmse[0:10])
        print(regr.df_lassoCV_r2[0:10])

        # %%
        
        # define alpha
        a_in = float(input('based on those results, what value do you want to use for alpha?'))



        train_val = 'train'
        print(train_val)
        ##### apply best alpha identified in cross validation
        # Lasso with chosen alpha

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'alpha{a_in}'

        regr.lasso_regression(
            alpha_in = float(a_in), # 0.01, # must be single float or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100
        )

        # print features kept by Lasso regression and the associated coefficients
        # regr.df_lasso_features_coef_.sort_values(by = 'coefficients')

        # plot regression
        mdl_in = regr.lasso_reg_
        expl_in = regr.expl_vars # df_train_expl.drop(columns = 'STAID')
        resp_in = train_resp
        # resp_in = train_mnanWY_tr
        id_in = train_ID

        print('training')
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy() # NOTE: copy so original df is not edited in place
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )


        #####
        train_val = 'testin'
        print(train_val)

        # return prediction metrics for validation data from stations that were trained on
        # plot regression
        mdl_in = mdl_in
        expl_in = df_testin_expl.drop(columns = 'STAID')
        resp_in = testin_resp
        id_in = testin_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        #####
        train_val = 'valnit'
        print(train_val)
        # return prediction metrics for validation data from stations that were NOT trained on
        # plot regression
        mdl_in = mdl_in
        expl_in = df_valnit_expl.drop(columns = 'STAID')
        resp_in = valnit_resp
        # resp_in = train_mnanWY_tr
        id_in = valnit_ID

        print('valnit')
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )


        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)



###############################################

    

    # %% 
    # MLR feature selection - 'forward'
    print('raw -> MLR')
    # define model name and parameter name(s) to be written to files and used in file names 

    # model name for saving in csv
    model_name = 'raw_mlr' # input('enter name for model (e.g., raw_mlr):') 

    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1

        # define klim  (how many variables to consider in model)
        # klim_in = int(input('How many variables do you want to consider (0<klim<81): '))
        klim_in = (df_train_expl.shape[1] - 2) if df_train_expl.shape[1] < df_train_expl.shape[0] else df_train_expl.shape[0] - 2
        
        ##### apply best alpha identified in cross validation
        # Lasso with chosen alpha

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'forwardklim{klim_in}'

        # model_name = 'raw_mlr'
        # param_name = 'forward_klim81'

        regr.lin_regression_select(
            sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
            float_opt = 'True', # 'True' or 'False'
            min_k = klim_in, # only active for 'exhaustive' option
            klim_in = klim_in, # controls max/min number of features for forward/backward selection
            timeseries = False, # if timeseries = True then NSE and KGE are also calculated
            n_jobs_in = -1) # number of cores to distribute to
                            # Note, on campus computer, using -1 instead of 1 sped up from about 16 minutes to about 6

        # define variable holding the selected features and vifs.
        n_f_in = int(input('enter the number of features in the model you want to use (e.g., 20): '))
        vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == n_f_in, 'VIF']

        # Extract feature names for selecting features
        features_in = pd.DataFrame(dict((vif_in))).index

        # Subset appropriate explanatory variables to columns of interest
        # validation data from catchments used in training
        expl_in = df_train_expl[features_in]

        # define response variable
        resp_in = train_resp

        # define id vars
        id_in = train_ID
        
        train_val = 'train'
        print(train_val)

        # OLS regression predict
        # specifiy input model
        mdl_in = LinearRegression().fit(
                    # df_train_expl[features_in], train_resp
                    expl_in, resp_in
                    )

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )

        ##### 
        train_val= 'testin'
        print(train_val)
        # Apply to validation catchments used in training (i.e., testin)
        # Subset appropriate explanatory variables to columns of interest
        expl_in = df_testin_expl[features_in]

        # define response variable
        resp_in = testin_resp

        # define id vars
        id_in = testin_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = mdl_in

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        #####
        train_val = 'valnit'
        print(train_val)
        # Apply to validation catchments not used in training (i.e., valnit)
        # Subset appropriate explanatory variables to columns of interest
        expl_in = df_valnit_expl[features_in]

        # define response variable
        resp_in = valnit_resp

        # define id vars
        id_in = valnit_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = mdl_in

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)


















    ########################
    #########################
    # NEXT MODELING PATH
    ########################
    #########################



    # %% Raw -> Rescale (standardize) -> Predict
    ##############################

    # regression on transformed explanatory variables
    # Instantiate a Regressor object 
    regr = Regressor(expl_vars = df_train_expl.drop(columns = ['STAID']),
        resp_var = train_resp) # train_mnanWY_tr)# 

    regr.stand_norm(method = 'standardize', # 'normalize'
        not_tr = not_tr_in)


    # %%
    ##### Lasso regression
    print('standardize -> lasso')
    # model name for saving in csv
    model_name = 'strd_lasso' # input('enter name for model (e.g., strd_lasso):') 

    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1

        # search via cross validation
        regr.lasso_regression(
            alpha_in = list(np.arange(0.01, 1.01, 0.01)), # must be single integer or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100,
            n_jobs_in = -1
        )
        # print all results from CV
        pd.DataFrame(regr.lassoCV_results.cv_results_)

        print('top 10 results from cross-validation for mae, rmse, and r2')
        print(regr.df_lassoCV_mae[0:10])
        print(regr.df_lassoCV_rmse[0:10])
        print(regr.df_lassoCV_r2[0:10])

        # %%

        train_val = 'train'
        print(train_val)
        ##### apply best alpha identified in cross validation
        # define alpha
        a_in = float(input('based on those results, what value do you want to use for alpha?'))
        
        ##### apply best alpha identified in cross validation
        # Lasso with chosen alpha

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'alpha{a_in}'

        # Lasso with alpha = 0.01
        regr.lasso_regression(
            alpha_in = float(a_in), # must be single integer or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100
        )

        # plot regression
        mdl_in = regr.lasso_reg_
        expl_in = regr.expl_vars_tr_
        resp_in = train_resp
        # resp_in = train_mnanWY_tr
        id_in = train_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy() # NOTE: copy so original df is not edited in place
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )

        #####
        train_val = 'testin'
        print(train_val)
        # return prediction metrics for validation data from stations that were trained on
        # plot regression
        mdl_in = mdl_in
        expl_in = pd.DataFrame(regr.scaler_.transform(df_testin_expl.drop(columns = 'STAID')))
        expl_in.columns = regr.expl_vars.columns
        expl_in[not_tr_in] = regr.expl_vars[not_tr_in]
        resp_in = testin_resp
        # resp_in = train_mnanWY_tr
        id_in = testin_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )


        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name # 'stdrd_lasso'
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        #####
        train_val = 'valnit'
        print(train_val)
        # return prediction metrics for validation data from stations that were NOT trained on
        # plot regression
        mdl_in = mdl_in
        expl_in = pd.DataFrame(regr.scaler_.transform(df_valnit_expl.drop(columns = 'STAID')))
        expl_in.columns = regr.expl_vars.columns
        expl_in[not_tr_in] = regr.expl_vars[not_tr_in]
        resp_in = valnit_resp
        # resp_in = train_mnanWY_tr
        id_in = valnit_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )


        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name # 'raw_lasso'
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)



    #################################



    ##### 
    # %% MLR feature selection - 'forward'

    print('stdrd -> MLR')

    # model name for saving in csv
    model_name = 'strd_mlr' # input('enter name for model (e.g., stdrd_mlr):') 
    
    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1
    
        # define klim  (how many variables to consider in model)
        # klim_in = int(input('How many variables do you want to consider (0<klim<81): '))
        klim_in = (df_train_expl.shape[1] - 2) if df_train_expl.shape[1] < df_train_expl.shape[0] else df_train_expl.shape[0] - 2
        
        ##### apply best alpha identified in cross validation
        # Lasso with chosen alpha

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'forwardklim{klim_in}'

        # define model name and parameter name(s) to be written to files and used in file names
        # model_name = 'stdrd_mlr'
        # param_name = 'forward_klim81'

        regr.lin_regression_select(
            sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
            float_opt = 'True', # 'True' or 'False'
            min_k = klim_in, # only active for 'exhaustive' option
            klim_in = klim_in, # controls max/min number of features for forward/backward selection
            timeseries = False, # if timeseries = True then NSE and KGE are also calculated
            n_jobs_in = -1) # number of cores to distribute to
                            # Note, on campus computer, using -1 instead of 1 sped up from about 16 minutes to about 6

        # %%
        train_val = 'train'
        print(train_val)
        # print performance metric dataframe subset to n_features of desired number
        # regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == 20,]
        # define variable holding the selected features and vifs.

        # define variable holding the selected features and vifs.
        n_f_in = int(input('enter the number of features in the model you want to use (e.g., 41): '))
        vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == n_f_in, 'VIF']

        # Extract feature names for selecting features
        features_in = pd.DataFrame(dict((vif_in))).index

        # Subset appropriate explanatory variables to columns of interest
        # validation data from catchments used in training
        expl_in = df_train_expl[features_in]

        # define response variable
        resp_in = train_resp

        # define id vars
        id_in = train_ID
        # id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

        # OLS regression predict
        # specifiy input model
        mdl_in = LinearRegression().fit(
                    # df_train_mnexpl[features_in], df_train_mnanWY
                    expl_in, resp_in
                    )

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )

        ##### 
        train_val = 'testin'
        print(train_val)
        # Apply to validation catchments used in training (i.e., testin)
        # Subset appropriate explanatory variables to columns of interest
        expl_in = df_testin_expl[features_in]

        # define response variable
        resp_in = testin_resp

        # define id vars
        id_in = testin_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = LinearRegression().fit(
                    # df_train_mnexpl[features_in], df_train_mnanWY
                    expl_in, resp_in
                    )

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        #####
        train_val = 'valnit'
        print(train_val)
        # Apply to validation catchments not used in training (i.e., valnit)
        # Subset appropriate explanatory variables to columns of interest
        expl_in = df_valnit_expl[features_in]

        # define response variable
        resp_in = valnit_resp

        # define id vars
        id_in = valnit_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = LinearRegression().fit(
                    # df_train_mnexpl[features_in], df_train_mnanWY
                    expl_in, resp_in
                    )

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)







   
















    ########################
    #########################
    # NEXT MODELING PATH
    ########################
    #########################



    # %%  Raw -> Rescale (Standardization) -> Reduce (PCA) -> Predict (Lasso)
    ##############################

    print('stdrd -> PCA -> lasso')

    # model name for saving in csv
    model_name = 'strd_PCA_lasso' # input('enter name for model (e.g., stdrd_PCA_lasso):') 


    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1

        # PCA
        # define cluster/reducer object
        clust = Clusterer(clust_vars = df_train_expl.drop(columns = ['STAID']),
            id_vars = df_train_expl['STAID'])

        # standardize data
        clust.stand_norm(method = 'standardize', # 'normalize'
            not_tr = not_tr_in) 

        # perform PCA on training data and plot
        clust.pca_reducer(
            nc = None, # None option includes all components
            color_in = train_ID, # 'blue'
            plot_out = True
        )


        # regression on transformed explanatory variables

        # define explanatory variables - subset to first 38 components
        # since they explain 95% of the variance in the explanatory variables
        # max_comp = int(input('based on those results, how many components do you want to consider? (e.g., 38): '))
        max_comp = clust.pca95
        expl_vars_in = clust.df_pca_embedding_.iloc[:, 0:max_comp]

        # Instantiate a Regressor object 
        regr = Regressor(expl_vars = expl_vars_in,
            resp_var = train_resp)



        # %%
        ##### Lasso regression
        # search via cross validation
        regr.lasso_regression(
            alpha_in = list(np.arange(0.01, 1.01, 0.01)), # must be single integer or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100,
            n_jobs_in = -1
        )
        # print all results from CV
        pd.DataFrame(regr.lassoCV_results.cv_results_)

        print('top 10 results from cross-validation for mae, rmse, and r2')
        print(regr.df_lassoCV_mae[0:10])
        print(regr.df_lassoCV_rmse[0:10])
        print(regr.df_lassoCV_r2[0:10])

        # %%

        ##### apply best alpha identified in cross validation
        # define alpha
        a_in = float(input('based on those results, what value do you want to use for alpha?'))

        
        
        ##### apply best alpha identified in cross validation
        train_val = 'train'
        print(train_val)
        # Lasso with chosen alpha

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'alpha{a_in}nc{max_comp}'

        # # define model name and parameter name(s) to be written to files and used in file names
        # model_name = 'stdrd_PCA_lasso'
        # param_name = 'alpha0.01nc38'

        # Lasso with alpha = 0.01
        regr.lasso_regression(
            alpha_in = float(a_in), # must be single integer or list
            max_iter_in = 1000,
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100
        )

        # plot regression
        mdl_in = regr.lasso_reg_
        expl_in = regr.expl_vars
        resp_in = train_resp
        # resp_in = train_mnanWY_tr
        id_in = train_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy() # NOTE: copy so original df is not edited in place
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})

        # Max VIF is 1, so no need to write to csv
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )

        #####

        # return prediction metrics for validation data from stations that were trained on
        # plot regression
        train_val = 'testin'
        print(train_val)

        # define model in
        mdl_in = regr.lasso_reg_
        # standardize explantory vars, give columns names, and replace vars not to be transformed
        # define expl vars to work with
        df_in = df_testin_expl.drop(columns = 'STAID')
        expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
        expl_in.columns = df_in.columns
        expl_in[not_tr_in] = df_in[not_tr_in]

        # project explanatory variables into pca space and give new df column names
        expl_in_pcatr = pd.DataFrame(
            clust.pca_fit_.transform(expl_in)
        )
        # pca reduced data column names
        expl_in_pcatr.columns = [f'Comp{i}' for i in np.arange(0, expl_in_pcatr.shape[1], 1)]
        # subset to number of components used in model (e.g., 38)
        expl_in_pcatr = expl_in_pcatr.iloc[:, 0:max_comp]
        expl_in = expl_in_pcatr
        resp_in = testin_resp
        # resp_in = train_mnanWY_tr
        id_in = testin_ID

        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name # 'stdrd_lasso'
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        #####
        # return prediction metrics for validation data from stations that were NOT trained on
        # plot regression
        train_val = 'valnit'
        print(train_val)
        
        
        # define model in
        mdl_in = regr.lasso_reg_
        # standardize explantory vars, give columns names, and replace vars not to be transformed
        # define expl vars to work with
        df_in = df_valnit_expl.drop(columns = 'STAID')
        expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
        expl_in.columns = df_in.columns
        expl_in[not_tr_in] = df_in[not_tr_in]

        # project explanatory variables into pca space and give new df column names
        expl_in_pcatr = pd.DataFrame(
            clust.pca_fit_.transform(expl_in)
        )
        # pca reduced data column names
        expl_in_pcatr.columns = [f'Comp{i}' for i in np.arange(0, expl_in_pcatr.shape[1], 1)]
        # subset to number of components used in model (e.g., 38)
        expl_in_pcatr = expl_in_pcatr.iloc[:, 0:max_comp]
        expl_in = expl_in_pcatr
        resp_in = valnit_resp
        # resp_in = train_mnanWY_tr
        id_in = valnit_ID


        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name # 'raw_lasso'
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)

    


######################################################################################



    ##### 
    # %%  Raw -> Rescale (Standardization) -> Reduce (PCA) -> Predict (OLS-MLR)
    # 
    # MLR feature selection - 'forward'

    # define model name and parameter name(s) to be written to files and used in file names
    # note there are 38 components in the model
    
    print('stdrd -> PCA -> MLR')

    


    # model name for saving in csv
    model_name = 'strd_PCA_mlr' # input('enter name for model (e.g., stdrd_PCA_mlr):') 

    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1
        
        # PCA
        # define cluster/reducer object
        clust = Clusterer(clust_vars = df_train_expl.drop(columns = ['STAID']),
            id_vars = df_train_expl['STAID'])

        # standardize data
        clust.stand_norm(method = 'standardize', # 'normalize'
            not_tr = not_tr_in) 

        # perform PCA on training data and plot
        clust.pca_reducer(
            nc = None, # None option includes all components
            color_in = train_ID, # 'blue'
            plot_out = True
        )

        
        # regression on transformed explanatory variables

        # define explanatory variables - subset to first 38 components
        # since they explain 95% of the variance in the explanatory variables
        # max_comp = int(input('based on those results, how many components do you want to consider? (e.g., 38): '))
        max_comp = clust.pca95
        expl_vars_in = clust.df_pca_embedding_.iloc[:, 0:max_comp]
        
        # define klim  (how many variables to consider in moder)
        klim_in = int(max_comp) # int(input('How many variables do you want to consider (0<klim<NC): '))

        # define model name and parameter name(s) to be written to files and used in file names
        param_name = f'forwardklim{klim_in}'

        # Instantiate a Regressor object 
        regr = Regressor(expl_vars = expl_vars_in,
            resp_var = train_resp)


        regr.lin_regression_select(
            sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
            float_opt = 'True', # 'True' or 'False'
            min_k = klim_in, # only active for 'exhaustive' option
            klim_in = klim_in, # controls max/min number of features for forward/backward selection
            timeseries = False, # if timeseries = True then NSE and KGE are also calculated
            n_jobs_in = -1) # number of cores to distribute to
                            
        # define variable holding the selected features and vifs.
        n_f_in = int(input('enter the number of features in the model you want to use (e.g., 41): '))
        vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == n_f_in, 'VIF']

        # Extract feature names for selecting features
        features_in = pd.DataFrame(dict((vif_in))).index

        # Subset appropriate explanatory variables to columns of interest
        # validation data from catchments used in training
        expl_in = regr.expl_vars[features_in]


        train_val = 'train'
        print(train_val)

        # define response variable
        resp_in = train_resp

        # define id vars
        id_in = train_ID
        # id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

        # OLS regression predict
        # specifiy input model
        mdl_in = LinearRegression().fit(
                    # df_train_mnexpl[features_in], df_train_mnanWY
                    expl_in, resp_in
                    )

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

        # # write vif results to csv
        df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
        df_vif = df_vif.rename(columns = {0: 'VIF'})
        df_vif.to_csv(
            f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
            index = True, 
            index_label = 'feature'
            )

        ##### 
        train_val = 'testin'
        print(train_val)
        # Apply to validation catchments used in training (i.e., testin)
        # standardize explantory vars, give columns names, and replace vars not to be transformed
        # define expl vars to work with
        df_in = df_testin_expl.drop(columns = 'STAID')
        expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
        expl_in.columns = df_in.columns
        expl_in[not_tr_in] = df_in[not_tr_in]

        # project explanatory variables into pca space and give new df column names
        expl_in_pcatr = pd.DataFrame(
            clust.pca_fit_.transform(expl_in)
        )
        # pca reduced data column names
        expl_in_pcatr.columns = [f'Comp{i}' for i in np.arange(0, expl_in_pcatr.shape[1], 1)]
        # subset to number of components used in model (e.g., 38)
        expl_in_pcatr = expl_in_pcatr[features_in]
        expl_in = expl_in_pcatr
        resp_in = testin_resp
        # resp_in = train_mnanWY_tr
        id_in = testin_ID

        # define response variable
        resp_in = testin_resp

        # define id vars
        id_in = testin_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = mdl_in

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        #####
        train_val = 'valnit'
        print(train_val)
        # Apply to validation catchments not used in training (i.e., valnit)
        # standardize explantory vars, give columns names, and replace vars not to be transformed
        # define expl vars to work with
        df_in = df_valnit_expl.drop(columns = 'STAID')
        expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
        expl_in.columns = df_in.columns
        expl_in[not_tr_in] = df_in[not_tr_in]

        # project explanatory variables into pca space and give new df column names
        expl_in_pcatr = pd.DataFrame(
            clust.pca_fit_.transform(expl_in)
        )
        # pca reduced data column names
        expl_in_pcatr.columns = [f'Comp{i}' for i in np.arange(0, expl_in_pcatr.shape[1], 1)]
        # subset to number of components used in model (e.g., 38)
        expl_in_pcatr = expl_in_pcatr[features_in]
        expl_in = expl_in_pcatr
        resp_in = valnit_resp
        # resp_in = train_mnanWY_tr
        id_in = valnit_ID

        # define response variable
        resp_in = valnit_resp

        # define id vars
        id_in = valnit_ID

        # OLS regression predict
        # specifiy input model
        mdl_in = mdl_in

        # Create predict-plot object
        regr.pred_plot(
            model_in = mdl_in,
            X_pred =  expl_in,
            y_obs = resp_in,
            id_vars = id_in
        )

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        # Write results to csv
        df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)













    ########################
    #########################
    # NEXT MODELING PATH
    ########################
    #########################













 # %%  XGBoost
    
    # print which model is being applied  
    print('XGBoost Regression')
    
    # model name for saving in csv
    model_name = 'XGBoost' # input('enter name for model (e.g., stdrd_PCA_mlr):') 

    if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
        (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # update model count
        mdl_count = mdl_count + 1

        
        # indicate if data is training, validating, testing, etc.
        train_val = 'train'

        regr = Regressor(expl_vars = df_train_expl.drop('STAID', axis = 1),
            resp_var = train_resp)


        regr.xgb_regression(
            n_splits_in = 10,
            n_repeats_in = 1,
            random_state_in = 100,
            grid_in = grid_in,
            timeseries = False,
            n_jobs_in = -1,
            dir_save = 'D:/Projects/GAGESii_ANNstuff/Python/Scripts/Learning_Results/xgbreg_classlearn_model.json'   
        )

        # example grid_in
        # grid_in = {
        #     'n_estimators': [100, 500], #  [100, 500], # [100, 250, 500], # [10], # 
        #     'colsample_bytree': [1], # [1], # [0.7, 1], 
        #     'max_depth': [6], # [6], # [4, 6, 8],
        #     'gamma': [0], # [0], # [0, 1], 
        #     'reg_lambda': [0], # [0], # [0, 1, 2]
        #     'learning_rate': [0.3], # [0.3] # [0.02, 0.1, 0.3]
        #     }

        expl_in = df_train_expl.drop('STAID', axis = 1)
        resp_in = train_resp
        id_in = train_ID

        # create predict/plot object
        regr.pred_plot(
            model_in = regr.xgb_reg_, 
            X_pred = expl_in, 
            y_obs = resp_in,
            id_vars = id_in)



        # define param_name to write to results indicated what parameter values were used
        nest = regr.xgboost_params_['n_estimators']
        colsmpl = regr.xgboost_params_['colsample_bytree']
        mdpth = regr.xgboost_params_['max_depth']
        gma = regr.xgboost_params_['gamma']
        lmbd = regr.xgboost_params_['reg_lambda']
        lrnrt = regr.xgboost_params_['learning_rate']  
        param_name = f'n_est{nest}_colsmpl{colsmpl}_mdpth{mdpth}_gma{gma}_lmbda{lmbd}_lrnrt{lrnrt}'

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


        ##########################


        # indicate if data is training, validating, testing, etc.
        train_val = 'testin'

        # define explanatory and response variables for modeling
        expl_in = df_testin_expl.drop('STAID', axis = 1)
        resp_in = testin_resp
        id_in = valnit_ID

        # create predict/plot object
        regr.pred_plot(
            model_in = regr.xgb_reg_, 
            X_pred = expl_in, 
            y_obs = resp_in,
            id_vars = id_in)

        # define param_name to write to results indicated what parameter values were used
        nest = regr.xgboost_params_['n_estimators']
        colsmpl = regr.xgboost_params_['colsample_bytree']
        mdpth = regr.xgboost_params_['max_depth']
        gma = regr.xgboost_params_['gamma']
        lmbd = regr.xgboost_params_['reg_lambda']
        lrnrt = regr.xgboost_params_['learning_rate']  
        param_name = f'n_est{nest}_colsmpl{colsmpl}_mdpth{mdpth}_gma{gma}_lmbda{lmbd}_lrnrt{lrnrt}'

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)



        ############################
        # indicate if data is training, validating, testing, etc.
        train_val = 'valnit'

        # define explanatory and response variables for modeling
        expl_in = df_valnit_expl.drop('STAID', axis = 1)
        resp_in = valnit_resp
        id_in = valnit_ID

        # create predict/plot object
        regr.pred_plot(
            model_in = regr.xgb_reg_, 
            X_pred = expl_in, 
            y_obs = resp_in,
            id_vars = id_in)

        # define param_name to write to results indicated what parameter values were used
        nest = regr.xgboost_params_['n_estimators']
        colsmpl = regr.xgboost_params_['colsample_bytree']
        mdpth = regr.xgboost_params_['max_depth']
        gma = regr.xgboost_params_['gamma']
        lmbd = regr.xgboost_params_['reg_lambda']
        lrnrt = regr.xgboost_params_['learning_rate']  
        param_name = f'n_est{nest}_colsmpl{colsmpl}_mdpth{mdpth}_gma{gma}_lmbda{lmbd}_lrnrt{lrnrt}'

        # append results to df_results_temp
        to_append = regr.df_pred_performance_.copy()
        # change VIF to max VIF instead of full array (full array saved to its own file for each model)
        to_append['VIF'] = to_append['VIF'][0].max()
        # include model
        to_append['model'] = model_name
        # include hyperparameters (tuning parameters)
        to_append['parameters'] = [param_name]
        # specify if results are from training data or validation (in) or validation (not it)
        to_append['train_val'] = train_val

        df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)




    # %% Append region/cluster label/name to df_results_temp
    df_results_temp.loc[(df_results_temp.shape[0] - 3 * mdl_count): df_results_temp.shape[0], 'region'] = reg_in
    # Append clustering method to df_results_temp
    df_results_temp.loc[(df_results_temp.shape[0] - 3 * mdl_count): df_results_temp.shape[0], 'clust_method'] = clust_meth

    # Write results to csv
    df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeries.csv', 
        index = False)

    # write to TEMP file to for investigation if needed
    df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', 
        index = False)


    print('------------Job complete------------')

    ######################################################################################



    








    # %%  Raw -> Rescale (Standardization) -> Reduce (UMAP) -> Predict (Lasso)
    ##############################

    # print('stdrd -> UMAP -> Lasso')

    # # model name for saving in csv
    # model_name = 'strd_UMAP_lasso' # input('enter name for model (e.g., stdrd_UMAP_lasso):') 

    # if (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
    #    (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # # update model count
        # mdl_count = mdl_count + 1

    #     # UMAP
    #     # define cluster/reducer object
    #     clust = Clusterer(clust_vars = df_train_expl.drop(columns = ['STAID']),
    #         id_vars = df_train_expl['STAID'])

    #     # standardize data
    #     clust.stand_norm(method = 'standardize', # 'normalize'
    #         not_tr = not_tr_in) 

    #     # apply UMAP on training data and plot

    #     # apply gridsearch cross validation to identify best parameters for nn and mind
    #     #########
    #     print('performing CV to investigate best UMAP parameters with alpha = 0.01')
    #     # %% Define models and pipeline
    #     # define standard scaler
    #     scaler = StandardScaler()

    #     # define umap object
    #     umap_reducer = umap.UMAP(
    #                     n_components = 80,
    #                     spread = 1, 
    #                     random_state = 100, 
    #                     n_epochs = 200, 
    #                     n_jobs = -1)

    #     # define lasso object
    #     lasso_mod = Lasso(max_iter = 1000)

    #     # define pipline
    #     pipe = Pipeline(steps = [('scaler', scaler), ('umap_reducer', umap_reducer), ('lasso_mod', lasso_mod)])

    #     # # %% Define parameter grid and gridsearch (exhaustive)

    #     # parameters for the pipeline search
    #     param_grid = {
    #         'umap_reducer__n_neighbors': [2, 20, 40, 60, 80, 100], # for initial run do 6 for each
    #         'umap_reducer__min_dist': np.arange(0.01, 1.1, 0.2),
    #         'lasso_mod__alpha': [0.01] # np.arange(0.01, 1.01, 0.01)
    #     }

    #     search = GridSearchCV(pipe, param_grid, n_jobs = -1)

    #     # Apply gridsearch to data and return best fit parameters

    #     search.fit(df_train_expl.drop(columns = 'STAID'), train_resp)
        
    #     print(search.best_params_)

    #     # print results
    #     # print(pd.DataFrame(search.cv_results_))

    #     ########


    #     # define numer of components for n_neighbors
    #     nc_in = 80 # 10
    #     nn_in = int(input('enter a value for n_neighbors (e.g., 80): ')) 
    #     mind_in = float(input('enter a value for minimum distance between points (e.g., 0.81): '))

    #     clust.umap_reducer(
    #         nn = nn_in, # 10
    #         mind = mind_in,
    #         sprd = 1, # dont' use for tuning - leave set to 1
    #         nc = nc_in,
    #         color_in = train_ID
    #     )

    #     # regression on transformed explanatory variables

    #     # define explanatory variables - subset df_embedding to only embedding columns
    #     expl_vars_in = clust.df_embedding_.iloc[:, 3:nc_in + 3]

    #     # Instantiate a Regressor object 
    #     regr = Regressor(expl_vars = expl_vars_in,
    #         resp_var = train_resp)





    #     # %%
    #     print('performing CV with UMAP values identified above, and varying alpha')
    #     #### Lasso regression
    #     # search via cross validation
    #     regr.lasso_regression(
    #         alpha_in = list(np.arange(0.01, 1.01, 0.01)), # must be single integer or list
    #         max_iter_in = 1000,
    #         n_splits_in = 10,
    #         n_repeats_in = 1,
    #         random_state_in = 100,
    #         n_jobs_in = -1
    #     )
    #     # %%
    #     # print all results from CV
    #     print(pd.DataFrame(regr.lassoCV_results.cv_results_))
    #     # %%
    #     # print('top 10 results from cross-validation for mae, rmse, and r2')
    #     # # %%
    #     # print(regr.df_lassoCV_mae[0:10])
    #     # # %%
    #     # print(regr.df_lassoCV_rmse[0:10])
    #     # # %%
    #     # print(regr.df_lassoCV_r2[0:10])

    #     # %%

    #     ##### apply best alpha identified in cross validation
    #     train_val = 'train'
    #     print(train_val)

    #     # define alpha
    #     a_in = float(input('based on those results, what value do you want to use for alpha?'))

        
        
    #     ##### apply best alpha identified in cross validation
    #     # Lasso with chosen alpha

    #     # define model name and parameter name(s) to be written to files and used in file names
    #     param_name = f'nn{nn_in}mind{mind_in}nc{nc_in}alpha{a_in}'

    #     # # define model name and parameter name(s) to be written to files and used in file names
    #     # model_name = 'stdrd_UMAP_lasso'
    #     # param_name = 'nn80mind0.81nc{nc_in}alpha0.01{a_in}'

    #     # Lasso with alpha = 0.01
    #     regr.lasso_regression(
    #         alpha_in = float(a_in), # must be single integer or list
    #         max_iter_in = 1000,
    #         n_splits_in = 10,
    #         n_repeats_in = 1,
    #         random_state_in = 100
    #     )

    #     # plot regression
    #     mdl_in = regr.lasso_reg_
    #     expl_in = regr.expl_vars
    #     resp_in = train_resp
    #     # resp_in = train_mnanWY_tr
    #     id_in = train_ID

    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )

    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy() # NOTE: copy so original df is not edited in place
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

    #     # write vif results to csv
    #     df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
    #     df_vif = df_vif.rename(columns = {0: 'VIF'})

    #     # Max VIF is 1, so no need to write to csv
    #     df_vif.to_csv(
    #         f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
    #         index = True, 
    #         index_label = 'feature'
    #         )

    #     #####

    #     # %%
    #     # return prediction metrics for validation data from stations that were trained on
    #     # plot regression

    #     train_val = 'testin'
    #     print(train_val)

    #     # define model in
    #     mdl_in = mdl_in
    #     # standardize explantory vars, give columns names, and replace vars not to be transformed
    #     # define expl vars to work with
    #     df_in = df_testin_expl.drop(columns = 'STAID')
    #     expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
    #     expl_in.columns = df_in.columns
    #     expl_in[not_tr_in] = df_in[not_tr_in]

    #     # project explanatory variables into umap space and give new df column names
    #     expl_in_umaptr = pd.DataFrame(
    #         clust.umap_embedding_.transform(expl_in)
    #     )
    #     # umap reduced data column names
    #     expl_in_umaptr.columns = [f'Emb{i}' for i in np.arange(0, expl_in_umaptr.shape[1], 1)]
    #     expl_in = expl_in_umaptr
    #     resp_in = testin_resp
    #     # resp_in = train_mnanWY_tr
    #     id_in = testin_ID

    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )


    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy()
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name # 'stdrd_lasso'
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

    #     #####
    #     # return prediction metrics for validation data from stations that were NOT trained on
    #     # plot regression

    #     train_val = 'valnit'
    #     print(train_val)

    #     # define model in
    #     mdl_in = mdl_in
    #     # standardize explantory vars, give columns names, and replace vars not to be transformed
    #     # define expl vars to work with
    #     df_in = df_valnit_expl.drop(columns = 'STAID')
    #     expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
    #     expl_in.columns = df_in.columns
    #     expl_in[not_tr_in] = df_in[not_tr_in]

    #     # project explanatory variables into umap space and give new df column names
    #     expl_in_umaptr = pd.DataFrame(
    #         clust.umap_embedding_.transform(expl_in)
    #     )
    #     # umap reduced data column names
    #     expl_in_umaptr.columns = [f'Emb{i}' for i in np.arange(0, expl_in_umaptr.shape[1], 1)]
    #     expl_in = expl_in_umaptr
    #     resp_in = valnit_resp
    #     # resp_in = train_mnanWY_tr
    #     id_in = valnit_ID


    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )

    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy()
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name # 'raw_lasso'
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

    #     # Write results to csv
    #     df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)


 # %%  Raw -> Rescale (Standardization) -> Reduce (umap) -> Predict (OLS-MLR)
    # # 
    # # MLR feature selection - 'forward'

    # print('stdrd -> UMAP -> MLR')
    # # define model name and parameter name(s) to be written to files and used in file names

    # # model name for saving in csv
    # model_name = 'strd_UMAP_mlr' # input('enter name for model (e.g., stdrd_UMAP_mlr):') 

    # (not any(df_results_temp.loc[(df_results_temp['model'] == model_name) &
    #    (df_results_temp['clust_method'] == clust_meth), 'region'] == reg_in)):

        # # update model count
        # mdl_count = mdl_count + 1

    #     klim_in = nc_in # int(input('input the maximum number of variables you want to consider (0<klim<NC): '))
    #     param_name = f'nn{nn_in}mind{mind_in}nc{nc_in}forwardklim{klim_in}'

    #     # model_name = 'stdrd_UMAP_mlr'
    #     # param_name = 'forward_klim80'

    #     regr.lin_regression_select(
    #         sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
    #         float_opt = 'True', # 'True' or 'False'
    #         min_k = 1, # only active for 'exhaustive' option
    #         klim_in = klim_in, # controls max/min number of features for forward/backward selection
    #         timeseries = False, # if timeseries = True then NSE and KGE are also calculated
    #         n_jobs_in = -1) # number of cores to distribute to



    #     # print smallest 5 BIC models
    #     print(regr.df_lin_regr_performance_.sort_values(by = 'BIC')[0:5])


    #     # define variable holding the selected features and vifs.
    #     n_f_in = int(input('enter the number of features in the model you want to use (e.g., 41): '))
    #     vif_in = regr.df_lin_regr_performance_.loc[regr.df_lin_regr_performance_['n_features'] == n_f_in, 'VIF']

    #     # Extract feature names for selecting features
    #     features_in = pd.DataFrame(dict((vif_in))).index

    #     # %%
        
    #     train_val = 'train'
    #     print(train_val)
        
    #     # Subset appropriate explanatory variables to columns of interest
    #     # validation data from catchments used in training
    #     expl_in = regr.expl_vars[features_in]

    #     # define response variable
    #     resp_in = train_resp

    #     # define id vars
    #     id_in = train_ID
    #     # id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

    #     # OLS regression predict
    #     # specifiy input model
    #     mdl_in = LinearRegression().fit(
    #                 # df_train_expl[features_in], train_resp
    #                 expl_in, resp_in
    #                 )

    #     # Create predict-plot object
    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )

    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy()
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)

    #     # write vif results to csv
    #     df_vif = pd.DataFrame(dict(regr.df_pred_performance_['VIF']))
    #     df_vif = df_vif.rename(columns = {0: 'VIF'})
    #     df_vif.to_csv(
    #         f'{dir_expl}/Results/VIF_dfs/{clust_meth}_{reg_in}_{model_name}_{param_name}_VIF.csv',
    #         index = True, 
    #         index_label = 'feature'
    #         )

    #     ##### 
    #     # %%
    #     # Apply to validation catchments used in training (i.e., testin)

    #     train_val = 'testin'
    #     print(train_val)

    #     # standardize explantory vars, give columns names, and replace vars not to be transformed
    #     # define expl vars to work with
    #     df_in = df_testin_expl.drop(columns = 'STAID')
    #     expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
    #     expl_in.columns = df_in.columns
    #     expl_in[not_tr_in] = df_in[not_tr_in]

    #     # project explanatory variables into umap space and give new df column names
    #     expl_in_umaptr = pd.DataFrame(
    #         clust.umap_embedding_.transform(expl_in)
    #     )
    #     # umap reduced data column names
    #     expl_in_umaptr.columns = [f'Emb{i}' for i in np.arange(0, expl_in_umaptr.shape[1], 1)]
    #     expl_in_umaptr = expl_in_umaptr[features_in]
    #     expl_in = expl_in_umaptr
    #     resp_in = testin_resp
    #     # resp_in = train_mnanWY_tr
    #     id_in = testin_ID

    #     # define response variable
    #     resp_in = testin_resp

    #     # define id vars
    #     id_in = testin_ID

    #     # OLS regression predict
    #     # specifiy input model
    #     mdl_in = mdl_in

    #     # Create predict-plot object
    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )

    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy()
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


    #     #####
    #     # Apply to validation catchments not used in training (i.e., valnit)

    #     train_val = 'valnit'
    #     print(train_val)

    #     # standardize explantory vars, give columns names, and replace vars not to be transformed
    #     # define expl vars to work with
    #     df_in = df_valnit_expl.drop(columns = 'STAID')
    #     expl_in = pd.DataFrame(clust.scaler_.transform(df_in))
    #     expl_in.columns = df_in.columns
    #     expl_in[not_tr_in] = df_in[not_tr_in]

    #     # project explanatory variables into umap space and give new df column names
    #     expl_in_umaptr = pd.DataFrame(
    #         clust.umap_embedding_.transform(expl_in)
    #     )
    #     # umap reduced data column names
    #     expl_in_umaptr.columns = [f'Emb{i}' for i in np.arange(0, expl_in_umaptr.shape[1], 1)]
    #     expl_in_umaptr = expl_in_umaptr[features_in]
    #     expl_in = expl_in_umaptr
    #     resp_in = valnit_resp
    #     # resp_in = train_mnanWY_tr
    #     id_in = valnit_ID

    #     # define response variable
    #     resp_in = valnit_resp

    #     # define id vars
    #     id_in = valnit_ID

    #     # OLS regression predict
    #     # specifiy input model
    #     mdl_in = mdl_in

    #     # Create predict-plot object
    #     regr.pred_plot(
    #         model_in = mdl_in,
    #         X_pred =  expl_in,
    #         y_obs = resp_in,
    #         id_vars = id_in
    #     )

    #     # append results to df_results_temp
    #     to_append = regr.df_pred_performance_.copy()
    #     # change VIF to max VIF instead of full array (full array saved to its own file for each model)
    #     to_append['VIF'] = to_append['VIF'][0].max()
    #     # include model
    #     to_append['model'] = model_name
    #     # include hyperparameters (tuning parameters)
    #     to_append['parameters'] = [param_name]
    #     # specify if results are from training data or validation (in) or validation (not it)
    #     to_append['train_val'] = train_val

    #     df_results_temp = pd.concat([df_results_temp, to_append], ignore_index = True)


    #     # Write results to csv
    #     df_results_temp.to_csv(f'{dir_expl}/Results/Results_NonTimeSeriesTEMP.csv', index = False)