# %% Import classes and libraries
# from statistics import LinearRegression
from GAGESii_Class import Clusterer
from GAGESii_Class import Regressor
from Regression_PerformanceMetrics_Functs import *
import pandas as pd
import plotnine as p9
import plotly.express as px # easier interactive plots
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression

# %% load data


# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'

# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned'

# directory to write csv holding removed columns (due to high VIF)
dir_VIF = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/VIF_Removed'

# GAGESii explanatory vars
# training
df_train_expl = pd.read_csv(
    f'{dir_expl}/Expl_train.csv',
    dtype = {'STAID': 'string'}
)
# test_in
df_testin_expl = pd.read_csv(
    f'{dir_expl}/Expl_testin.csv',
    dtype = {'STAID': 'string'}
)
# val_nit
df_valnit_expl = pd.read_csv(
    f'{dir_expl}/Expl_valnit.csv',
    dtype = {'STAID': 'string'}
)


# Water yield variables
# Annual Water yield
# training
df_train_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_train.csv',
    dtype = {"site_no":"string"}
    )

# val_in
df_testin_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_testin.csv',
    dtype = {"site_no":"string"}
    )

# val_nit
df_valnit_anWY = pd.read_csv(
    f'{dir_WY}/annual/WY_Ann_valnit.csv',
    dtype = {"site_no":"string"}
    )

# mean annual water yield
# training
df_train_mnanWY = df_train_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])
# val_in
df_testin_mnanWY = df_testin_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])
# val_nit
df_valnit_mnanWY = df_valnit_anWY.groupby(
    'site_no', as_index = False
).mean().drop(columns = ["yr"])

# mean GAGESii explanatory vars
# training
df_train_mnexpl = df_train_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])
# val_in
df_testin_mnexpl = df_testin_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])
#val_nit
df_valnit_mnexpl = df_valnit_expl.groupby(
    'STAID', as_index = False
).mean().drop(columns = ['year'])

# ID vars (e.g., ecoregion)

# training ID
df_train_ID = pd.read_csv(f'{dir_expl}/ID_train.csv',
    dtype = {'STAID': 'string'})
# val_in ID
df_testin_ID = df_train_ID
# val_nit ID
df_valnit_ID = pd.read_csv(f'{dir_expl}/ID_valnit.csv',
    dtype = {'STAID': 'string'})

# Read in categories to be used in specified hierarchical clustering
df_cats = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/FeatureCategories.csv',
    usecols = ['Custom_Cat', 'Features', 'Category', 'Coarse_Cat', 'Coarsest_Cat']
).drop([0, 82], axis = 0)

del(df_train_anWY, df_train_expl, df_testin_anWY, df_testin_expl, df_valnit_anWY, df_valnit_expl)

#############


# %%
# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']

test = Clusterer(clust_vars = df_train_mnexpl.drop(columns = ['STAID']),
    id_vars = df_train_mnexpl['STAID'])

# note that once input data is transformed, the transformed
# version will be used automatically in all functions related to
# Clusterer object
test.stand_norm(method = 'standardize', # 'normalize'
    not_tr = not_tr_in)

test.k_clust(
    ki = 2, kf = 20, 
    method = 'kmeans', 
    plot_mean_sil = True, 
    plot_distortion = True)

#####
# Based on results from previous chunk, chose k = 10
# Here defining the test object to have a k-means model of k = 10 for projecting new data
test.k_clust(
    ki = 10, kf = 10, 
    method = 'kmeans', 
    plot_mean_sil = False, 
    plot_distortion = False)

#######
# data to project into k-means (or other clustering) space
df_valnit_trnsfrmd = pd.DataFrame(test.scaler_.transform(df_valnit_mnexpl.drop(
    columns = ['STAID'])
    ))
# give column names to transformed dataframe
df_valnit_trnsfrmd.columns = df_valnit_mnexpl.drop(
    columns = ['STAID']
).columns

# replace ohe columns with untransformed data
df_valnit_trnsfrmd[not_tr_in] = df_valnit_mnexpl[not_tr_in]
#########
# get predicted k's for each catchment           
km_valnit_pred = pd.DataFrame(
    {'STAID': df_valnit_mnexpl['STAID'],
    'K': test.km_clusterer_.predict(
            df_valnit_trnsfrmd
            )
    }
)
##########
# K-medoids

test.k_clust(
    ki = 2, kf = 30, 
    method = 'kmedoids', 
    plot_mean_sil = True, 
    plot_distortion = True,
    kmed_method = 'alternate')

for i in range(19, 23):
    test.plot_silhouette_vals(k = i)
#####
# Based on results from previous chunk, chose k = 8
# Here defining the test object to have a k-means model of k = 8 for projecting new data
test.k_clust(
    ki = 8, kf = 8, 
    method = 'kmedoids', 
    plot_mean_sil = False, 
    plot_distortion = False)

# get predicted k's for each catchment           
km_valnit_pred = test.km_clusterer_.predict(
    df_valnit_trnsfrmd
    )
# predicted clusters dataframe
df_km_valnit_pred = pd.DataFrame({
    'STAID': df_valnit_mnexpl['STAID'],
    'K_predicted': km_valnit_pred
})

########
#fuzzy c-means
test.fuzzy_cm_clusterer(
    c_i = 2, c_f = 30,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)

# based on previous results use 2 clusters
test.fuzzy_cm_clusterer(
    c_i = 5, c_f = 5,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)

# note that the test object now has an attribute fcm_centers which holds
# cluster centers to be used in prediction

# predict the clusters for the valnit data
# first standardize it
df_in = df_valnit_mnexpl.drop(columns = 'STAID')
df_in = pd.DataFrame(test.scaler_.transform(df_in))
test.fuzzy_cm_predictor(
    data_pred = df_in,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)


#####
# PCA

# perform PCA on training data and plot
test.pca_reducer(
    nc = None,
    color_in = df_train_ID['AggEcoregion'], # 'blue'
    plot_out = True
)

# project valnit stations into PCA space
# first standardize it 
df_in = df_valnit_mnexpl.drop(columns = 'STAID')
df_in = pd.DataFrame(test.scaler_.transform(df_in))
# give column names
df_in.columns = df_valnit_mnexpl.drop(columns = 'STAID').columns
df_in[not_tr_in] = test.expl_vars[not_tr_in]
# Replace columns not to be transformed

df_valnit_pcatr = pd.DataFrame(
                    test.pca_fit_.transform(df_in)
)
# give column names
df_valnit_pcatr.columns = [f'Comp{i}' for i in np.arange(0, df_valnit_pcatr.shape[1], 1)]

# plot projected data
# add color, colsize, and id column to dataframe to plotted
df_valnit_pcatr['Color'] = df_valnit_ID['AggEcoregion']
df_valnit_pcatr['ColSize'] = 0.1
df_valnit_pcatr['ID'] = df_valnit_mnexpl['STAID']
# Plot embedding
fig = px.scatter_3d(df_valnit_pcatr,
                    x = 'Comp0',
                    y = 'Comp1',
                    z = 'Comp2',
                    color = 'Color',
                    size = 'ColSize',
                    size_max = 10,
                    title = 'PCA',
                    custom_data = ['ID']
                    )
# Edit so hover shows station id
fig.update_traces(
hovertemplate = "<br>".join([
    "STAID: %{customdata[0]}"
])
)

# print(fig.show())
#####
# hdbscan

test.hdbscanner(
    min_clustsize = 10,
    gm_spantree = True,
    metric_in = 'manhattan', # 'euclidean' or 'manhattan', #
    clustsel_meth = 'eom') # 'eom' or 'leaf') #

# predict clusters for valin data
# assigns output dataframe named df_hd_pred_ to object
test.hd_predictor(
    points_to_predict = df_valnit_trnsfrmd,
    id_vars = df_valnit_mnexpl['STAID']
)

# print output dataframe
test.df_hd_pred_

test.umap_reducer(
    nn = 30, # 10,
    mind = 0, # 0.01,
    sprd = 1, # don't use for tuning - leave set to 1
    nc = 3,
    color_in = df_train_ID['AggEcoregion'])

# project valnit data into the umap space
df_umap_valnit_pred = pd.DataFrame(
    test.umap_embedding_.transform(df_valnit_trnsfrmd)
)
df_umap_valnit_pred.columns = [f'Emb{i}' for i in range(0, df_umap_valnit_pred.shape[1])]

df_umap_valnit_pred['STAID'] = df_valin_mnexpl['STAID']
df_umap_valnit_pred['Color'] = df_valin_ID['AggEcoregion']
df_umap_valnit_pred['ColSize'] = 0.1

# plot projected embeddings
fig = px.scatter_3d(df_umap_valnit_pred,
                    x = 'Emb0',
                    y = 'Emb1',
                    z = 'Emb2',
                    color = 'Color',
                    size = 'ColSize',
                    size_max = 10,
                    title = f"nn: 20",
                    custom_data = ["STAID"]
                    )
# Edit so hover shows station id
fig.update_traces(
    hovertemplate = "<br>".join([
    "STAID: %{customdata[0]}"
]))


##### 

# new clustering on umap reduced data
test2 = Clusterer(clust_vars = test.df_embedding_.drop(columns = ['STAID', 'Color', 'ColSize']),
    id_vars = df_train_mnexpl['STAID'])

test2.k_clust(ki = 2, kf = 15, 
    method = 'kmeans', 
    plot_mean_sil = True, 
    plot_distortion = True)

for i in range(2, 15):
    test2.plot_silhouette_vals(k = i)
    
test2.k_clust(
    ki = 2, kf = 15, 
    method = 'kmedoids', 
    plot_mean_sil = True, 
    plot_distortion = True,
    kmed_method = 'alternate') #'pam'

for i in range(2, 10):
    test2.plot_silhouette_vals(k = i,
        method = 'kmedoids',
        kmed_method = 'alternate')

test2.hdbscanner(
    min_clustsize = 50, # 50 resulted in 0 -1's but only three clusters (when nc = 10 in umap)
    gm_spantree = True,
    metric_in = 'euclidean', #'manhattan', #
    clustsel_meth = 'eom') #'leaf') #

#########
# predict hdbscan clusters for valnit data and plot in umap embedding space
# project valnit data into the umap space
# predict clusters for valnit gauges
test2.hd_predictor(
    points_to_predict = df_umap_valnit_pred.drop(columns = ['STAID', 'Color', 'ColSize']),
    id_vars = df_valnit_mnexpl['STAID']
)

# plot barplot of number of catchments in clusters predicted
(
    p9.ggplot(data = test2.df_hd_pred_) +
    p9.geom_bar(p9.aes(x = 'pred_cluster'))
)

# change color for umap plot
df_umap_valnit_pred['Color'] = test2.df_hd_pred_['pred_cluster'].astype('category')
# df_umap_valnit_pred['Color'] = df_valnit_ID['AggEcoregion']

# plot projected embeddings
fig = px.scatter_3d(df_umap_valnit_pred,
                    x = 'Emb0',
                    y = 'Emb1',
                    z = 'Emb4',
                    color = 'Color',
                    size = 'ColSize',
                    size_max = 10,
                    title = f"nn: 20",
                    custom_data = ["STAID"]
                    )
# Edit so hover shows station id
fig.update_traces(
hovertemplate = "<br>".join([
    "STAID: %{customdata[0]}"
]))

# plot trained embedding with hdbscan clusters as colors
tmp_datain = test.df_embedding_
tmp_datain['hd_clust'] = test2.hd_out_['labels'].astype('category')

fig = px.scatter_3d(tmp_datain,
                    x = 'Emb0',
                    y = 'Emb1',
                    z = 'Emb2',
                    color = 'hd_clust',
                    size = 'ColSize',
                    size_max = 10,
                    title = f"nn: 20",
                    custom_data = ["STAID"]
                    )
# Edit so hover shows station id
fig.update_traces(
hovertemplate = "<br>".join([
    "STAID: %{customdata[0]}"
]))
########
# fuzzy c-means on umap transformed data
test2.fuzzy_cm_clusterer(
    c_i = 2, c_f = 15,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)

# based on previous results use best number of clusters
test2.fuzzy_cm_clusterer(
    c_i = 2, c_f = 2,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)

# note that the test object now has an attribute fcm_centers which holds
# cluster centers to be used in prediction

# predict the clusters for the valnit data
df_in = df_umap_valnit_pred.iloc[:, 0:3]

test2.fuzzy_cm_predictor(
    data_pred = df_in,
    m_in = 2,
    error_in = 0.005,
    maxiter_in = 1000,
    init_in = None,
    seed_in = 100
)




# %% Apply regressor class methods
not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']
            # ,
            # 'TS_NLCD_11', 'TS_NLCD_12', 'TS_NLCD_21', 'TS_NLCD_22', 'TS_NLCD_23',
            # 'TS_NLCD_24', 'TS_NLCD_31', 'TS_NLCD_41', 'TS_NLCD_42', 'TS_NLCD_43',
            # 'TS_NLCD_52', 'TS_NLCD_71', 'TS_NLCD_81', 'TS_NLCD_82', 'TS_NLCD_90',
            # 'TS_NLCD_95', 'TS_NLCD_AG_SUM', 'TS_NLCD_DEV_SUM', 'TS_NLCD_imperv',
            # 'TS_ag_hcrop', 'TS_ag_irrig']

# regression on untransformed explanatory variables
# Instantiate a Regressor object 
testreg = Regressor(expl_vars = df_train_mnexpl.drop(columns = ['STAID']),
    resp_var = df_train_mnanWY['Ann_WY_ft'])

# apply lasso regression
testreg.lasso_regression(
    alpha_in = 1,
    max_iter_in = 1000,
    n_splits_in = 10,
    n_repeats_in = 3,
    random_state_in = 100
)

# print perfromance metrics again
testreg.df_lasso_regr_performance_

# print output metrics available from model
testreg.lasso_scores_.keys()

# print features kept by Lasso regression and the associated coefficients
testreg.df_lasso_features_coef_.sort_values(by = 'coefficients')

# apply Lasso model to gages not used in training
pred_lasso_nit = testreg.lasso_reg.predict(df_valnit_mnexpl.drop(columns = 'STAID'))
pred_lasso_nit_mae = sum(np.abs(pred_lasso_nit - df_valnit_mnanWY['Ann_WY_ft']))/len(pred_lasso_nit)
pred_lasso_nit_rmse = np.sqrt(sum((pred_lasso_nit - df_valnit_mnanWY['Ann_WY_ft'])**2)/len(pred_lasso_nit))

# apply Lasso model to gages used in training
pred_lasso_in = testreg.lasso_reg_.predict(df_valin_mnexpl.drop(columns = 'STAID'))
pred_lasso_in_mae = sum(np.abs(pred_lasso_in - df_valin_mnanWY['Ann_WY_ft']))/len(pred_lasso_in)
pred_lasso_in_rmse = np.sqrt(sum((pred_lasso_in - df_valin_mnanWY['Ann_WY_ft'])**2)/len(pred_lasso_in))


#######
# multiple linear regression
# lin_regression_select produces plots and a pandas DataFrame with performance metrics
# and features from models with various numbers of features
testreg.lin_regression_select(
    sel_meth = 'forward', # 'forward', 'backward', or 'exhaustive'
    float_opt = 'True', # 'True' or 'False'
    min_k = 18, # only active for 'exhaustive' option
    klim_in = 40, # controls max/min number of features for forward/backward selection
    timeseries = False) # if timeseries = True then NSE and KGE are also calculated

# print performance metric dataframe subset to n_features of desired number
testreg.df_lin_regr_performance_.loc[testreg.df_lin_regr_performance_['n_features'] == 20,]
# define variable holding the selected features and vifs.
vif_in = testreg.df_lin_regr_performance_.loc[testreg.df_lin_regr_performance_['n_features'] == 20, 'VIF']

# Extract feature names for selecting features
features_in = pd.DataFrame(dict((vif_in))).index

# Note, if using Lasso model, then expects all variables/features to be supplied, 
# It simply applies coefficients of 0 to those variables not included

# Subset appropriate explanatory variables to columns of interest
# validation data from catchments used in training
expl_in = df_valnit_mnexpl[features_in]
# for lasso model
# expl_in = df_valnit_mnexpl.drop(columns = 'STAID')
# expl_in = df_train_mnexpl['PPTAVG_BASIN'].values.reshape(-1, 1)
# expl_in = df_valnit_mnexpl['PPTAVG_BASIN'].values.reshape(-1, 1)

# define response variable
resp_in = df_valnit_mnanWY['Ann_WY_ft']

# define id vars
# id_in = df_valnit_ID['AggEcoregion']
id_in = pd.Series(test2.df_hd_pred_['pred_cluster'], dtype = 'category')

######## 
# OLS regression predict
# specifiy input model
mdl_in = LinearRegression().fit(
            # df_train_mnexpl[features_in], df_train_mnanWY['Ann_WY_ft']
            expl_in, resp_in
            )
# model in from lasso
# mdl_in = testreg.lasso_reg_

# Create predict-plot object
testreg.pred_plot(
    model_in = mdl_in,
    X_pred =  expl_in,
    y_obs = resp_in,
    id_vars = id_in
)

# print performance metrics again
testreg.df_pred_performance_
# print features and VIF
dict(testreg.df_pred_performance_['VIF'])



# %% Check assumptions of linear regerssion and explore transformations
# plot histogram of flow data
dist_flow = (
    p9.ggplot(data = df_train_mnanWY) +
    p9.geom_histogram(p9.aes(x = 'Ann_WY_ft'))
)

dist_flow

# apply box-cox transformation
df_train_mnanWY_tr, best_lambda = stats.boxcox(df_train_mnanWY['Ann_WY_ft'])

dist_flow_tr = (
    p9.ggplot() +
    p9.geom_histogram(p9.aes(x = df_train_mnanWY_tr))
)

dist_flow_tr








# %% XGBOOST

# not_tr_in = ['GEOL_REEDBUSH_DOM_gneiss', 'GEOL_REEDBUSH_DOM_granitic', 
#             'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
#             'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']

# regression on untransformed explanatory variables
# Instantiate a Regressor object 
testreg = Regressor(expl_vars = df_train_mnexpl.drop(columns = ['STAID']),
    resp_var = df_train_mnanWY['Ann_WY_ft'])

testreg.xgb_regression(
    n_splits_in = 10,
    n_repeats_in = 1,
    random_state_in = 100,
    grid_in = {
     'n_estimators': [100, 500], #  [100, 500], # [100, 250, 500], # [10], # 
     'colsample_bytree': [1], # [1], # [0.7, 1], 
     'max_depth': [6], # [6], # [4, 6, 8],
     'gamma': [0], # [0], # [0, 1], 
     'reg_lambda': [0], # [0], # [0, 1, 2]
     'learning_rate': [0.3], # [0.3] # [0.02, 0.1, 0.3]
     },
    timeseries = False,
    n_jobs_in = -1,
    dir_save = 'D:/Projects/GAGESii_ANNstuff/Python/Scripts/Learning_Results/xgbreg_classlearn_model.json'   
)

# predict train WY values
testreg.pred_plot(
    model_in = testreg.xgb_reg_, 
    X_pred = df_train_mnexpl.drop('STAID', axis = 1), 
    y_obs = df_train_mnanWY['Ann_WY_ft'],
    id_vars = df_train_ID['AggEcoregion'])

# predict testin WY values
testreg.pred_plot(
    model_in = testreg.xgb_reg_, 
    X_pred = df_testin_mnexpl.drop('STAID', axis = 1), 
    y_obs = df_testin_mnanWY['Ann_WY_ft'],
    id_vars = df_testin_ID['AggEcoregion'])

# predict valnit WY values
testreg.pred_plot(
    model_in = testreg.xgb_reg_, 
    X_pred = df_valnit_mnexpl.drop('STAID', axis = 1), 
    y_obs = df_valnit_mnanWY['Ann_WY_ft'],
    id_vars = df_valnit_ID['AggEcoregion'])


# predict testnit WY values
ypred_testin = testreg.xgb_reg_.predict(df_testin_mnexpl.drop('STAID', axis = 1))
# predict valnit WY values
ypred_valnit = testreg.xgb_reg_.predict(df_valnit_mnexpl.drop('STAID', axis = 1))


# %%
