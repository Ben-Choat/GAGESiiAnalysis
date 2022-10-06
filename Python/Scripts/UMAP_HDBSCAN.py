# Ben Choat 7/28/2022

# Script to perform specified hiearchical clustering
# perform UMAP followed by HDBSCAN


# %% import libraries and classes

from GAGESii_Class import Clusterer
from GAGESii_MeanAnnual_Callable import *
# from GAGESii_Class import Regressor
# from Regression_PerformanceMetrics_Functs import *
import pandas as pd
import os
# import plotnine as p9
# import plotly.express as px # easier interactive plots
# from scipy import stats
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error

# %% load data

dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work'

# water yield directory
# dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
dir_WY = f'{dir_work}/data_work/USGS_discharge/annual'

# explantory var (and other data) directory
dir_expl = f'{dir_work}/data_work/GAGESiiVariables'
# 'D:/Projects/GAGESii_ANNstuff/AllVars_Partitioned'

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

# %%

# define place holder dataframes
cidtrain_in = df_train_ID
cidtestin_in = df_testin_ID
cidvalnit_in = df_valnit_ID


# Water yield
train_resp_in = pd.merge(
    df_train_mnanWY, cidtrain_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
testin_resp_in = pd.merge(
    df_testin_mnanWY, cidtestin_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
valnit_resp_in = pd.merge(
    df_valnit_mnanWY, cidvalnit_in, left_on = 'site_no', right_on = 'STAID'
    )['Ann_WY_ft']
# explanatory variables
train_expl_in = pd.merge(df_train_mnexpl, cidtrain_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
testin_expl_in = pd.merge(df_testin_mnexpl, cidtestin_in, left_on = 'STAID', right_on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
valnit_expl_in = pd.merge(df_valnit_mnexpl, cidvalnit_in, on = 'STAID').drop(
    columns = ['Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site',
                'DRAIN_SQKM_y', 'LAT_GAGE', 'LNG_GAGE', 'HUC02']
)
# ID dataframes
train_ID_in = pd.merge(
    df_train_ID, cidtrain_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
testin_ID_in = pd.merge(
    df_testin_ID, cidtestin_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
valnit_ID_in = pd.merge(
    df_valnit_ID, cidvalnit_in, on = ['STAID', 'Class', 'AggEcoregion', 'ECO3_Site', 'USDA_LRR_Site']
    )['AggEcoregion'] # ['ECO3_Site']
##########



# %% ###################
# UMAP followed by HDBSCAN
########################

# define dataframe to add labels to (i.e., to hold )
df_labels = pd.DataFrame({
    'STAID': df_train_mnexpl['STAID']
})

#############
# Clustering using all variables at once
############

# Standardize data

# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']
            # 'GEOL_REEDBUSH_DOM_gneiss', 


# define vars to use in clustering
clust_vars_in = train_expl_in.drop('STAID', axis = 1)
# define id vars
id_vars_in = train_expl_in['STAID']

# define clusterer object
cl_obj = Clusterer(clust_vars = clust_vars_in,
    id_vars = id_vars_in)

# note that once input data is transformed, the transformed
# version will be used automatically in all functions related to
# Clusterer object
cl_obj.stand_norm(method = 'standardize',
    not_tr = not_tr_in), # 'normalize'
    

# try several values for min_clustsize

# HDBSCAN without UMAP
# HDBSCAN on transformed data
for mcs in [30, 40, 50, 75, 100, 200, 300]:
    for ms in [1, 20, 50]:
        print(f'min clustsize: {mcs}')
        print(f'min sample: {ms}')
        cl_obj.hdbscanner(
            min_clustsize = mcs, # 35,
            min_sample = ms, # 1, # None,
            gm_spantree = False,
            metric_in =  'euclidean', # 'manhattan', #
            clust_seleps = 0, # 0.5,
            clustsel_meth = 'leaf' #'leaf', #'eom' #
        )

#####

# UMAP then HDBSCAN
cl_obj.umap_reducer(
    nn = 30, # 10,
    mind = 0, # 0.01,
    sprd = 1, # don't use for tuning - leave set to 1
    nc = 3,
    color_in = df_train_ID['AggEcoregion'])


# new clustering on umap reduced data
cl_obj_umap = Clusterer(clust_vars =cl_obj.df_embedding_.drop(columns = ['STAID', 'Color', 'ColSize']),
    id_vars = df_train_mnexpl['STAID'])

cl_obj_umap.hdbscanner(
    min_clustsize = 50,
    min_sample =  1, #None, #
    gm_spantree = False,
    metric_in =  'euclidean', # 'manhattan', #
    clust_seleps = 0, # 0.05, # 0.5,
    clustsel_meth = 'eom' #'leaf', #'leaf' #
)

# # predict clusters for valin data
# # assigns output dataframe named df_hd_pred_ to object
# test.hd_predictor(
#     points_to_predict = df_valnit_trnsfrmd,
#     id_vars = df_valnit_mnexpl['STAID']
# )

# # print output dataframe
# test.df_hd_pred_



# %% ############
# specified hieararchical clustering
###########


# using all variables
# train, testin, and valnit
train_expl_in = df_train_mnexpl.reset_index(drop = True)
testin_expl_in = df_testin_mnexpl.reset_index(drop = True)
valnit_expl_in = df_valnit_mnexpl.reset_index(drop = True)

# divide training into specified number of categories (num_cats) of area
# how many categories:
num_Ks = 4

# calculate chunk size
chk_size = np.ceil(1/num_Ks * train_expl_in.shape[0])
# define temp dataframe ordered by DRAIN_SQKM
df_temp = train_expl_in[['STAID', 'DRAIN_SQKM']].sort_values(
    by = 'DRAIN_SQKM'
    ).reset_index(drop = True)

# define empty column to add cluster labels to
df_temp['Area_K'] = np.zeros(df_temp.shape[0]).astype(np.int16)

for i in range(0, num_Ks):
    if i < (num_Ks - 1):
        df_temp.loc[(chk_size * i):(chk_size * (i + 1)), 'Area_K'] = i
    else:
        df_temp.loc[(chk_size * i)::, 'Area_K'] = i

# print ranges of clusters
for i in range(0, num_Ks):
    temp_min = df_temp.loc[df_temp['Area_K'] == i, 'DRAIN_SQKM'].min()
    temp_max = df_temp.loc[df_temp['Area_K'] == i, 'DRAIN_SQKM'].max()
    print(f'K = {i}: \n min: {temp_min:.2f} \n max: {temp_max:.2f} \n \n')

    
# Standardize data

# define list of columns not to transform
# these columns are OHE so already either 0 or 1. 
# for distance metrics, use Manhattan which lends itself to capturing 
not_tr_in = ['GEOL_REEDBUSH_DOM_granitic', 
            'GEOL_REEDBUSH_DOM_quarternary', 'GEOL_REEDBUSH_DOM_sedimentary', 
            'GEOL_REEDBUSH_DOM_ultramafic', 'GEOL_REEDBUSH_DOM_volcanic']
            # 'GEOL_REEDBUSH_DOM_gneiss', 


# define vars to use in clustering
clust_vars_in = train_expl_in.loc[
    :, train_expl_in.columns.isin(df_cats.loc[df_cats['Custom_Cat'] == 'Climate', 'Features'])
]
# define id vars
id_vars_in = train_expl_in['STAID']

# define clusterer object
cl_obj = Clusterer(clust_vars = clust_vars_in,
    id_vars = id_vars_in)

# note that once input data is transformed, the transformed
# version will be used automatically in all functions related to
# Clusterer object
cl_obj.stand_norm(method = 'standardize'), # 'normalize'
    
# HDBSCAN on transformed data
cl_obj.hdbscanner(
    min_clustsize = 300,
    gm_spantree = True,
    metric_in = ''
)



# %%
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