'''
Ben Choat 10/12/2022

This script reads in candidate parameters from csv's created by
UMAP_HDBSCAN_ParamIdentify.py and then writes out labels to the ID files, 
which can later be used for subsetting regions/clusters in predictive modeling.

NOTE: testin catchments are same as train, so testin is not included in this 
script other than writing the training labels to testin labels ID file

'''
# %% import libraries and classes


from GAGESii_Class import Clusterer
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Load_Data import load_data_fun
# from itertools import product


# %%
# define variables and load data
############
# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# directory where to place outputs
dir_umaphd = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'

# read in feature categories to be used for subsetting explanatory vars
# into cats of interest
feat_cats = pd.read_csv(f'{dir_umaphd}/FeatureCategories.csv')

# read in candidate parameters from clustering using all vars at once
cand_parms_all = pd.read_csv(
    f'{dir_umaphd}/UMAP_HDBSCAN_AllVars_ParamSearch_Results.csv'
    )
# and then from clustering based on Nat then Anth vars
cand_parms_nat = pd.read_csv(
    f'{dir_umaphd}/UMAP_HDBSCAN_NatVars_ParamSearch_Results.csv'
    )
# and then from clustering based on Anth vars
cand_parms_anth = pd.read_csv(
    f'{dir_umaphd}/UMAP_HDBSCAN_AnthroVars_ParamSearch_Results.csv'
    )


# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'None' # 'AggEcoregion', 'None', 

# define which region to work with
region =  'All' # 'WestXeric' # 'NorthEast' # 'MxWdShld' #

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
# should be 'mean_annual' for clustering
time_scale = 'mean_annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# load data (explanatory, water yield, ID)
# training
df_trainexpl, df_trainWY, df_trainID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'train',
    clust_meth = clust_meth,
    region = region,
    standardize = False # whether or not to standardize data
    )

# valnit
df_valnitexpl, df_valnitWY, df_valnitID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'valnit',
    clust_meth = clust_meth,
    region = region,
    standardize = False # whether or not to standardize data
    )


# drop lookback/antecedent climate vars
df_trainexpl = df_trainexpl.drop(
    df_trainexpl.columns[
        df_trainexpl.columns.str.contains('tmin') |
        df_trainexpl.columns.str.contains('tmax') |
        df_trainexpl.columns.str.contains('prcp') |
        df_trainexpl.columns.str.contains('vp') |
        df_trainexpl.columns.str.contains('swe')
        ],
    axis = 1
    )

# drop lookback/antecedent climate vars
df_valnitexpl = df_valnitexpl.drop(
    df_valnitexpl.columns[
        df_valnitexpl.columns.str.contains('tmin') |
        df_valnitexpl.columns.str.contains('tmax') |
        df_valnitexpl.columns.str.contains('prcp') |
        df_valnitexpl.columns.str.contains('vp') |
        df_valnitexpl.columns.str.contains('swe')
        ],
    axis = 1
    )

# remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
# subset WY to version desired (ft)
# store staid's and date/year/month

# training
STAIDtrain = df_trainexpl['STAID']  
df_trainexpl.drop('STAID', axis = 1, inplace = True)
# valnit
STAIDvalnit = df_valnitexpl['STAID']  
df_valnitexpl.drop('STAID', axis = 1, inplace = True)




# %%
# Get labels for clustering using all features at once
########
# loop through candidate params, define pipeline, fit to training data
# predict training and valnit clusters, assign labels it ID_dfs

# create empty dataframes to hold scores results
# i.e., how well each catchment fit within the catchment it
# was placed

df_trainscores = pd.DataFrame({'STAID': df_trainID['STAID']})
df_valnitscores = pd.DataFrame({'STAID': df_valnitID['STAID']})
# define counter to be used in naming columns
cnt = 0

# keep only 5 parameter sets with best validity index scores
cand_parms_all = cand_parms_all[0:5]

for params in cand_parms_all.itertuples(index = False):
    print(params)

    TempParams = np.array(params)
    
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(TempParams[0]),
                    min_dist = int(TempParams[1]),
                    metric = TempParams[2],
                    n_components = int(TempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1
                    )

    hd_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size = int(TempParams[4]),
                    min_samples = int(TempParams[5]), 
                    gen_min_span_tree = True,
                    metric = TempParams[6], # 'euclidean' or 'manhattan'
                    cluster_selection_epsilon = float(TempParams[7]),
                    cluster_selection_method = TempParams[8],
                    prediction_data = True
                    )


       
    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline
    pipe.fit(df_trainexpl)   

    # hdbscan implements 'approximate_predict' instead of 'predict'
    # so cannot use pipeline for prediction.

    # scale, transform, and predict valnit data
    valnit_temp = scaler.transform(df_valnitexpl)
    valnit_temp = umap_reducer.transform(valnit_temp)
    valnit_labels, scores = hdbscan.approximate_predict(
        hd_clusterer, valnit_temp)


    # assign labels to df_ID's
    df_trainID[f'All_{cnt}'] = hd_clusterer.labels_
    df_valnitID[f'All_{cnt}'] = valnit_labels

    # update strength dataframe
    df_trainscores[f'All_{cnt}'] = hd_clusterer.labels_
    df_trainscores[f'All_{cnt}_score'] = hd_clusterer.probabilities_

    df_valnitscores[f'All_{cnt}'] = valnit_labels
    df_valnitscores[f'All_{cnt}_score'] = scores

    cnt += 1



# %%
# Get labels for clustering using natural features
########

# %%
# Get labels for clustering using natural vars
########
# loop through candidate params, define pipelikne, fit to training data
# predict training and valnit clusters, assign labels it ID_dfs

# subset explanatory vars to natural and anthropogenic subsets
# train Nat
df_trainexpl_Nat = df_trainexpl[
    df_trainexpl.columns.intersection(
        feat_cats.loc[feat_cats['Coarsest_Cat'] == 'Natural', 'Features'])
    ]
# valnit Nat
df_valnitexpl_Nat = df_valnitexpl[
    df_valnitexpl.columns.intersection(
        feat_cats.loc[feat_cats['Coarsest_Cat'] == 'Natural', 'Features'])
    ]

# loop through candidate params, define pipeline, fit to training data
# predict training and valnit clusters, assign labels it ID_dfs

# define counter to be used in naming columns
cnt = 0

# keep only 5 parameter sets with best validity index scores
cand_parms_nat = cand_parms_nat[0:1]

for params in cand_parms_nat.itertuples(index = False):
    print(params)

    TempParams = np.array(params)
    
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(TempParams[0]),
                    min_dist = int(TempParams[1]),
                    metric = TempParams[2],
                    n_components = int(TempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1
                    )

    hd_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size = int(TempParams[4]),
                    min_samples = int(TempParams[5]), 
                    gen_min_span_tree = True,
                    metric = TempParams[6], # 'euclidean' or 'manhattan'
                    cluster_selection_epsilon = float(TempParams[7]),
                    cluster_selection_method = TempParams[8],
                    prediction_data = True
                    )


       
    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline
    pipe.fit(df_trainexpl_Nat)   

    # hdbscan implements 'approximate_predict' instead of 'predict'
    # so cannot use pipeline for prediction.

    # scale, transform, and predict valnit data
    valnit_temp = scaler.transform(df_valnitexpl_Nat)
    valnit_temp = umap_reducer.transform(valnit_temp)
    valnit_labels, scores = hdbscan.approximate_predict(
        hd_clusterer, valnit_temp)


    # assign labels to df_ID's
    df_trainID[f'Nat_{cnt}'] = hd_clusterer.labels_
    df_valnitID[f'Nat_{cnt}'] = valnit_labels

    # update strength dataframe
    df_trainscores[f'Nat_{cnt}'] = hd_clusterer.labels_
    df_trainscores[f'Nat_{cnt}_score'] = hd_clusterer.probabilities_

    df_valnitscores[f'Nat_{cnt}'] = valnit_labels
    df_valnitscores[f'Nat_{cnt}_score'] = scores
    
    cnt += 1



# %%
# Get labels for clustering using anthropogenic vars
########
# train anthropogenic
df_trainexpl_Anthro = df_trainexpl[
    df_trainexpl.columns.intersection(
        feat_cats.loc[feat_cats['Coarsest_Cat'] == 'Anthro', 'Features'])
    ]

# train anthropogenic
df_valnitexpl_Anthro = df_valnitexpl[
    df_valnitexpl.columns.intersection(
        feat_cats.loc[feat_cats['Coarsest_Cat'] == 'Anthro', 'Features'])
    ]

# loop through candidate params, define pipeline, fit to training data
# predict training and valnit clusters, assign labels it ID_dfs

# define counter to be used in naming columns
cnt = 0

# keep only 5 parameter sets with best validity index scores
cand_parms_anth = cand_parms_anth[0:5]

for params in cand_parms_anth.itertuples(index = False):
    print(params)

    TempParams = np.array(params)
    
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(TempParams[0]),
                    min_dist = int(TempParams[1]),
                    metric = TempParams[2],
                    n_components = int(TempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1
                    )

    hd_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size = int(TempParams[4]),
                    min_samples = int(TempParams[5]), 
                    gen_min_span_tree = True,
                    metric = TempParams[6], # 'euclidean' or 'manhattan'
                    cluster_selection_epsilon = float(TempParams[7]),
                    cluster_selection_method = TempParams[8],
                    prediction_data = True
                    )


       
    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline
    pipe.fit(df_trainexpl_Anthro)   

    # hdbscan implements 'approximate_predict' instead of 'predict'
    # so cannot use pipeline for prediction.

    # scale, transform, and predict valnit data
    valnit_temp = scaler.transform(df_valnitexpl_Anthro)
    valnit_temp = umap_reducer.transform(valnit_temp)
    valnit_labels, scores = hdbscan.approximate_predict(
        hd_clusterer, valnit_temp)


    # assign labels to df_ID's
    df_trainID[f'Anth_{cnt}'] = hd_clusterer.labels_
    df_valnitID[f'Anth_{cnt}'] = valnit_labels

    # update strength dataframe
    df_trainscores[f'Anth_{cnt}'] = hd_clusterer.labels_
    df_trainscores[f'Anth_{cnt}_score'] = hd_clusterer.probabilities_

    df_valnitscores[f'Anth_{cnt}'] = valnit_labels
    df_valnitscores[f'Anth_{cnt}_score'] = scores



    cnt += 1



# %%
# write ID and score dataframes to csv
#####################

df_trainID.to_csv(
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_train.csv',
    index = False
    )

df_valnitID.to_csv(
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/AllVars_Partitioned/ID_valnit.csv',
    index = False
    )


df_trainscores.to_csv(
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/ClusterScores_train.csv',
    index = False
    )

df_valnitscores.to_csv(
    f'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/ClusterScores_valnit.csv',
    index = False
    )



# %%

# create dataframes to write results
df_tr_noise = pd.DataFrame(columns = ['Cluster', 'Noise', 'Clust_Count'])
df_val_noise = pd.DataFrame(columns = ['Cluster', 'Noise', 'Clust_Count'])

# loop through training and valnit cluster labels and count noise in each
# training
for col in df_trainID.columns[
    df_trainID.columns.str.contains('All') | 
    df_trainID.columns.str.contains('Nat') |
    df_trainID.columns.str.contains('Anth')
    ]:
        noise = len(df_trainID[df_trainID[col] == -1])

        # number of clusters
        clustcount = len(df_trainID[col].unique())

        print(f'{col}: {noise} \n {col}: {clustcount}')

        df_temp = pd.DataFrame({
            'Cluster': col,
            'Noise': [noise],
            'Clust_Count': [clustcount]
        })

        df_tr_noise = pd.concat([df_tr_noise, df_temp])

# valnit
for col in df_valnitID.columns[
    df_valnitID.columns.str.contains('All') | 
    df_valnitID.columns.str.contains('Nat') |
    df_valnitID.columns.str.contains('Anth')
    ]:
        noise = len(df_valnitID[df_valnitID[col] == -1])

        # number of clusters
        clustcount = len(df_valnitID[col].unique())

        print(f'{col}: {noise} \n {col}: {clustcount}')

        df_temp = pd.DataFrame({
            'Cluster': col,
            'Noise': [noise],
            'Clust_Count': [clustcount]
        })

        df_val_noise = pd.concat([df_val_noise, df_temp])


# add label for training or valnit and join dataframes
df_tr_noise['Data'] = 'train'
df_val_noise['Data'] = 'valnit'

df_noise = pd.concat([df_tr_noise, df_val_noise]).reset_index(drop = True)

# write to csv
df_noise.to_csv(
    f'{dir_umaphd}/UMAP_HDBSCAN_NoiseCounts.csv',
    index = False
)






######################################

# NOTE: CODE BELOW HERE WAS INTENDED TO FURTHER CLUSTER CATCHMENTS 
# BASED ON ANTHROPOGENIC FEATURES, AFTER NATURAL FEATURES
# # define counter to be used in naming columns
# cnt = 0

# for params in cand_parms_nat.itertuples(index = False):
#     print(params)

#     TempParams = np.array(params)
    
#     # define standard scaler
#     scaler = StandardScaler()

#     # define umap object
#     umap_reducer = umap.UMAP(
#                     n_neighbors = int(TempParams[0]),
#                     min_dist = int(TempParams[1]),
#                     metric = TempParams[2],
#                     n_components = int(TempParams[3]),
#                     spread = 1, 
#                     random_state = 100, 
#                     # n_epochs = 200, 
#                     n_jobs = -1)

#     hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(TempParams[4]),
#                                     min_samples = int(TempParams[5]), 
#                                     gen_min_span_tree = True,
#                                     metric = TempParams[6], # 'euclidean' or 'manhattan'
#                                     cluster_selection_epsilon = float(TempParams[7]),
#                                     cluster_selection_method = TempParams[8],
#                                     prediction_data = True)


       
#     # define pipline
#     pipe = Pipeline(
#         steps = [('scaler', scaler), 
#             ('umap_reducer', umap_reducer), 
#             ('hd_clusterer', hd_clusterer)]
#         )

#     # fit the model pipeline
#     pipe.fit(df_trainexpl_Nat)   

#     # hdbscan implements 'approximate_predict' instead of 'predict'
#     # so cannot use pipeline for prediction.

#     # scale, transform, and predict valnit data
#     valnit_temp = scaler.transform(df_valnitexpl_Nat)
#     valnit_temp = umap_reducer.transform(valnit_temp)
#     valnit_labels_nat, scores = hdbscan.approximate_predict(
#         hd_clusterer, valnit_temp)



#     # for each cluster identified, further cluster the data based
#     # on anthropogenic features

#     for label in hd_clusterer.labels_:

#         # start a new counter to use in labeling columns
#         cnt2 = 0

#         # define standard scaler
#         scaler2 = StandardScaler()

#         # define umap object
#         umap_reducer2 = umap.UMAP(
#                         n_neighbors = int(TempParams[9]),
#                         min_dist = int(TempParams[10]),
#                         metric = TempParams[11],
#                         n_components = int(TempParams[12]),
#                         spread = 1, 
#                         random_state = 100, 
#                         # n_epochs = 200, 
#                         n_jobs = -1)

#         hd_clusterer2 = hdbscan.HDBSCAN(min_cluster_size = int(TempParams[13]),
#                                         min_samples = int(TempParams[14]), 
#                                         gen_min_span_tree = True,
#                                         metric = TempParams[15], # 'euclidean' or 'manhattan'
#                                         cluster_selection_epsilon = float(TempParams[16]),
#                                         cluster_selection_method = TempParams[17],
#                                         prediction_data = True)


        
#         # define pipline
#         pipe2 = Pipeline(
#             steps = [('scaler', scaler2), 
#                 ('umap_reducer', umap_reducer2), 
#                 ('hd_clusterer', hd_clusterer2)]
#             )

#         # fit the model pipeline
#         pipe2.fit(df_trainexpl_Anthro)   

#         # hdbscan implements 'approximate_predict' instead of 'predict'
#         # so cannot use pipeline for prediction.

#         # scale, transform, and predict valnit data
#         valnit_temp = scaler2.transform(df_valnitexpl_Nat)
#         valnit_temp = umap_reducer2.transform(valnit_temp)
#         valnit_labels_anth, scores = hdbscan.approximate_predict(
#             hd_clusterer2, valnit_temp)

        
#         # concatenate cluster labels together for final label
#         lbls_train = [label + '_' + x for x in hd_clusterer2.labels_]
#         lbls_valnit = [label + '_' + x for x in valnit_labels_anth]

#         # assign labels to df_ID's
#         df_trainID[f'All_{cnt}'] = lbls_train
#         df_valnitID[f'All_{cnt}'] = lbls_valnit

#         cnt += 1




# %%
