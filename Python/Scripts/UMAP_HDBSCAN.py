'''
Ben Choat 10/6/2022

Script to perform specified hiearchical clustering
perform UMAP followed by HDBSCAN

some info ob HDBSCAN parameters:
min_cluster_size: integer
    Primary parameter to effect the resulting clustering.
    The minimum number of samples in a cluster
min_samples: integer
    Provides measure of how conservative you want your clustering to be.
    Larger the value, the more conservative the clustering - more points
    will be declared as noise and clusters will be restricted to progressively
    more dense areas.
    If not specified (commented out), then defaults to same value as min_clustize
metric_in: string 'euclidean' or 'manhattan'
gm_spantree: boolean
    Whether to generate the minimum spanning tree with regard to mutual reachability
    distance for later analysis
cluster_selection_epsilon: float between 0 and 1
    default value is 0. E.g., if set to 0.5 then DBSCAN clusters will be extracted
    for epsilon = 0.5, and clusters that emerged at distances greater than 0.5 will
    be untouched (helps allow small clusters w/o splitting large clusters)
cluster_selection_method: string
    'leaf' or 'eom' - Excess of Mass
    which cluster method to use
    eom has greater tendency to pick small number of clusters 
    leaf tends towards smaller homogeneous clusters

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
from itertools import product


# %%
# define variables
############

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'None' # 'AggEcoregion', 'None', 

# define which region to work with
region =  'All' # 'WestXeric' # 'NorthEast' # 'MxWdShld' #

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'mean_annual' # 'mean_annual', 'annual', 'monthly', 'daily'

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 

# directory where to place outputs
dir_umaphd = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'

# read in feature categories to be used for subsetting explanatory vars
# into cats of interest
feat_cats = pd.read_csv(f'{dir_umaphd}/FeatureCategories.csv')

# %%
# Load data
###########
# load data (explanatory, water yield, ID)
df_expl, df_WY, df_ID = load_data_fun(
    dir_work = dir_work, 
    time_scale = time_scale,
    train_val = 'train',
    clust_meth = clust_meth,
    region = region,
    standardize = False # whether or not to standardize data
    )

# remove id and time variables (e.g., STAID, year, month, etc.) from explanatory vars
# subset WY to version desired (ft)
# store staid's and date/year/month
if(time_scale == 'mean_annual'):
    STAID = df_expl['STAID']  
    df_expl.drop('STAID', axis = 1, inplace = True)
    #df_WY = df_WY['Ann_WY_ft']





# %% ############################
# USING PIPELINE FOR UMAP -> HDBSCAN HERE
#############################

# # %% Define parameter grid and gridsearch (exhaustive)

# parameters for the pipeline search
param_grid = {
    'umap_n_neighbors': [30, 60, 90], # for initial run do 6 for each
    'umap_min_dist': [0],
    'umap_metric': ['euclidean', 'cosine'],
    'umap_n_components': [10, 20, 30],
    'hd_min_cluster_size': [15, 30, 45, 60], #, 60],
    'hd_min_samples': [1],
    'hd_metric': ['euclidean'], #, 'manhattan'],
    'hd_cluster_selection_epsilon': [0, 0.2, 0.4, 0.6],
    'hd_cluster_selection_method': ['leaf']  # 'leaf' (smaller homogeneous) or 'eom' (fewer clusters)
}


# allNames = sorted(param_grid)
combos = product(*(param_grid[Name] for Name in param_grid))
combosdf = pd.DataFrame(combos)

combosdf.columns = [
    'n_neighbors',
    'min_distance',
    'umap_metric',
    'n_components',
    'min_cluster_size',
    'min_samples',
    'hdbscan_metric',
    'cluster_selection_epsilon',
    'cluster_selection_method'
    ]


# %%
# UMAP -> HDBSCAN on all expl vars
###############



# NOTE: THIS SECTION IS COMMENTED OUT BECAUSE IT HAS ALREADY BEEN EXECUTED, AND THE
# RELATED FILE IS ALREADY CREATED - IT IS READ IN BELOW

# subset to random sample for increased speed in testing
# combosAllVars = combosAllVars.sample(n = 5) # frac = 0.5
# print(combosAllVars)

# # define empty list to hold absolute DBCV scores
# dbcv_out = []
# # define empty list to hold relative DBCV scores
# reldbcv_out = []
# # define empty list to hold number of values identified as noise
# noise_count = []
# # define empty list to hold number of clusters
# num_clusters = []

# for i in combosAllVars.itertuples(index = False):
#     tempParams = np.array(i)
#     # define standard scaler
#     scaler = StandardScaler()

#     # define umap object
#     umap_reducer = umap.UMAP(
#                     n_neighbors = int(tempParams[0]),
#                     min_dist = int(tempParams[1]),
#                     metric = tempParams[2],
#                     n_components = int(tempParams[3]),
#                     spread = 1, 
#                     random_state = 100, 
#                     # n_epochs = 200, 
#                     n_jobs = -1)

#     hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(tempParams[4]),
#                                     min_samples = int(tempParams[5]), 
#                                     gen_min_span_tree = True,
#                                     metric = tempParams[6], # 'euclidean' or 'manhattan'
#                                     cluster_selection_epsilon = float(tempParams[7]),
#                                     cluster_selection_method = tempParams[8],
#                                     prediction_data = True)


#     # define pipline
#     pipe = Pipeline(
#         steps = [('scaler', scaler), 
#             ('umap_reducer', umap_reducer), 
#             ('hd_clusterer', hd_clusterer)]
#         )

#     # fit the model pipeline
#     pipe.fit(df_expl)   

#     # define temp vars for relative validity, number catchments 
#     # as noise, and number of clusters
#     relative_validity = hd_clusterer.relative_validity_
#     noise = len(hd_clusterer.labels_[hd_clusterer.labels_ == -1])
#     numclusts = len(np.unique(hd_clusterer.labels_))

#     # calculate absolute validity index
#     absValInd = hdbscan.validity_index(
#         X = umap_reducer.embedding_.astype(np.float64), 
#         labels = hd_clusterer.labels_,
#         metric = tempParams[6])

#     # append abslute validity index to list to later add to dataframe
#     dbcv_out.append(absValInd)
#     # append validity score to list to later add to dataframe holding parameters
#     reldbcv_out.append(relative_validity)
#     # calculate number of vals assigned -1 (noise)
#     noise_count.append(noise)
#     # count number of clusters (including noise)
#     num_clusters.append(numclusts)

#     print(f'\nparameters: {tempParams}\n')
#     print(f'Rel_DBCV: {hd_clusterer.relative_validity_}\n')
#     print(f'DBCV: {absValInd}\n')
#     print(f'Number of catchments assigned to noise: {noise}\n')
#     print(f'Number of Clusters {numclusts}\n')


# combosAllVars['Validity_Ind'] = dbcv_out
# combosAllVars['Rel_Validity_Ind'] = reldbcv_out
# combosAllVars['Noise_Count'] = noise_count
# combosAllVars['Numb_Clusters'] = num_clusters

# del(reldbcv_out, noise_count)

# # print 20 best performing parameter configurations
# combosAllVars.sort_values(
#     by = 'Validity_Ind', 
#     ascending = False)[80:100]#.head(40)

# # write results sorted by validity index to csv
# combosAllVars.sort_values(
#     by = 'Validity_Ind', ascending = False
#     ).to_csv(
#         'D:/Projects/GAGESii_ANNstuff/Data_Out/'
#         'UMAP_HDBSCAN/UMAP_HDBSCAN_ParamSearch_Results.csv',
#         index = False
#     )


# if above already executed, then read in combosAllVars 
combosAllVars = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/'
    'UMAP_HDBSCAN/UMAP_HDBSCAN_ParamSearch_Results.csv',
)

# combosAllVars = combosAllVars[combosAllVars['cluster_selection_method'] == 'leaf']

# combosAllVars.to_csv(
#         'D:/Projects/GAGESii_ANNstuff/Data_Out/'
#         'UMAP_HDBSCAN/UMAP_HDBSCAN_ParamSearch_Results.csv',
#         index = False
#     )

combosAllVars.loc[combosAllVars['Validity_Ind'] > 0.45, 'Validity_Ind'].unique()
combosAllVars.loc[combosAllVars['Validity_Ind'] > 0.45, 'Noise_Count'].unique()
combosAllVars.loc[combosAllVars['Validity_Ind'] > 0.45, 'Numb_Clusters'].unique()
combosAllVars = combosAllVars.loc[combosAllVars['Validity_Ind'] > 0.45]

# drop duplicated results
combosAllVars.drop_duplicates(
    subset = ['Validity_Ind', 'Rel_Validity_Ind', 'Noise_Count', 'Numb_Clusters'],
    inplace = True)
    
combosAllVars.reset_index(drop = True, inplace = True)

# define empty dataframe to append parameters with which to work in the next step
combosAllBest = pd.DataFrame(columns = combosAllVars.columns[0:9])


# filter to only highest validity index for each unique number of clusters
for i in combosAllVars['Numb_Clusters'].unique():
    print(i)
    # subset to clusters = i
    tmp = combosAllVars[combosAllVars['Numb_Clusters'] == i]
    # keep only max
    tmp = tmp[tmp['Validity_Ind'] == np.max(tmp['Validity_Ind'])]

    combosAllBest = pd.concat([combosAllBest, tmp.iloc[:, 0:9]])

combosAllBest.reset_index(drop = True, inplace = True)
print(combosAllBest)


# %% 
# Loop through best performing parameters, combosAllBest, and add columns
# to df_ID







# %%
# UMAP -> HDBSCAN on Natural then Anthropogenic vars
###############

# define features that are natural features
nat_feats = feat_cats.loc[
    feat_cats['Coarsest_Cat'] == 'Natural', 'Features'
].reset_index(drop = True)

# define features that are anthropogenic features
anthro_feats = feat_cats.loc[
    feat_cats['Coarsest_Cat'] == 'Anthro', 'Features'
].reset_index(drop = True)



#####
# loop through parameter grid for natural features
######


# allNames = sorted(param_grid)
combos = product(*(param_grid[Name] for Name in param_grid))
combosdf = pd.DataFrame(combos)

combosdf.columns = [
    'n_neighbors',
    'min_distance',
    'umap_metric',
    'n_components',
    'min_cluster_size',
    'min_samples',
    'hdbscan_metric',
    'cluster_selection_epsilon',
    'cluster_selection_method'
    ]

# subset to random sample for increased speed in testing
# combosdf = combosdf.sample(n = 5) # frac = 0.5
# print(combosdf)

# define empty list to hold absolute DBCV scores
dbcv_out = []
# define empty list to hold relative DBCV scores
reldbcv_out = []
# define empty list to hold number of values identified as noise
noise_count = []
# define empty list to hold number of clusters
num_clusters = []

df_work = df_expl[df_expl.columns.intersection(nat_feats)]


# subset params to random sample for increased speed in testing
combosdf = combosdf.sample(n = 5) # frac = 0.5
print(combosdf)
combosout_df = combosdf
for i in combosout_df.itertuples(index = False):
    tempParams = np.array(i)
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(tempParams[0]),
                    min_dist = int(tempParams[1]),
                    metric = tempParams[2],
                    n_components = int(tempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1)

    hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(tempParams[4]),
                                    min_samples = int(tempParams[5]), 
                                    gen_min_span_tree = True,
                                    metric = tempParams[6], # 'euclidean' or 'manhattan'
                                    cluster_selection_epsilon = float(tempParams[7]),
                                    cluster_selection_method = tempParams[8],
                                    prediction_data = True)


    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline
    pipe.fit(df_work)   

    # define temp vars for relative validity, number catchments 
    # as noise, and number of clusters
    relative_validity = hd_clusterer.relative_validity_
    noise = len(hd_clusterer.labels_[hd_clusterer.labels_ == -1])
    numclusts = len(np.unique(hd_clusterer.labels_))

    # calculate absolute validity index
    absValInd = hdbscan.validity_index(
        X = umap_reducer.embedding_.astype(np.float64), 
        labels = hd_clusterer.labels_,
        metric = tempParams[6])

    # append abslute validity index to list to later add to dataframe
    dbcv_out.append(absValInd)
    # append validity score to list to later add to dataframe holding parameters
    reldbcv_out.append(relative_validity)
    # calculate number of vals assigned -1 (noise)
    noise_count.append(noise)
    # count number of clusters (including noise)
    num_clusters.append(numclusts)

    print(f'\nparameters: {tempParams}\n')
    print(f'Rel_DBCV: {hd_clusterer.relative_validity_}\n')
    print(f'DBCV: {absValInd}\n')
    print(f'Number of catchments assigned to noise: {noise}\n')
    print(f'Number of Clusters {numclusts}\n')


combosout_df['Validity_Ind'] = dbcv_out
combosout_df['Rel_Validity_Ind'] = reldbcv_out
combosout_df['Noise_Count'] = noise_count
combosout_df['Numb_Clusters'] = num_clusters
combosout_df['VarType'] = np.repeat('Natural', combosout_df.shape[0])

del(reldbcv_out, noise_count)

# print 20 best performing parameter configurations
combosout_df.sort_values(
    by = 'Validity_Ind', 
    ascending = False).head(20)


#######
# %%
# loop through best performing grid combos, and follow up by
# performing HDBSCAN on anthropogenic featuers

# sort results from above clustering based on validity index
# then, for each number of clusters found where validity index > 0.45
# take the highest validity index result, and further cluster the results
# based on anthro features

prev_results = combosout_df[
    combosout_df['Validity_Ind'] > 0.45
].sort_values(
    by = 'Validity_Ind', ascending = False
    ).reset_index(drop = True)

# define empty dataframe to append parameters with which to work in the next step
combosBest = pd.DataFrame(columns = combosdf.columns[0:9])


# filter to only highest validity index for each unique number of clusters
for i in prev_results['Numb_Clusters'].unique():
    print(i)
    # subset to clusters = i
    tmp = prev_results[prev_results['Numb_Clusters'] == i]
    # keep only max
    tmp = tmp[tmp['Validity_Ind'] == np.max(tmp['Validity_Ind'])]

    combosBest = pd.concat([combosBest, tmp.iloc[:, 0:9]])

print(combosBest)


# loop through parameter sets in combosBest and for each cluster present
# based on natural vars, further cluster based on anthro vars
# first, loop through and write out labels indicating which cluster
# each station belongs to
# second, for each label within each natural cluster, further cluster
# based on anthro vars

df_worknats = df_expl[df_expl.columns.intersection(nat_feats)]
df_workanth = df_expl[df_expl.columns.intersection(anthro_feats)]


# allNames = sorted(param_grid)
combos = product(*(param_grid[Name] for Name in param_grid))
combosanth_df = pd.DataFrame(combos)

# combosanth_df.columns = [
#     'n_neighbors',
#     'min_distance',
#     'umap_metric',
#     'n_components',
#     'min_cluster_size',
#     'min_samples',
#     'hdbscan_metric',
#     'cluster_selection_epsilon',
#     'cluster_selection_method'
#     ]

# subset to random sample for increased speed in testing
# combosanth_df = combosanth_df.sample(n = 5) # frac = 0.5
# print(combosanth_df)

# define empty list to hold absolute DBCV scores
dbcv_out = []
# define empty list to hold relative DBCV scores
reldbcv_out = []
# define empty list to hold number of values identified as noise
noise_count = []
# define empty list to hold number of clusters
num_clusters = []

df_work = df_expl[df_expl.columns.intersection(nat_feats)]



# subset params to random sample for increased speed in testing
combosanth_df = combosanth_df.sample(n = 2) # frac = 0.5
print(combosanth_df)


for i in combosBest.itertuples(index = False):
    # define counter to use in column labels
    cnt = 0
    # assign parameter array to a variable
    tempParams = np.array(i)
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(tempParams[0]),
                    min_dist = int(tempParams[1]),
                    metric = tempParams[2],
                    n_components = int(tempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1)

    hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(tempParams[4]),
                                    min_samples = int(tempParams[5]), 
                                    gen_min_span_tree = True,
                                    metric = tempParams[6], # 'euclidean' or 'manhattan'
                                    cluster_selection_epsilon = float(tempParams[7]),
                                    cluster_selection_method = tempParams[8],
                                    prediction_data = True)


    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline to natural features
    pipe.fit(df_worknats)   

    nat_lbls_temp = pd.DataFrame({
        'STAID': STAID,
        'Nat_label': hd_clusterer.labels_
    })



    # Now loop through the individual clusters from the natural clustering and 
    # cluster based on antrhopogenic features
    
    for cluster in np.unique(hd_clusterer.labels_):
        # subset stations of interest 
        expl_work = df_workanth[hd_clusterer.labels_ == cluster]
        
        for params in combosanth_df.itertuples(index = False):
            tempParams = np.array(params)
            # define standard scaler
            scaler = StandardScaler()

            # define umap object
            umap_reducer = umap.UMAP(
                            n_neighbors = int(tempParams[0]),
                            min_dist = int(tempParams[1]),
                            metric = tempParams[2],
                            n_components = int(tempParams[3]),
                            spread = 1, 
                            random_state = 100, 
                            # n_epochs = 200, 
                            n_jobs = -1)

            hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(tempParams[4]),
                                            min_samples = int(tempParams[5]), 
                                            gen_min_span_tree = True,
                                            metric = tempParams[6], # 'euclidean' or 'manhattan'
                                            cluster_selection_epsilon = float(tempParams[7]),
                                            cluster_selection_method = tempParams[8],
                                            prediction_data = True)


            # define pipline
            pipe = Pipeline(
                steps = [('scaler', scaler), 
                    ('umap_reducer', umap_reducer), 
                    ('hd_clusterer', hd_clusterer)]
                )

            # fit the model pipeline
            pipe.fit(df_workanth)   

            # assign hdbscan cluster labels to staids
            anth_lbls_temp = pd.DataFrame({
                'STAID': STAID,
                'Anth_label': hd_clusterer.labels_
            })
        
            # write cluster labels to a new column in the ID dataframe
            df_ID[f'Nat_Anth_{cnt}'] = nat_lbls_temp['Nat_label'].astype(str) + \
                            '_' + \
                            anth_lbls_temp['Anth_label'].astype(str)
            
            # increment the counter
            cnt += 1

# write id data frame back to file

# df_ID.to_csv(
#     'place name here noker',jfkj
# )

# write results sorted by validity index to csv
# combosout_df.sort_values(
#     by = 'Validity_Ind', ascending = False
#     ).to_csv(
#         'D:/Projects/GAGESii_ANNstuff/Data_Out/'
#         'UMAP_HDBSCAN/UMAP_HDBSCAN_ParamSearch_Results.csv',
#         index = False
#     )










# %% ############################
# USING CLASS I WROTE BELOW HERE
# USE AFTER IDEAL PARAMS ARE IDENTIFIED
#############################
# %% ###################
# UMAP followed by HDBSCAN
########################

# define dataframe to add labels to (i.e., to hold )
df_labels = pd.DataFrame({
    'STAID': df_expl['STAID']
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
    id_vars = df_expl['STAID'])

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
train_expl_in = df_expl.reset_index(drop = True)
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
    id_vars = df_expl['STAID'])