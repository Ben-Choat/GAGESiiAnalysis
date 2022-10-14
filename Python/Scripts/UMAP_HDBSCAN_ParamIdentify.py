'''
Ben Choat 10/6/2022

This script identifies best parameters for user-specified hieararchical
clustering, and writes candidates out to csv files.

The script, UMAP_HDBSCAN_GetLables.py is then used to read in the
candidate parameters and write out labels to the ID files, which can 
later be used for subsetting regions/clusters in predictive modeling.

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

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'None' # 'AggEcoregion', 'None', 

# define which region to work with
region =  'All' # 'WestXeric' # 'NorthEast' # 'MxWdShld' #

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'mean_annual' # 'mean_annual', 'annual', 'monthly', 'daily'

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
    'hd_min_cluster_size': [30, 45, 60], #, 60],
    'hd_min_samples': [1],
    'hd_metric': ['euclidean'], #, 'manhattan'],
    'hd_cluster_selection_epsilon': [0], # , 0.2, 0.4, 0.6],
    'hd_cluster_selection_method': ['leaf', 'eom']  # 'leaf' (smaller homogeneous) or 'eom' (fewer clusters)
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



# define empty list to hold absolute DBCV scores
dbcv_out = []
# define empty list to hold relative DBCV scores
reldbcv_out = []
# define empty list to hold number of values identified as noise
noise_count = []
# define empty list to hold number of clusters
num_clusters = []

# assign combosdf to working variable
combosAllVars = combosdf
# subset to random sample for increased speed in testing
# combosAllVars = combosAllVars.sample(n = 5) # frac = 0.5
# print(combosAllVars)

for i in combosAllVars.itertuples(index = False):
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
    pipe.fit(df_expl)   

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


combosAllVars['Validity_Ind'] = dbcv_out
combosAllVars['Rel_Validity_Ind'] = reldbcv_out
combosAllVars['Noise_Count'] = noise_count
combosAllVars['Numb_Clusters'] = num_clusters

del(reldbcv_out, noise_count)

# # print 20 best performing parameter configurations
# combosAllVars.sort_values(
#     by = 'Validity_Ind', 
#     ascending = False)[80:100]#.head(40)

# subset to best performing parameter sets
# drop duplicated results
combosAllVars.drop_duplicates(
    subset = ['Validity_Ind', 'Rel_Validity_Ind', 'Noise_Count', 'Numb_Clusters'],
    inplace = True)
    
combosAllVars.reset_index(drop = True, inplace = True)

combosAllVars.sort_values(by = 'Validity_Ind', ascending = False)


# drop parameters where clusters had dbcv < 0.45
combosAllVars = combosAllVars.loc[
    combosAllVars['Validity_Ind'].astype(float) >= 0.45
    ].sort_values(
    by = 'Validity_Ind', ascending = False
    ).reset_index(drop = True)

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


# write results sorted by validity index to csv
combosAllBest.to_csv(
        'D:/Projects/GAGESii_ANNstuff/Data_Out/'
        'UMAP_HDBSCAN/UMAP_HDBSCAN_AllVars_ParamSearch_Results.csv',
        index = False
    )


# if above already executed, then read in combosAllVars 
combosAllBest = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/'
    'UMAP_HDBSCAN/UMAP_HDBSCAN_AllVars_ParamSearch_Results.csv',
)






# %%
# UMAP -> HDBSCAN on Natural then Anthropogenic vars
###############

# parameters for the natural feature pipeline search
paramNat_grid = {
    'umap_n_neighbors': [30, 60, 90], # for initial run do 6 for each
    'umap_min_dist': [0],
    'umap_metric': ['euclidean', 'cosine'],
    'umap_n_components': [10, 20, 30],
    'hd_min_cluster_size': [120, 180, 240], # Set larger here because will later susbet clusters
    'hd_min_samples': [1],
    'hd_metric': ['euclidean'], #, 'manhattan'],
    'hd_cluster_selection_epsilon': [0], #, 0.2, 0.4, 0.6],
    'hd_cluster_selection_method': ['leaf', 'eom']  # 'leaf' (smaller homogeneous) or 'eom' (fewer clusters)
}



# define features that are natural features
nat_feats = feat_cats.loc[
    feat_cats['Coarsest_Cat'] == 'Natural', 'Features'
].reset_index(drop = True)

# define features that are anthropogenic features
anthro_feats = feat_cats.loc[
    feat_cats['Coarsest_Cat'] == 'Anthro', 'Features'
].reset_index(drop = True)



#####
# loop through parameter grid for natural features identifying best param sets
######

# allNames = sorted(paramNat_grid)
combos = product(*(paramNat_grid[Name] for Name in paramNat_grid))
combosNat = pd.DataFrame(combos)

combosNat.columns = [
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
# combosNat = combosNat.sample(n = 5) # frac = 0.5
# print(combosNat)
combosout_df = combosNat
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

# # print 20 best performing parameter configurations
# combosout_df.sort_values(
#     by = 'Validity_Ind', 
#     ascending = False).head(20)


# sort results from above clustering based on validity index
# then, for each number of clusters found where validity index > 0.45
# take the highest validity index result, and further cluster the results
# based on anthro features

prev_results = combosout_df[
    combosout_df['Validity_Ind'].astype(float) >= 0.45
].sort_values(
    by = 'Validity_Ind', ascending = False
    ).reset_index(drop = True)

# define empty dataframe to append parameters with which to work in the next step
combosNatBest = pd.DataFrame(columns = combosNat.columns[0:9])


# filter to only highest validity index for each unique number of clusters
for i in prev_results['Numb_Clusters'].unique():
    print(i)
    # subset to clusters = i
    tmp = prev_results[prev_results['Numb_Clusters'] == i]
    # keep only max
    tmp = tmp[tmp['Validity_Ind'] == np.max(tmp['Validity_Ind'])]

    combosNatBest = pd.concat([combosNatBest, tmp.iloc[:, 0:9]])

print(combosNatBest)

#######
# %%
# loop through best performing grid combos, and follow up by
# performing HDBSCAN on anthropogenic featuers
##############



# loop through parameter sets in combosBest and for each cluster present
# based on natural vars, further cluster based on anthro vars
# first, loop through and write out labels indicating which cluster
# each station belongs to
# second, for each label within each natural cluster, further cluster
# based on anthro vars

# define explanatory features to be used in clustering for
# natural and anthropogenic clustering stages
df_worknats = df_expl[df_expl.columns.intersection(nat_feats)]
df_workanth = df_expl[df_expl.columns.intersection(anthro_feats)]


# parameters for the natural feature pipeline search
paramAnth_grid = {
    'umap_n_neighbors': [30, 60, 90], # for initial run do 6 for each
    'umap_min_dist': [0],
    'umap_metric': ['euclidean', 'cosine'],
    'umap_n_components': [10, 20, 30],
    'hd_min_cluster_size': [30, 45, 90], #, 60],
    'hd_min_samples': [1],
    'hd_metric': ['euclidean'], #, 'manhattan'],
    'hd_cluster_selection_epsilon': [0], #, 0.2, 0.4, 0.6],
    'hd_cluster_selection_method': ['leaf', 'eom']  # 'leaf' (smaller homogeneous) or 'eom' (fewer clusters)
}

# allNames = sorted(param_grid)
combos = product(*(paramAnth_grid[Name] for Name in paramAnth_grid))
combosanth_df = pd.DataFrame(combos)


# subset to random sample for increased speed in testing
# combosanth_df = combosanth_df.sample(n = 5) # frac = 0.5
# print(combosanth_df)

# # define empty list to hold absolute DBCV scores
# Nat_dbcvout = []
# Anth_dbcvout = []
# # define empty list to hold relative DBCV scores
# Nat_reldbcvout = []
# Anth_reldbcvout = []
# # define empty list to hold number of values identified as noise
# Nat_noise_count = []
# Anth_noise_count = []
# # define empty list to hold number of clusters
# Nat_num_clusters = []
# Anth_num_clusters = []

# define empty params dataframe to hold param combos

nat_anth_combos = pd.DataFrame(
    columns = 
        list(f'Nat_{key}' for key in list(paramAnth_grid.keys())) +
        list(f'Anth_{key}' for key in list(paramAnth_grid.keys())) +
        list(f'Nat_{pm}' for pm in ['Validity_Ind', 'Noise_Count', 'Numb_Clusters']) +
        list(f'Anth_{pm}' for pm in ['Validity_Ind', 'Noise_Count', 'Numb_Clusters'])
    
)

# define holder to work with
temp_df = nat_anth_combos

# loop through best params from natural features and identify
# best params for anth features
for i in combosNatBest.itertuples(index = False):

    print(i)

    # assign parameter array to a variable
    NatTempParams = np.array(i)
    # define standard scaler
    scaler = StandardScaler()

    # define umap object
    umap_reducer = umap.UMAP(
                    n_neighbors = int(NatTempParams[0]),
                    min_dist = int(NatTempParams[1]),
                    metric = NatTempParams[2],
                    n_components = int(NatTempParams[3]),
                    spread = 1, 
                    random_state = 100, 
                    # n_epochs = 200, 
                    n_jobs = -1)

    hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(NatTempParams[4]),
                                    min_samples = int(NatTempParams[5]), 
                                    gen_min_span_tree = True,
                                    metric = NatTempParams[6], # 'euclidean' or 'manhattan'
                                    cluster_selection_epsilon = float(NatTempParams[7]),
                                    cluster_selection_method = NatTempParams[8],
                                    prediction_data = True)


    # define pipline
    pipe = Pipeline(
        steps = [('scaler', scaler), 
            ('umap_reducer', umap_reducer), 
            ('hd_clusterer', hd_clusterer)]
        )

    # fit the model pipeline to natural features
    pipe.fit(df_worknats)

    # define temp vars for relative validity, number catchments 
    # as noise, and number of clusters
    # Nat_relative_validity = hd_clusterer.relative_validity_
    Nat_noise = len(hd_clusterer.labels_[hd_clusterer.labels_ == -1])
    Nat_numclusts = len(np.unique(hd_clusterer.labels_))

    # calculate absolute validity index
    Nat_absValInd = hdbscan.validity_index(
        X = umap_reducer.embedding_.astype(np.float64), 
        labels = hd_clusterer.labels_,
        metric = NatTempParams[6])



    # Now loop through the individual clusters from the natural clustering and 
    # cluster based on antrhopogenic features
    
    for cluster in np.unique(hd_clusterer.labels_):
        # subset stations of interest 
        expl_work = df_workanth[hd_clusterer.labels_ == cluster]
        
        for params in combosanth_df.itertuples(index = False):
            AnthTempParams = np.array(params)
            # define standard scaler
            scaler = StandardScaler()

            # define umap object
            umap_reducer = umap.UMAP(
                            n_neighbors = int(AnthTempParams[0]),
                            min_dist = int(AnthTempParams[1]),
                            metric = AnthTempParams[2],
                            n_components = int(AnthTempParams[3]),
                            spread = 1, 
                            random_state = 100, 
                            # n_epochs = 200, 
                            n_jobs = -1)

            hd_clusterer = hdbscan.HDBSCAN(min_cluster_size = int(AnthTempParams[4]),
                                            min_samples = int(AnthTempParams[5]), 
                                            gen_min_span_tree = True,
                                            metric = AnthTempParams[6], # 'euclidean' or 'manhattan'
                                            cluster_selection_epsilon = float(AnthTempParams[7]),
                                            cluster_selection_method = AnthTempParams[8],
                                            prediction_data = True)


            # define pipline
            pipe = Pipeline(
                steps = [('scaler', scaler), 
                    ('umap_reducer', umap_reducer), 
                    ('hd_clusterer', hd_clusterer)]
                )

            # fit the model pipeline
            pipe.fit(df_workanth)   


            # define temp vars for relative validity, number catchments 
            # as noise, and number of clusters
            # Anth_relative_validity = hd_clusterer.relative_validity_
            Anth_noise = len(hd_clusterer.labels_[hd_clusterer.labels_ == -1])
            Anth_numclusts = len(np.unique(hd_clusterer.labels_))

            # calculate absolute validity index
            Anth_absValInd = hdbscan.validity_index(
                X = umap_reducer.embedding_.astype(np.float64), 
                labels = hd_clusterer.labels_,
                metric = AnthTempParams[6])


            # write nat and anth performance metrics to numpy arrays
            tempNat = np.array([Nat_absValInd, Nat_noise, Nat_numclusts])
            tempAnth = np.array([Anth_absValInd, Anth_noise, Anth_numclusts])

            # combine all parameters and metrics into a single numpy array
            tempRow = list(np.hstack(
                [NatTempParams, 
                AnthTempParams,
                tempNat,
                tempAnth]
            ))

            # write all params and metrics to dataframe and append to nat_anth_combos df
            temp_df.loc[len(temp_df)] = tempRow


combosNatAnthVars = temp_df

# subset to best performing parameter sets
# drop duplicated results
combosNatAnthVars.drop_duplicates(
    subset = ['Nat_Validity_Ind',  
                'Nat_Noise_Count', 
                'Nat_Numb_Clusters',
                'Anth_Validity_Ind', 
                'Anth_Noise_Count', 
                'Anth_Numb_Clusters'],
    inplace = True)
    
combosNatAnthVars.reset_index(drop = True, inplace = True)

combosNatAnthVars = combosNatAnthVars.loc[
    combosNatAnthVars['Anth_Validity_Ind'].astype(float) >= 0.45
    ].sort_values(
    by = 'Anth_Validity_Ind', ascending = False
    ).reset_index(drop = True)

# define empty dataframe to append parameters with which to work in the next step
combosNatAnthBest = pd.DataFrame(columns = combosNatAnthVars.columns[0:18])


# filter to only highest validity index for each unique number of clusters
for i in combosNatAnthVars['Nat_Numb_Clusters'].unique():
    print(i)
    for j in combosNatAnthVars['Anth_Numb_Clusters'].unique():
        # subset to clusters = i
        tmp = combosNatAnthVars[
            (combosNatAnthVars['Nat_Numb_Clusters'] == i) &
            (combosNatAnthVars['Anth_Numb_Clusters'] == j)
        ]
        # keep only max
        tmp = tmp[tmp['Anth_Validity_Ind'] == np.max(tmp['Anth_Validity_Ind'])]

        combosNatAnthBest = pd.concat([combosNatAnthBest, tmp.iloc[:, 0:9]])

combosNatAnthBest.reset_index(drop = True, inplace = True)

print(combosNatAnthBest)


# write results sorted by validity index to csv
combosNatAnthBest.to_csv(
        'D:/Projects/GAGESii_ANNstuff/Data_Out/'
        'UMAP_HDBSCAN/UMAP_HDBSCAN_NatAnthVars_ParamSearch_Results.csv',
        index = False
    )


# if above already executed, then read in combosNatAnthVars 
combosNatAnthBest = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/'
    'UMAP_HDBSCAN/UMAP_HDBSCAN_NatAnthVars_ParamSearch_Results.csv',
)




# %%
