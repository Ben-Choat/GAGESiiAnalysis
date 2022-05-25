# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:44:56 2022

@author: bench
"""

# %% Load packages

import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly as ply
import plotly.io as pio5
import plotly.express as px
import seaborn as sns
import pandas as pd
import umap
import hdbscan
from mlxtend.plotting import heatmap
# plot inline
# %matplotlib inline 
# try to make plot in separate window
# %matplotlib auto 
# make plots interactive
# %matplotlib notebook # doesn't seem to work
# try to make plot in separate window
# %matplotlib qt 
# set some seaborn plotting options
#sns.set_context('poster')
#sns.set_style('white')
#sns.set_color_codes()
#plot_kwds = {'alpha': 0.5, 's': 0.1, 'linewidths':0}

# open plots in browser
#pio.renderers.default='browser'

# %% Read in data

# water yield directory
dir_WY = 'D:/DataWorking/USGS_discharge/train_val_test'
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'

# Annual Water yield
df_train_anWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/annual_WY/Ann_WY_train.csv',
    dtype={"site_no":"string"}
    )

# mean annual water yield
df_train_mnanWY= df_train_anWY.groupby(
    'site_no', as_index=False
).mean().drop(columns = ["wtryr"])

# Monthly Water yield
df_train_mnthWY = pd.read_csv(
    f'{dir_WY}/yrs_98_12/monthly_WY/Mnthly_WY_train.csv',
    dtype={"site_no":"string"}
    )

df_train_mnthWY.shape

# GAGESii-ts explanatory vars
# reading in expl vars where ts vars were taken to be the value of the 
# nearest year represented in the data
df_Ex_ts = pd.read_csv(
    f'{dir_expl}/GAGESii_ts/GAGESts_NearestYrs_Wide.csv',
    dtype={"STAID": "string"}
    )
df_Ex_ts.shape

#####
# subset to stations in training data
df_train_Exts = df_Ex_ts[df_Ex_ts['STAID'].
                         isin(df_train_mnthWY['site_no'])
                         ][df_Ex_ts['year'].
                         isin(df_train_mnthWY['wtryr'])
                           ]
df_train_Exts.shape
# create dataframe with average ts variables
df_train_Exts_mn = (
    df_train_Exts.
    groupby('STAID', as_index=False).mean()
    ).drop(columns = 'year')

#####
# GAGESii - Static explanatory vars
df_Ex_st = pd.read_csv(
    f'{dir_expl}/GAGES_Static_Filtered.csv',
    dtype={'STAID': 'string'}
    )

# Subset to training stations
df_train_Exst = df_Ex_st[df_Ex_st['STAID'].
                         isin(df_train_mnthWY['site_no'])
                         ]
                             
# Create one-hot encoded variables for GEOL_REEDBUSH_DOM
GEOL_ohe = pd.get_dummies(df_train_Exst['GEOL_REEDBUSH_DOM'])
GEOL_ohe['STAID'] = df_train_Exst['STAID']

# remove columns not of interest
df_train_Exst = df_train_Exst.drop(
    columns = ['GEOL_REEDBUSH_DOM',
        'GEOL_HUNT_DOM_CODE',
        'HYDRO_DISTURB_INDX',
        'CLASS',
        'BFI_AVE']
    ).merge(
        GEOL_ohe,
        on = 'STAID')                            
 
#####
# read in id data
df_id = pd.read_csv(
    f'{dir_expl}/GAGES_idVars.csv',
    dtype = {"STAID":"string"}
    )
# subset to only stations in df_train
df_train_id = df_id[df_id['STAID'].isin(df_train_mnthWY['site_no'])]

#####
# Merge ts and static vars to one df
df_train_Exall = pd.merge(df_train_Exts_mn,
                          df_train_Exst)


# %% UMAP on static vars

for nn in [25]: #[10, 15, 25, 35, 50]:
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = 3, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small
    
    # working input data
    # removing more time dependent vars and a few that provide potentially
    # redundant information
    data_in = df_train_Exst.drop(
        columns = ['STAID', 
                   'LAT_GAGE', 
                   'LNG_GAGE',
                   'NDAMS_2009',
                   'DDENS_2009',
                   'STOR_NID_2009',
                   'MINING92_PCT',
                   'POWER_SUM_MW',
                   'FRAGUN_BASIN',
                   'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief
    
    # sns.pairplot(data_in.iloc[:, 0:4])
    
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
    
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    # save as dataframe
    df_embedding = pd.DataFrame.from_dict({
        #'STAID': df_train_Exst['STAID'],
        'Emb1': embedding[:, 0],
        'Emb2': embedding[:, 1],
        'Emb3': embedding[:, 2],
        'Ecoregion': df_train_id['AggEcoregion'],
        'ColSize' : np.repeat(0.1, len(embedding[:, 0]))
        }).reset_index().drop(columns = ["index"])
    
    df_embedding['STAID'] = df_train_Exst['STAID']
    
    # Plot embedding
    
    fig = px.scatter_3d(df_embedding,
                        x = 'Emb1',
                        y = 'Emb2',
                        z = 'Emb3',
                        color = 'Ecoregion',
                        size = 'ColSize',
                        size_max = 10,
                        title = f"nn: {nn}",
                        custom_data = ["STAID"]
                        )
    # Edit so hover shows station id
    fig.update_traces(
    hovertemplate="<br>".join([
        "STAID: %{customdata[0]}" 
    ])
    )
    
    fig.show()


# %% UMAP on gages-ts vars

for nn in [15]: #, 25, 35, 50]:
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = 3, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small

    # working input data
    # removing more time dependent vars and a few that provide potentially
    # redundant information
    data_in = df_train_Exts_mn.drop(
        columns = ["STAID",
                   "TS_AG4344_SUM",
                   "TS_AG4346_SUM",
                   "TS_DEV_SUM",
                   "TS_SEMIDEV_SUM",
                   #"TS_HDEN",
                   #"TS_PDEN",
                   #"TS_hcrop",
                   #"TS_imperv",
                   #"TS_irrig",
                   #"TS_wu"
                   ])
    
    # sns.pairplot(data_in.iloc[:, 0:4])
    
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
    
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    # save as dataframe
    df_embedding = pd.DataFrame.from_dict({
        #'STAID': df_train_Exst['STAID'],
        'Emb1': embedding[:, 0],
        'Emb2': embedding[:, 1],
        'Emb3': embedding[:, 2],
        'Ecoregion': df_train_id['AggEcoregion'],
        'Class': df_train_id['Class'],
        'ColSize' : np.repeat(0.1, len(embedding[:, 0]))
        }).reset_index()
    
    df_embedding['STAID'] = df_train_Exst['STAID']
    
    # Plot embedding
    
    fig = px.scatter_3d(df_embedding,
                        x = 'Emb1',
                        y = 'Emb2',
                        z = 'Emb3',
                        color = 'Ecoregion',
                        size = 'ColSize',
                        size_max = 10,
                        title = f"nn = {nn}",
                        custom_data = ["STAID"]
                        )
    # Edit so hover shows station id
    fig.update_traces(
    hovertemplate="<br>".join([
        "STAID: %{customdata[0]}" 
    ])
    )
    
    
    fig.show()
    

#%% UMAP on all gages data: gages-ts and gages-static vars

for nn in [15]:
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = 3, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small

    # working input data
    # removing more time dependent vars and a few that provide potentially
    # redundant information
    data_in = df_train_Exall.drop(
        columns = ["STAID",
                   "TS_AG4344_SUM", # gages-ts vars
                   "TS_AG4346_SUM",
                   "TS_DEV_SUM",
                   "TS_SEMIDEV_SUM",
                   #"TS_HDEN",
                   #"TS_PDEN",
                   #"TS_hcrop",
                   #"TS_imperv",
                   #"TS_irrig",
                   #"TS_wu",
                   'LAT_GAGE', # gages-static vars
                   'LNG_GAGE',
                   'NDAMS_2009',
                   'DDENS_2009',
                   'STOR_NID_2009',
                   'MINING92_PCT',
                   'POWER_SUM_MW',
                   'FRAGUN_BASIN',
                   'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief
    
    # sns.pairplot(data_in.iloc[:, 0:4])
    
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
    
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    # save as dataframe
    df_embedding = pd.DataFrame.from_dict({
        #'STAID': df_train_Exst['STAID'],
        'Emb1': embedding[:, 0],
        'Emb2': embedding[:, 1],
        'Emb3': embedding[:, 2],
        'Ecoregion': df_train_id['AggEcoregion'],
        'Class': df_train_id['Class'],
        'ColSize' : np.repeat(0.1, len(embedding[:, 0]))
        }).reset_index()
    
    df_embedding['STAID'] = df_train_Exst['STAID']
    
    # Plot embedding
    
    fig = px.scatter_3d(df_embedding,
                        x = 'Emb1',
                        y = 'Emb2',
                        z = 'Emb3',
                        color = 'Ecoregion',
                        size = 'ColSize',
                        size_max = 10,
                        title = f"nn = {nn}",
                        custom_data = ["STAID"]
                        )
    # Edit so hover shows station id
    fig.update_traces(
    hovertemplate="<br>".join([
        "STAID: %{customdata[0]}" 
    ])
    )
    
    
    fig.show()
    
    
#%% HBDSCAN on static vars

# instantiate a hdbscan class object
clusterer = hdbscan.HDBSCAN(min_cluster_size = 20,
                            gen_min_span_tree = True,
                            cluster_selection_method = 'leaf')

 # working input data
 # removing more time dependent vars and a few that provide potentially
 # redundant information
data_in = df_train_Exst.drop(
    columns = ['STAID', 
               'LAT_GAGE', 
               'LNG_GAGE',
               'NDAMS_2009',
               'DDENS_2009',
               'STOR_NID_2009',
               'MINING92_PCT',
               'POWER_SUM_MW',
               'FRAGUN_BASIN',
               'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief

# sns.pairplot(data_in.iloc[:, 0:4])

# Scale data to z-scores (number of stdevs from mean)
data_in_scld = StandardScaler().fit_transform(data_in)

# cluster data
clusterer.fit(data_in_scld)

# plot the spanning tree
clusterer.minimum_spanning_tree_.plot(edge_cmap = 'viridis',
                                     edge_alpha = 0.6,
                                     node_size = 1,
                                     edge_linewidth = 0.1)

# plot cluster heirarchy
#clusterer.single_linkage_tree_.plot(cmap = 'viridis',
#                                    colorbar = True)
# condensed tree
# clusterer.condensed_tree_.plot()
clusterer.condensed_tree_.plot(select_clusters = True,
                               selection_palette = sns.color_palette())


# look at outliers
# sns.distplot(clusterer.outlier_scores_,
#             rug = True)

# get 90th quantile of outlier scores
# pd.Series(clusterer.outlier_scores_).quantile(0.9)

clst_lbls = np.unique(clusterer.labels_)

for i in clst_lbls:
    temp_sum = sum(clusterer.labels_ == i)
    print(f'cluster {i}: sum(clst_lbls == {temp_sum})')
    
# histogram of cluster probabilities (how strongly staid fits into cluster)
sns.distplot(clusterer.probabilities_,
             rug = True)


#%% HDBSCAN on UMAP - reduced GAGES-static data

# empty vectors to receive number of umap components, minimum cluster size, 
# relative validity (DBCV), number not clustered, and number of clusters 
nmb_cmpnts = []
min_clust = []
rel_val = []
ncl = []
nmbrcl = []

# Minimum number of clusters to try
mnclst_in = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100, 150, 200]
ncmp_in = [3, 5, 10, 15]

# # of clusters (measure with relative validity & Number unclustered)
# for nn in [15]: #, 25, 35, 50]:
for ncmp in ncmp_in:
    
    ##### UMAP
    
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = 15, #nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = ncmp, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small
    
     # working input data
     # removing more time dependent vars and a few that provide potentially
     # redundant information
    data_in = df_train_Exst.drop(
        columns = ['STAID', 
                   'LAT_GAGE', 
                   'LNG_GAGE',
                   'NDAMS_2009',
                   'DDENS_2009',
                   'STOR_NID_2009',
                   'MINING92_PCT',
                   'POWER_SUM_MW',
                   'FRAGUN_BASIN',
                   'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief
    
    # sns.pairplot(data_in.iloc[:, 0:4])
    
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
    
    # sns.pairplot(data_in.iloc[:, 0:4])
    
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    
    ##### HDBSCAN

    # try different minimum cluster sizes and log performance metrics
    for mnclst in mnclst_in:

        # instantiate a hdbscan class object
        clusterer = hdbscan.HDBSCAN(min_cluster_size = mnclst,
                                    gen_min_span_tree = True,
                                    cluster_selection_method = 'eom') #'leaf')
        
        
        # cluster data
        clusterer.fit(embedding)

        clst_lbls = np.unique(clusterer.labels_)

        for i in clst_lbls:
            temp_sum = sum(clusterer.labels_ == i)
            print(f'cluster {i}: sum(clst_lbls == {temp_sum})')
        
        # update performance metrics and outputs
        nmb_cmpnts.append(
            ncmp
        )

        min_clust.append(
            mnclst
        )

        rel_val.append(
            clusterer.relative_validity_
            )
        
        ncl.append(
            sum(clusterer.labels_ == -1)
            )
        
        nmbrcl.append(
            len(np.unique(clusterer.labels_)) - 1
            )
        
        
out_data = pd.DataFrame({
    'Number_UMAP_Components': nmb_cmpnts, 
    'MinClustSize': min_clust,
    'Relative_Validity': rel_val,
    'NotClustered': ncl,
    'NumberCluster': nmbrcl
    })

# print(out_data)

# write summary data to csv
out_data.to_csv(f'{dir_expl}/umap_hdbscan_static_summary.csv',
index=False)

#%% HDBSCAN on UMAP - reduced GAGES-ts data

# empty vectors to receive number of umap components, minimum cluster size, 
# relative validity (DBCV), number not clustered, and number of clusters 
nmb_cmpnts = []
min_clust = []
rel_val = []
ncl = []
nmbrcl = []

# Minimum number of clusters to try
mnclst_in = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100, 150, 200]
ncmp_in = [3, 5, 10, 15]

# # of clusters (measure with relative validity & Number unclustered)
# for nn in [15]: #, 25, 35, 50]:
for ncmp in ncmp_in:
    
    ##### UMAP
    
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = 15, #nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = ncmp, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small
    
     # working input data
    # removing more time dependent vars and a few that provide potentially
    # redundant information
    data_in = df_train_Exts_mn.drop(
        columns = ["STAID",
                   "TS_AG4344_SUM",
                   "TS_AG4346_SUM",
                   "TS_DEV_SUM",
                   "TS_SEMIDEV_SUM",
                   #"TS_HDEN",
                   #"TS_PDEN",
                   #"TS_hcrop",
                   #"TS_imperv",
                   #"TS_irrig",
                   #"TS_wu"
                   ])
    
   
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
       
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    
    ##### HDBSCAN

    # try different minimum cluster sizes and log performance metrics
    for mnclst in mnclst_in:

        # instantiate a hdbscan class object
        clusterer = hdbscan.HDBSCAN(min_cluster_size = mnclst,
                                    gen_min_span_tree = True,
                                    cluster_selection_method = 'eom') #'leaf')
        
        
        # cluster data
        clusterer.fit(embedding)

        clst_lbls = np.unique(clusterer.labels_)

        for i in clst_lbls:
            temp_sum = sum(clusterer.labels_ == i)
            print(f'cluster {i}: sum(clst_lbls == {temp_sum})')
        
        # update performance metrics and outputs
        nmb_cmpnts.append(
            ncmp
        )

        min_clust.append(
            mnclst
        )

        rel_val.append(
            clusterer.relative_validity_
            )
        
        ncl.append(
            sum(clusterer.labels_ == -1)
            )
        
        nmbrcl.append(
            len(np.unique(clusterer.labels_)) - 1
            )
        
        
out_data_ts = pd.DataFrame({
    'Number_UMAP_Components': nmb_cmpnts, 
    'MinClustSize': min_clust,
    'Relative_Validity': rel_val,
    'NotClustered': ncl,
    'NumberCluster': nmbrcl
    })

# print(out_data)

# write summary data to csv
out_data_ts.to_csv(f'{dir_expl}/umap_hdbscan_ts_summary.csv',
index=False)


out_data_ts.sort_values(by = ["Relative_Validity"], ascending = False)







#%% HDBSCAN on UMAP - reduced GAGES-all data

# empty vectors to receive number of umap components, minimum cluster size, 
# relative validity (DBCV), number not clustered, and number of clusters 
nmb_cmpnts = []
min_clust = []
rel_val = []
ncl = []
nmbrcl = []

# Minimum number of clusters to try
mnclst_in = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100, 150, 200]
ncmp_in = [3, 5, 10, 15]

# # of clusters (measure with relative validity & Number unclustered)
# for nn in [15]: #, 25, 35, 50]:
for ncmp in ncmp_in:
    
    ##### UMAP
    
    # instantiate a umap class object
    reducer = umap.UMAP(n_neighbors = 15, #nn, # 2 to 100
                        min_dist = 0.1, # 
                        spread= 1,
                        n_components = ncmp, # 2 to 100
                        random_state = 100,
                        n_epochs = 200) # 200 for large datasets; 500 small
    
     # working input data
    # removing more time dependent vars and a few that provide potentially
    # redundant information
    data_in = df_train_Exall.drop(
        columns = ["STAID",
                   "TS_AG4344_SUM", # gages-ts vars
                   "TS_AG4346_SUM",
                   "TS_DEV_SUM",
                   "TS_SEMIDEV_SUM",
                   #"TS_HDEN",
                   #"TS_PDEN",
                   #"TS_hcrop",
                   #"TS_imperv",
                   #"TS_irrig",
                   #"TS_wu",
                   'LAT_GAGE', # gages-static vars
                   'LNG_GAGE',
                   'NDAMS_2009',
                   'DDENS_2009',
                   'STOR_NID_2009',
                   'MINING92_PCT',
                   'POWER_SUM_MW',
                   'FRAGUN_BASIN',
                   'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief
    
   
    # Scale data to z-scores (number of stdevs from mean)
    data_in_scld = StandardScaler().fit_transform(data_in)
       
    # train reducer
    embedding = reducer.fit_transform(data_in_scld)
    embedding.shape
    
    
    ##### HDBSCAN

    # try different minimum cluster sizes and log performance metrics
    for mnclst in mnclst_in:

        # instantiate a hdbscan class object
        clusterer = hdbscan.HDBSCAN(min_cluster_size = mnclst,
                                    gen_min_span_tree = True,
                                    cluster_selection_method = 'eom') #'leaf')
        
        
        # cluster data
        clusterer.fit(embedding)

        clst_lbls = np.unique(clusterer.labels_)

        for i in clst_lbls:
            temp_sum = sum(clusterer.labels_ == i)
            print(f'cluster {i}: sum(clst_lbls == {temp_sum})')
        
        # update performance metrics and outputs
        nmb_cmpnts.append(
            ncmp
        )

        min_clust.append(
            mnclst
        )

        rel_val.append(
            clusterer.relative_validity_
            )
        
        ncl.append(
            sum(clusterer.labels_ == -1)
            )
        
        nmbrcl.append(
            len(np.unique(clusterer.labels_)) - 1
            )
        
        
out_data_all = pd.DataFrame({
    'Number_UMAP_Components': nmb_cmpnts, 
    'MinClustSize': min_clust,
    'Relative_Validity': rel_val,
    'NotClustered': ncl,
    'NumberCluster': nmbrcl
    })

# print(out_data)

# write summary data to csv
out_data_all.to_csv(f'{dir_expl}/umap_hdbscan_all_summary.csv',
index=False)


out_data_all.sort_values(by = ["Relative_Validity"], ascending = False)




# %%
# Regression on umap-hdbscan-all expl vars mean discharge
data_in = pd.merge(
    df_train_Exall,
    df_train_mnanWY,
    left_on = 'STAID',
    right_on = 'site_no'
).drop(
    columns = ["STAID",
                "site_no",
                "Ann_WY_ft3",
                "TS_AG4344_SUM", # gages-ts vars
                "TS_AG4346_SUM",
                "TS_DEV_SUM",
                "TS_SEMIDEV_SUM",
                #"TS_HDEN",
                #"TS_PDEN",
                #"TS_hcrop",
                #"TS_imperv",
                #"TS_irrig",
                #"TS_wu",
                'LAT_GAGE', # gages-static vars
                'LNG_GAGE',
                'NDAMS_2009',
                'DDENS_2009',
                'STOR_NID_2009',
                'MINING92_PCT',
                'POWER_SUM_MW',
                'FRAGUN_BASIN',
                'RRMEAN']) #RR = relief ratio (ELEV_Mean-ELEV_min)/total relief


cor_out = np.corrcoef(data_in.values.T)
hm = heatmap(cor_out, 
    row_names = data_in.columns, 
    column_names = data_in.columns,
    figsize = (20, 20),
    cell_font_size = 5)

plt.tight_layout()
plt.show()


# %% 
# Plot umap-hdbscan results
##### Plot



# save embedding as dataframe
df_embedding = pd.DataFrame.from_dict({
    #'STAID': df_train_Exst['STAID'],
    'Emb1': embedding[:, 0],
    'Emb2': embedding[:, 1],
    'Emb3': embedding[:, 2],
    'Emb4': embedding[:, 3],
    'Emb5': embedding[:, 4],
    'Emb6': embedding[:, 5],
    'Emb7': embedding[:, 6],
    'Emb8': embedding[:, 7],
    'Emb9': embedding[:, 8],
    'Emb10': embedding[:, 9],
    'Ecoregion': df_train_id['AggEcoregion'],
    'Class': df_train_id['Class'],
    'Cluster': clusterer.labels_.astype(str),
    'ColSize' : np.repeat(0.1, len(embedding[:, 0]))
    }).reset_index()

df_embedding['STAID'] = df_train_Exst['STAID']

# Plot embedding
fig = px.scatter_3d(df_embedding,
                    x = 'Emb1',
                    y = 'Emb2',
                    z = 'Emb3',
                    color = 'Cluster', # 'Ecoregion', #
                    size = 'ColSize',
                    size_max = 10,
                    title = f"nn = {nn}",
                    custom_data = ["STAID"]
                    )
# Edit so hover shows station id
fig.update_traces(
hovertemplate="<br>".join([
    "STAID: %{customdata[0]}" 
])
)



    
# %%
