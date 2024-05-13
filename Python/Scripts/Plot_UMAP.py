'''
BChoat 2024/05/08

Generate embeddings for uamp models that
have been identified, and create plots.
'''

# %% load libraries
#################################
from GAGESii_Class import Clusterer
import pandas as pd
import numpy as np
import umap
# import hdbscan
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Load_Data import load_data_fun
import plotly.express as px
import os
# from itertools import product
from sklearn.decomposition import PCA # dimension reduction



# %%
# define variables and load data
############
# directory with data to work with
# dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 
dir_work = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                    'HPC_Files/GAGES_Work'

# directory where to place outputs
# dir_umaphd = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN'
dir_umaphd = 'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/'\
                    'Data_Out/UMAP_HDBSCAN'

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

# should embeddings be saved?
save_emb = False

# which vars were clusters based on?
clust_vars = ['All'] # , 'Nat', 'Anth']#  ['All'] # ['Nat']


# %%
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

# define counter to be used in naming columns
cnt = 0

# keep only 5 parameter sets with best validity index scores
# cand_parms_all = cand_parms_all[0:1]
# cols = [col for col in df_trainID.columns if \
#         ('All' in col)  | ('Nat' in col) | ('Anth' in col)]
# cols_in = cols.copy()
# cols_in.extend(['STAID', 'partition'])
cols_in = ['STAID', 'partition']



# create partitionlabels for output
part_in = np.repeat('train', df_trainID.shape[0])
part_in = np.append(part_in, np.repeat('valnit', df_valnitID.shape[0]))

# append series for staid for output dataframe
staid_in = df_trainID['STAID'].append(df_valnitID['STAID']).reset_index(drop=True)


for clust_var in clust_vars:

    if clust_var == 'All':
        params_in = cand_parms_all
    elif clust_var == 'Nat':
        params_in = cand_parms_nat
    elif clust_var == 'Anth':
        params_in = cand_parms_anth
    else:
        print('invalid clust_var')

    for i, params in params_in[0:5].iterrows():
        print(params)

        if os.path.exists(f'{dir_umaphd}/UMAP_Embeddings_{clust_var}_{i}.csv') & \
                        (not save_emb):
            print('\n embedding file exists, reading it in.')
            allEmb_temp = pd.read_csv(
                f'{dir_umaphd}/UMAP_Embeddings_{clust_var}_{i}.csv'
                )

        else:

            print('computing umap')
    
            # labels = df_trainID[f'{col_in[i]}']

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

            
            # define pipline
            pipe = Pipeline(
                steps = [('scaler', scaler), 
                    ('umap_reducer', umap_reducer)]
                )

            # fit the model pipeline
            pipe.fit(df_trainexpl)   



            # scale, transform, and predict valnit data
            valnit_temp = scaler.transform(df_valnitexpl)
            valnit_temp = umap_reducer.transform(valnit_temp)

            allEmb_temp = pd.DataFrame(np.vstack([umap_reducer.embedding_,
                                valnit_temp]))
            allEmb_temp.columns = [f'Emb {x}' for x in allEmb_temp.columns]

            allEmb_temp['STAID'] = staid_in
            allEmb_temp['partition']  = part_in
            allEmb_temp['Cluster'] = df_trainID[f'{clust_var}_{i}'].append(
                df_valnitID[f'{clust_var}_{i}']).reset_index(drop=True)
            allEmb_temp['Cluster'] = allEmb_temp['Cluster'].astype(str)
            allEmb_temp = allEmb_temp.sort_values(by = ['partition', 'Cluster'])


            if save_emb:
                allEmb_temp.to_csv(f'{dir_umaphd}/UMAP_Embeddings_{clust_var}_{i}.csv',
                                index = False)


        ## %%
        # Plot last embedding
        ##########

        # rename columns
        # df_plot.columns = [f'Emb {x}' for x in df_plot.columns]

        # get cluster labels
        # df_plot['clusters'] = train_labels.astype('str')
        
        # change order so clusters appear in order
        df_plot = allEmb_temp[allEmb_temp['partition'] == 'valnit']

        fig = px.scatter_3d(
            df_plot,
            x = 'Emb 0',
            y = 'Emb 1',
            z = 'Emb 2',
            color = 'Cluster',
            title = f'{clust_var}_{i}',
            custom_data = ['STAID'])
        fig.update_traces(marker_size = 5,
                        hovertemplate = "<br>".join([
                        "STAID: %{customdata[0]}"
                            ]))
        fontsize = 10
        fig.update_layout(template='presentation',
            #   legend_title_text='Operation<br>Mode',
            #   title = dict(text = f'Wind Rose: {years_in[0]}-{years_in[1]}',
                        #    font = dict(size=50)),
            margin=dict(t=50, r=0, b=50, l=0),
            scene=dict(
                xaxis=dict(
                    tickfont=dict(size=fontsize)  # Adjust the size of the x-axis tick labels
                    )
                ,
                yaxis=dict(
                    tickfont=dict(size=fontsize)  # Adjust the size of the y-axis tick labels
                    )
                ,
                zaxis=dict(
                    tickfont=dict(size=fontsize)  # Adjust the size of the z-axis tick labels
                    )
                )
            )
        
                #   legend = dict(font = dict(size=35))
                    

        fig.show()



        

        cnt += 1


# clusterer = Clusterer(clust_vars = df_plot.drop(['STAID', 'partition', 'Cluster'], axis = 1), 
#                       id_vars = df_plot['STAID'])
# # clusterer.clust_vars_tr_
# clusterer.stand_norm(method = 'standardize')
# # clusterer.df_pca_components_ 
# pca_out = clusterer.pca_reducer(nc = None,
#                         color_in = 'blue',
#                         plot_out = False)
# # cumulative explained variance
# cum_exp = np.cumsum(clusterer.pca_fit_.explained_variance_ratio_)
# # get most important embeddings for each component
# comp0 = pd.DataFrame({'Features': clusterer.pca_fit_.feature_names_in_,
#                         'abs_values': np.abs(clusterer.pca_fit_.components_[0])})
# comp0 = comp0.sort_values(by = 'abs_values', ascending = False)
# comp1 = pd.DataFrame({'Features': clusterer.pca_fit_.feature_names_in_,
#                         'abs_values': np.abs(clusterer.pca_fit_.components_[1])})
# comp1 = comp1.sort_values(by = 'abs_values', ascending = False)


# %% Define function to create 3d rotation plot 
#############################


# from https://community.plotly.com/t/rotating-3d-plots-with-plotly/34776


# Helix equation
# t = np.linspace(0, 10, 50)
# x, y, z = np.cos(t), np.sin(t), t


def animate3dScatter(df_in, x, y, z, color_in, labs_in, title_id):
    '''
    parameters:
    ------------------------------
    df_in (pandas dataframe): dataframe containing 3 columns, one for values in each 
        of the three directions for the scatter
    x, y, z (strings): names of columns in df_in to plot
    color_in (str): name of column in df_in with values to be used for 
        coloring points
    labs_in (str): name of column in df_in with values to show in hover pop up
    title_id (str): used in title of plot (as part of title, not entire title)
    '''
    import plotly.graph_objects as go
    import numpy as np

    # define colormap to be used (should match cluster plots)
    cmap_in = ['black', 'orange', 'blue', 'purple', 'brown', 
            'gray', 'dodgerblue',  'lightcoral', 'darkkhaki', 'lime', 'cyan', 
            'red', 'slateblue', 'pink', 'indigo', 'maroon', 'chocolate', 'teal',
            'yellowgreen', 'silver', 'yellow', 'darkgoldenrod', 'deeppink',
            'lightgreen', 'peru', 'crimson', 'saddlebrown', 'green']


    # sort values so clusters and colors align
    df_in = df_in.sort_values(by = color_in)
    # color_in = df_in[color_in].unique().values
    

    fig= go.Figure()

    for j in range(len(df_in[color_in].unique())):
        col_temp = df_in[color_in].unique()[j]
        
        print(f'col_temp: {col_temp}')
        print(f'x: {x}')
        df_temp = df_in[df_in[color_in] == col_temp]
        # .query("@color_in == @col_temp")

        # color_in = df_in[color_in].values
        
        color_in_temp = df_temp[color_in]
        cmap_temp = cmap_in[j]
        x_temp = df_temp[x]
        y_temp = df_temp[y]
        z_temp = df_temp[z]
        labs_in_temp = df_temp[labs_in].values
    
        fig.add_trace(go.Scatter3d(x=x_temp, y=y_temp, z=z_temp, 
                                    mode='markers',
                                    marker=dict(
                                        color=cmap_temp,# col_temp,
                                        # colorscale='Viridis',
                                        size=8,
                                        # colorbar=dict(thickness=20)
                                    ),
                                    customdata=np.stack([labs_in_temp, color_in_temp], axis=1),
                                    name=str(col_temp)))
    
    fig.update_traces(hovertemplate = "<br>".join([
                        "STAID: %{customdata[0]}"]) +
                        '<br>' + 
                            '<br>'.join([
                                'Cluster: %{customdata[1]}']) +
                            '<extra></extra>'
                            )
    

    x_eye = -1 # -1.25
    y_eye = 2
    z_eye = 1
    

    fig.update_layout(
            title=f'UMAP-HDBSCAN Clusters<br>Based on {title_id}',
            template='presentation',
            width=1000,
            height=700,
            scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
            updatemenus=[dict(type='buttons',
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=25, r=10),
                    buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=5, redraw=True), 
                                                                transition=dict(duration=0),
                                                                fromcurrent=True,
                                                                mode='immediate'
                                                                )]
                                                )
                                        ]
                                )
                            ],
            margin=dict(t=30, r=0, b=50, l=0),
            scene=dict(
                xaxis=dict(
                    tickfont=dict(size=fontsize),  # Adjust the size of the x-axis tick labels
                    title = 'Embedding 1'
                    )
                ,
                yaxis=dict(
                    tickfont=dict(size=fontsize),  # Adjust the size of the y-axis tick labels
                    title = 'Embedding 2'
                    )
                ,
                zaxis=dict(
                    tickfont=dict(size=fontsize),  # Adjust the size of the z-axis tick labels
                    title = 'Embedding 3'
                    )
                ),
                legend=dict(
                    x=0.8,
                    y=0.5,
                    traceorder='normal',
                    bgcolor='rgba(255, 255, 255, 0.5)',
                    # bordercolor='rgba(0, 0, 0, 0.5)',
                    # borderwidth=2
                )
                        

    ),

    # fig.update_layout(coloraxis=dict(
    #         colorbar=dict(
    #             title='Color Scale')))  # Add colorbar
    
    
    


    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    frames=[]
    for t in np.arange(0, 6.26, 0.1):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames=frames

    fig.show()
    fig.write_html('C:/Python/test.html')

    # return (fig)


# MAKE PLOT
animate3dScatter(df_plot, 
                'Emb 1', 'Emb 2', 'Emb 3',
                 'Cluster', 
                 'STAID', 
                 f'{clust_var} variables ({i})')