'''

2023/06/22 BChoat

Calcualte adjusted mutual information between groups of clusters results
from each clustering group.

'''

# %% import libraries
#########################################


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

        

# %% define directories, variables, and such


# directory from which to read in data
dir_datain = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/'\
                'GAGES_Work/data_work/GAGESiiVariables'

# directory where to place figure
dir_figout = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'


# the heatmap figs should be written to file (True or False)?
write_fig = False
# if write_fig = True, then provide filenames for the output files
file_trainout = f'{dir_figout}/AMI_train.png' 
file_testout = f'{dir_figout}/AMI_valnit.png'

# define list of columns to use when calculating ami scores
cols_in = [
    'Class',
    'AggEcoregion', 
    # 'ECO3_Site', 
    # 'USDA_LRR_Site', 
    'CAMELS', 
    'HLR', 
    'All_0', 'All_1', 'All_2',
    'Nat_0', 'Nat_1', 'Nat_2', 'Nat_3', 'Nat_4', 
    'Anth_0', 'Anth_1'
]



# %% read in and pre-preprocess data as needed
######################################


df_trainid = pd.read_csv(f'{dir_datain}/ID_train.csv',
                         dtype = {'STAID': 'string'})

df_testid = pd.read_csv(f'{dir_datain}/ID_valnit.csv',
                         dtype = {'STAID': 'string'})


# %% calculate ami between all clustering approaches and create matrix
#############################


# first for training data
#########

# create an empty dataframe to hold results
df_trainami = pd.DataFrame(columns = cols_in, index = cols_in)

# loop through all combinations of two dataframe names (i.e., cluster results)
for c1 in df_trainami.columns:
    for c2 in df_trainami.index:
        print(c1, c2)

        # calc adjusted mutual information index
        # AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
        ami_temp = adjusted_mutual_info_score(df_trainid[c1], df_trainid[c2])
        
        # add ami_tempto df_trainami
        df_trainami.loc[c1, c2] = ami_temp

# convert df_trainami dtype to float for heatmap
df_trainami = df_trainami.astype(float)


# next for testing data
#########

# create an empty dataframe to hold results
df_testami = pd.DataFrame(columns = cols_in, index = cols_in)

# loop through all combinations of two dataframe names (i.e., cluster results)
for c1 in df_testami.columns:
    for c2 in df_testami.index:
        print(c1, c2)

        # calc adjusted mutual information index
        # AMI(U, V) = [MI(U, V) - E(MI(U, V))] / [avg(H(U), H(V)) - E(MI(U, V))]
        ami_temp = adjusted_mutual_info_score(df_testid[c1], df_testid[c2])
        
        # add ami_tempto df_testami
        df_testami.loc[c1, c2] = ami_temp

# convert df_testami dtype to float for heatmap
df_testami = df_testami.astype(float)




# %% plot heatmap of ami scores
######################################

# first training
############################

# plot heatmap
# create mask to show inly lower tirnagular portion
mask = np.tri(*df_trainami.shape, k = -1).T

# Plot the correlogram heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df_trainami, 
            mask = mask,
            annot = True, # show ami value
            fmt = '.2f', # '.2f'two decimal places
            cmap = 'seismic',
            vmin = 0,
            vmax = 1,
            annot_kws = {'fontsize': 10})

# make cbar labels larger
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize = 12)
cbar.set_label('Adjusted Mutual Information', fontsize = 12)


# make tick labels larger
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.title('Adjusted Mutual Information Between Clustering'\
            'Approaches\n(Training Data)', fontsize = 16)


if write_fig:
    plt.savefig(file_trainout,
                bbox_inches = 'tight',
                dpi = 300)
else:
    plt.show()



# next testing
############################

# plot heatmap
# create mask to show inly lower tirnagular portion
mask = np.tri(*df_testami.shape, k = -1).T

# Plot the correlogram heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df_testami, 
            mask = mask,
            annot = True, # show ami value
            fmt = '.2f', # '.2f'two decimal places
            cmap = 'seismic',
            vmin = 0,
            vmax = 1,
            annot_kws = {'fontsize': 10})

# make cbar labels larger
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize = 12)
cbar.set_label('Adjusted Mutual Information', fontsize = 12)


# make tick labels larger
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.title('Adjusted Mutual Information Between Clustering'\
            'Approaches\n(Testing Data)', fontsize = 16)


if write_fig:
    plt.savefig(file_testout,
                bbox_inches = 'tight',
                dpi = 300)
else:
    plt.show()





# %%
#