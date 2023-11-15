'''
BChoat 2023/11/12 gagesiienv

Script to add units to feature_categories
'''


# %% import libs
###############################

import pandas as pd




# %% define dirs, vars, read in data
###############################



# files/dirs
feats_cats_file = 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
ggs_file = 'D:/DataWorking/GAGESii/basinchar_and_report_sept_2011/gagesII_sept30_2011_var_desc.xlsx'


# read in data
df_featcats = pd.read_csv(feats_cats_file)
df_ggs = pd.read_excel(ggs_file)


# %% combine and wirte out
#########################################

df_out = pd.merge(
    df_featcats,
    df_ggs[['VARIABLE_NAME', 'UNITS (numeric values)']],
    left_on = 'Features', right_on = 'VARIABLE_NAME',
    how = 'left'
)

# make any desired edits
# NOTE: Expects a column in feature_cats to be 'Units' and for climate-related variables
#   and gagesii-ts variables to already have units assigned in that column

# replace gagesii units with already assigned units
df_out.loc[(df_out['VARIABLE_NAME'].isna()) & ~df_out['Units'].isna(), \
       'UNITS (numeric values)'] = df_out.loc[(df_out['VARIABLE_NAME'].isna()) & \
                                              ~df_out['Units'].isna(), \
                                                'Units']

# df_out.tail(30)
# df_out['UNITS (numeric values)'].unique()
df_out.loc[
    df_out['UNITS (numeric values)'].isna(), 'UNITS (numeric values)'
    ] = '-'

# clean up a bit
df_out['Units'] = df_out['UNITS (numeric values)']
df_out = df_out.drop(['UNITS (numeric values)', 'VARIABLE_NAME'], axis = 1)

# save
df_out.to_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories_wUnits.csv',
    index = False
)
# %%
