'''
BChoat 2022/11/11
Script to investigate trees from XGBoost models
'''


# %% 
# Import Libraries
###########

import pandas as pd
import numpy as np
from Load_Data import load_data_fun
import glob
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from itertools import product
import os



# from xgboost import plot_tree # can't get to work on 
# windows (try on linux)


#################### To-Do #############################
# get aliases from FeaturesCategories.csv
# xxxxxxxxxxxloop through all regions/cluster methods(?)
# add string labels to peaks in plots
# use alias name from feature_categories 
# plot and save plot only if in top n most important features
# write distance threshold, prominence_threshold, height to data frame
# write same parameters to 
# add units to feature_categories.csv and include them in figs/tables
#    units for features uesd in splits



# %%
# define variables
############

# AggEcoregion regions:
# CntlPlains, EastHghlnds, MxWdShld, NorthEast, SECstPlain, SEPlains, 
# WestMnts, WestPlains, WestXeric 
# define which region to work with
#`` region =   'EastHghlnds' # 'SECstPlain' # 'WestMnts' # 'All' # 'CntlPlains' # 'Non-ref' # 'All'
             
# define time scale working with. This vcombtrainariable will be used to read and
# write data from and to the correct directories
timescales = ['monthly', 'annual', 'mean_annual']
# time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly', 'daily'

# directory with data to work with
dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work' 
# dir_work = '/media/bchoat/Local Disk/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/'

# name to save output peaks dataframe to
file_out = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results/Peaks.csv'

# directory in which to save figs, if created I in second half
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures/Thresholds'

# csv with aliases and categories
feat_cats = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
)

# csv with general info about catchments
df_ID = pd.read_csv(
    'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_work/' \
        'GAGESiiVariables/ID_train.csv'
)


# %%
# if file_out does not exist, then loop through and creat it and save it
# WILL TAKE SEVERAL HOURS
################################

if not os.path.exists(file_out):

    print(f'{file_out} does not exist\n\n')
        
    # define which clustering method is being combined. This variable 
    # will be used for collecting data from the appropriate directory as well as
    # naming the combined file
    # clust_meth =  'AggEcoregion' # 'Class' # 'None' # 'AggEcoregion', 'None', 
    clust_meths = ['None', 'Class', 'AggEcoregion', 
            'All_0', 'All_1', 'All_2', 'Anth_0', 'Anth_1', 
            'CAMELS', 'HLR', 'Nat_0', 'Nat_1', 'Nat_2',
            'Nat_3', 'Nat_4']

    # define dict to hold cluster methods: cluster
    df_regions = {}

    # define regions for different clustering methods
    for cl in clust_meths:
        if cl == 'None':
            df_regions[cl] = 'All'
        else:
            df_regions[cl] = np.sort(df_ID[cl].unique())

    # define lists to hold outputs
    clust_methout = []
    region_out = []
    timescale_out = []
    voi_array = []
    peaks_out = []
    peak_gains = []
    peak_heights = []
    prominences = []
    left_bases = []
    right_bases = []


    # %%
    # Loop through data and find peaks
    ##########

    for cl, time_scale in product(clust_meths, timescales):
        
        # if reg string, thm use whole thing
        regions = df_regions[cl]
        if isinstance(regions, str):
            regions = [regions]
            
        for reg in regions:

            print(f'\nclust-method: {cl}; region: {reg}\n')

            
            # first define xgbreg object
            model = xgb.XGBRegressor()

            # define temp time_scale var since model names do not have '_' in them
            temp_time = time_scale.replace('_', '')

            # reload model into object
            model.load_model(
                f'{dir_work}/data_out/{time_scale}'
                f'/Models/XGBoost_{temp_time}_{cl}_{reg}_model.json'
                )
            
            
            # loop through variables of interest and find peaks
            for voi in model.get_booster().feature_names:


    # # %%
    # Write tree to dataframe and Return branches with specified variable
    ###################

                # print trees to dataframe
                df_tree = model.get_booster().trees_to_dataframe()

                # remove '_x' from DRAIN_SQKM feature label
                df_tree['Feature'] = df_tree['Feature'].str.replace('_x', '')

                # subset to shallower branches
                # df_tree = df_tree[
                #     (df_tree['ID'].str.split('-').str[1] == '0') | \
                #     (df_tree['ID'].str.split('-').str[1] == '1')   
                #     # df_tree['ID'].str.contains('|'.join(['-0', '-1'])).tolist()
                # ]

                df_treeGain = df_tree.groupby(['Feature', 'Split'])['Gain'].sum().reset_index()
                df_treeGain['Feat_Split'] = \
                    df_treeGain['Feature'] + '-' + df_treeGain['Split'].round(2).astype(str)
                df_treeGain.sort_values(by = 'Gain', ascending = False).head(20)
                # drop 'leaaf' feature
                df_tree = df_tree[df_tree['Feature'] != 'Leaf']




                # # subset trees to nodes where voi appears
                df_tree_voi = df_tree[df_tree['Feature'] == voi]
                # df_tree_voi
                # df_tree_voi.sort_values(by = 'Gain', ascending = False)
                # df_tree_voi.sort_values(by = 'Tree')

                # # print number of times each unique value of the voi was uesd for splitting
                # # df_tree_voi.loc[df_tree_voi['Feature'] == 'DRAIN_SQKM', 'Split'].value_counts().head(20)

                # sum total gain by each value of area used for splitting nodes
                df_splitGain = df_tree_voi.groupby('Split').sum('Gain').sort_values(
                    by = 'Gain', ascending = False
                    ).head(20)[['Gain']]
                df_splitGain = df_splitGain.reset_index()


                # %% find peaks and plot
                #############################

                # define data
                # Compute the histogram


                df_plot = df_tree[
                    df_tree['Feature'] == voi
                    # df_tree['Feature'].str.contains(voi)
                    ].sort_values(by = 'Split').reset_index(drop = True)

                df_plot = df_plot.groupby(['Split'])['Gain'].sum().reset_index()

                x_temp = df_plot['Split']
                y_temp = df_plot['Gain']

                if y_temp.sum() <= 50000:
                    # print update or something
                    continue

                # if only one value then handle since peaks can't be found
                if len(x_temp) == 1:
                    # update outputs
                    clust_methout = np.append(clust_methout, cl)
                    region_out = np.append(region_out, reg)
                    timescale_out = np.append(timescale_out, time_scale)
                    voi_array = np.append(voi_array, voi)
                    peaks_out = np.append(peaks_out, x_temp[0])
                    peak_gains = np.append(peak_gains, y_temp[0])
                    peak_heights = np.append(peak_heights, 'NA')
                    prominences = np.append(prominences, 'NA')
                    left_bases = np.append(left_bases, 'NA')
                    right_bases = np.append(right_bases, 'NA')

                    continue

                # Sum y-values for ranges of 2 on the x-axis
                x = []
                y = []
                # if x_temp.max() > 
                range_len = np.round(x_temp.max()/100, 1) # length of values to sum over
                if range_len == 0: range_len = np.round(x_temp.max()/100, 4)
                if range_len == 0: range_len = np.round(x_temp.max()/100, 6)
                # if neither of these worked, then assume the x-split is too tiny
                # to be practical and comtinue
                if range_len == 0: continue

                for i in np.arange(0, np.ceil(x_temp.max() + range_len), range_len):

                    y.append(sum(y_temp[(x_temp >= i) & (x_temp < i+range_len)]))
                    x.append((i+(i+range_len))/2)
                print(x, y)

                # plt.bar(x, y)
                # plt.xticks(rotation=90)


                # calc distance threshold and prominence_threshold
                y_range = np.max(y) - np.min(y)
                x_range = x_temp.max() - x_temp.min()
                distance_fraction = 0.0001  # e.g., 0.05% of data range
                prominence_fraction = 0.3  # e.g., 30% of data range
                height_in = 0.05 * y_temp.sum() # e.g., 15% total gain
                distance_threshold = x_range * distance_fraction
                if distance_threshold < 5: distance_threshold = 5
                prominence_threshold = y_range * prominence_fraction

                print(f'\n\ndistance_threshold: {distance_threshold}')
                print(f'\nprominence_threshold: {prominence_threshold}')
                print(f'\nheight: {height_in}')


                # mirror the first 10 values to enable first value to be identified as peak
                # how many values to add to front?
                prpndN = 8

                y = [y[i] for i in range(prpndN-1, 0, -1)] + list(y)

                peaks, peakMetrics = find_peaks(
                    # df_plot['Gain'], 
                    y,
                    distance = distance_threshold, 
                    prominence = prominence_threshold,
                    height = height_in # 5000 # peak must have at least this much gain
                    )
                
                # if no peaks, then continue
                if len(peaks) == 0: 
                    continue

                peaks = peaks - (prpndN-1)

                y = y[prpndN-1:len(y)]

                peakSplits = np.array(x)[peaks]

                peakGain = np.array(y)[peaks]

                # update outputs
                clust_methout = np.append(
                    clust_methout,
                    np.repeat(cl, len(peakMetrics['peak_heights']))
                )
                region_out = np.append(
                    region_out,
                    np.repeat(reg, len(peakMetrics['peak_heights']))
                )
                timescale_out = np.append(
                    timescale_out,
                    np.repeat(time_scale, len(peakMetrics['peak_heights']))
                )
                voi_array = np.append(
                    voi_array,
                    np.repeat(voi, len(peakMetrics['peak_heights']))
                    )
                peaks_out = np.append(peaks_out, peakSplits)
                peak_gains = np.append(peak_gains, peakGain)
                peak_heights = np.append(peak_heights, peakMetrics['peak_heights'])
                prominences = np.append(prominences, peakMetrics['prominences'])
                left_bases = np.append(left_bases, peakMetrics['left_bases'])
                right_bases = np.append(right_bases, peakMetrics['right_bases'])

    # create output dataframe
    df_peaksOut = pd.DataFrame({
        'clust_meth': clust_methout,
        'region': region_out,
        'timescale': timescale_out,
        'variable': voi_array,
        'peaks': peaks_out,
        'peakGain': peak_gains,
        'peak_heights': peak_heights,
        'prominences': prominences,
        'left_base': left_bases,
        'right_bases': right_bases
    })

    df_peaksOut.to_csv(file_out, index = False)

else:
    df_peaksOut = pd.read_csv(file_out).drop('Unnamed: 0', axis = 1)



########################################################################
########################################################################
########################################################################
########################################################################


# %% some exploration
#################################
df_ex = df_peaksOut.copy()
df_ex = df_ex[df_ex['timescale'].isin(['monthly'])]
df_ex = df_ex[df_ex['clust_meth'] == 'None']
df_ex = df_ex[df_ex['region'] == 'All']
df_ex = df_ex.sort_values(by = 'peakGain', ascending = False)
df_ex.head(30)
df_ex.head(100).variable.unique()


# %%
if not os.path.exists(dir_figs):
    os.mkdir(dir_figs)
# subset data to df for plotting
ts = 'monthly'
cm = 'AggEcoregion'
reg = 'WestMnts'
# voi = 'BDAVE'
# voi = 'prcp'
# voi = 'prcp_1'
# voi = 'TS_WaterUse_wu'
# voi = 'TS_Housing_HDEN'
# voi = 'TS_NLCD_95' # emergent herbaceous wetlands
# voi = 'DRAIN_SQKM'
# voi = 'CANALS_PCT'
voi = 'STOR_NID_2009'
# df_peakPlot = df_peaksOut[
#     (df_peaksOut['timescale'] == ts) &
#     (df_peaksOut['clust_meth'] == cm) & 
#     (df_peaksOut['region'] == reg) & 
#     (df_peaksOut['variable'] == voi)
# ]

## %% find peaks and plot
############################

# define data
# Compute the histogram


# first define xgbreg object
model = xgb.XGBRegressor()

# define temp time_scale var since model names do not have '_' in them
temp_time = ts.replace('_', '')

# reload model into object
model.load_model(
    f'{dir_work}/data_out/{ts}'
    f'/Models/XGBoost_{temp_time}_{cm}_{reg}_model.json'
    )


# print trees to dataframe
df_tree = model.get_booster().trees_to_dataframe()

# remove '_x' from DRAIN_SQKM feature label
df_tree['Feature'] = df_tree['Feature'].str.replace('_x', '')


df_treeGain = df_tree.groupby(['Feature', 'Split'])['Gain'].sum().reset_index()
df_treeGain['Feat_Split'] = \
    df_treeGain['Feature'] + '-' + df_treeGain['Split'].round(2).astype(str)
df_treeGain.sort_values(by = 'Gain', ascending = False).head(20)
# drop 'leaaf' feature
df_tree = df_tree[df_tree['Feature'] != 'Leaf']


# # subset trees to nodes where voi appears
df_tree_voi = df_tree[df_tree['Feature'] == voi]

# sum total gain by each value of area used for splitting nodes
df_splitGain = df_tree_voi.groupby('Split').sum('Gain').sort_values(
    by = 'Gain', ascending = False
    ).head(20)[['Gain']]
df_splitGain = df_splitGain.reset_index()


## %% find peaks and plot
#############################

# define data
# Compute the histogram


df_plot = df_tree[
    df_tree['Feature'] == voi
    # df_tree['Feature'].str.contains(voi)
    ].sort_values(by = 'Split').reset_index(drop = True)

df_plot = df_plot.groupby(['Split'])['Gain'].sum().reset_index()

x_temp = df_plot['Split']
y_temp = df_plot['Gain']

# Sum y-values for ranges of 2 on the x-axis
x = []
y = []
# if x_temp.max() > 
range_len = np.round(x_temp.max()/100, 1) # length of values to sum over
if range_len == 0: range_len = np.round(x_temp.max()/100, 4)
if range_len == 0: range_len = np.round(x_temp.max()/100, 6)

for i in np.arange(0, np.ceil(x_temp.max() + range_len), range_len):

    y.append(sum(y_temp[(x_temp >= i) & (x_temp < i+range_len)]))
    x.append((i+(i+range_len))/2)
print(x, y)

# plt.bar(x, y)
# plt.xticks(rotation=90)


# calc distance threshold and prominence_threshold
y_range = np.max(y) - np.min(y)
x_range = x_temp.max() - x_temp.min()
distance_fraction = 0.0001  # e.g., 0.05% of data range
prominence_fraction = 0.3  # e.g., 30% of data range
height_in = 0.05 * y_temp.sum() # e.g., 15% total gain
distance_threshold = x_range * distance_fraction
if distance_threshold < 5: distance_threshold = 5
prominence_threshold = y_range * prominence_fraction

print(f'\n\ndistance_threshold: {distance_threshold}')
print(f'\nprominence_threshold: {prominence_threshold}')
print(f'\nheight: {height_in}')


# mirror the first 10 values to enable first value to be identified as peak
# how many values to add to front?
prpndN = 8

y = [y[i] for i in range(prpndN-1, 0, -1)] + list(y)

peaks, peakMetrics = find_peaks(
    y,
    distance = distance_threshold, 
    prominence = prominence_threshold,
    height = height_in # 5000 # peak must have at least this much gain
    )


peaks = peaks - (prpndN-1)

y = y[prpndN-1:len(y)]

peakSplits = np.array(x)[peaks]

# plot
plt.figure(figsize=(8, 4))

# Plot the continuous data
plt.plot(
    # x_temp, y_temp,
    x, y,
    label='Continuous Data', color='b')
# plt.scatter(x_temp, y_temp, 
#             c='b', marker='o', label='Peaks')
# plt.scatter(x_temp.iloc[peaks], 
#             y_temp.iloc[peaks], 
#             c='r', marker='o', label='Peaks')
plt.scatter(x, y, 
            c='b', marker='o')
plt.scatter(np.array(x)[peaks], 
            np.array(y)[peaks], 
            c='r', marker='o', label='Peaks')
# Add text labels to the points
for j in range(len(np.array(x)[peaks]) if len(x) > 1 else 1):
    i=j-1
    plt.text(
        np.array(x)[peaks][i], 
        np.array(y)[peaks][i], 
        np.round(np.array(x)[peaks][i], 2), 
        fontsize=12, ha='center', va='bottom')


plt.title(f"Peaks in Split Values of {voi} Based on Gain ({ts}, {cm}, {reg})")
plt.xlabel("Split Value")
plt.ylabel("Gain")
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig(
    f'{dir_figs}/Thresholds_{ts}_{cm}_{reg}_{voi}.png', dpi = 300
)



# %%
# # plots
# x = df_plot['']

# # plot
# plt.figure(figsize=(8, 4))

# # Plot the continuous data
# plt.plot(
#     # x_temp, y_temp,
#     x, y,
#     label='Continuous Data', color='b'
#     )
# # plt.scatter(x_temp, y_temp, 
# #             c='b', marker='o', label='Peaks')
# # plt.scatter(x_temp.iloc[peaks], 
# #             y_temp.iloc[peaks], 
# #             c='r', marker='o', label='Peaks')
# plt.scatter(x, y, 
#         c='b', marker='o')
# plt.scatter(np.array(x)[peaks], 
#         np.array(y)[peaks], 
#         c='r', marker='o', label='Peaks')


# plt.title(f"Identifying Peaks in Split Values of {voi} Based on Gain")
# plt.xlabel("Split Value")
# plt.ylabel("Gain")
# plt.legend()
# plt.grid(True)
# plt.show()


# %% plot histogram of peaks
# import matplotlib.pyplot as plt
# import math

# timescale_plot = 'monthly'
# var_plot = 'prcp'
# clust_meth_plot = 'AggEcoregion'

# #define points values by group
# df_in = df_peaksOut[
#     (df_peaksOut['clust_meth'] == clust_meth_plot) &
#     (df_peaksOut['timescale'] == timescale_plot) &
#     (df_peaksOut['variable'] == var_plot)
#     ] 
# var_in = 'region'

# for group in df_in[var_in].unique():
#     plt_vals = df_in.loc[df_in[var_in] == group, 'peaks']
#     nbins = math.ceil(math.log(len(plt_vals), 2) + 1) * 10

#     plt.hist(plt_vals, 
#              label = group, 
#              bins = nbins,
#              alpha = 0.5
#              )

# #add plot title and axis labels
# plt.title('Threshold Splits by Region')
# plt.xlabel('Split values')
# plt.ylabel('Frequency')

# #add legend
# plt.legend(title='Team')

# #display plot
# plt.show()








##################################################

# %%
################################
# gains of each split of each var
# Calculate the count of values in the 'Category' column
# value_counts = df_tree['Feature'].value_counts()[20:40]

# df_plot = df_treeGain.sort_values(
#     by = 'Gain', ascending = False
#     ).reset_index(drop = True)  # [120:130]
# df_plot.loc[0:125, 'Feature'].unique()
# Create a bar plot
# plt.bar(df_plot['Feat_Split'], df_plot['Gain'])

# # Adding labels and title
# plt.xlabel('Category')
# plt.ylabel('Gain')
# plt.title('Total gain at split value')
# # Rotate x-axis tick labels and anchor them to the end
# plt.xticks(rotation=45, horizontalalignment='right')
# # Show the plot
# plt.show()

# #####################

# ################################
# # gain of each split
# # Calculate the count of values in the 'Category' column
# # value_counts = df_tree['Feature'].value_counts()[20:40]

# # Create a bar plot
# plt.bar(
#     df_tree.loc[df_tree['Feature'] == voi, 'Split'], 
#     df_tree.loc[df_tree['Feature'] == voi, 'Gain'],
#     width = 1)

# # Adding labels and title
# plt.xlabel('Split')
# plt.ylabel('Gain')
# plt.title(f'Gains at Specific Split Values - {voi} - {region}')
# # Rotate x-axis tick labels and anchor them to the end
# plt.xticks(rotation=45, horizontalalignment='right')
# # Show the plot
# plt.show()

# #####################


# ################################
# # count of each var
# # Calculate the count of values in the 'Category' column
# value_counts = df_tree['Feature'].value_counts()[20:40]

# # Create a bar plot
# plt.bar(value_counts.index, value_counts.values)

# # Adding labels and title
# plt.xlabel('Category')
# plt.ylabel('Count')
# plt.title('Count of Splits Using Feature')
# # Rotate x-axis tick labels and anchor them to the end
# plt.xticks(rotation=45, horizontalalignment='right')
# # Show the plot
# plt.show()

# #####################

# print(df_tree.sort_values(
#     by = 'Gain', ascending = False
#     ).reset_index(drop = True)['Feature'].unique())


# # subset trees to nodes where voi appears
# df_tree_voi = df_tree[df_tree['Feature'] == voi]
# df_tree_voi
# df_tree_voi.sort_values(by = 'Gain', ascending = False)
# df_tree_voi.sort_values(by = 'Tree')

# # print number of times each unique value of the voi was uesd for splitting
# # df_tree_voi.loc[df_tree_voi['Feature'] == 'DRAIN_SQKM', 'Split'].value_counts().head(20)

# sum total gain by each value of area used for splitting nodes
# df_splitGain = df_tree_voi.groupby('Split').sum('Gain').sort_values(
#     by = 'Gain', ascending = False
#     ).head(20)[['Gain']]
# df_splitGain = df_splitGain.reset_index()
# fig, ax = plt.subplots(figsize = [8,8])
# ax.plot(df_splitGain['Split'], df_splitGain['Gain'])
# ax.scatter(df_splitGain['Split'], df_splitGain['Gain'], color = 'orange')
# ax.set(xlabel = 'Split Values', ylabel = 'Gain at split', title = voi)
# plt.show()

# import math
# # print the number of times each split occurs.
# if df_tree_voi.shape[0] == 0:
#     raise ValueError(
#         'Splits based on \{voi} not found, try another feature.\n'
#     )
# nbins = math.ceil(math.log(df_tree_voi.shape[0], 2) + 1) * 2
# # df_tree_voi.value_counts('Split').hist(bins = nbins)
# ax = df_tree_voi.Split.hist(bins = nbins)
# ax.set_title(f'Counts of splits in {voi}')
# plt.show()
# df_tree_voi.value_counts('Split')


# %% find peaks and plot
# ############################

# define data
# Compute the histogram


# df_plot = df_tree[
#     df_tree['Feature'] == voi
#     # df_tree['Feature'].str.contains(voi)
#     ].sort_values(by = 'Split').reset_index(drop = True)

# df_plot = df_plot.groupby(['Split'])['Gain'].sum().reset_index()

# x_temp = df_plot['Split']
# y_temp = df_plot['Gain']



# # Sum y-values for ranges of 2 on the x-axis
# x = []
# y = []
# # if x_temp.max() > 
# range_len = np.round(x_temp.max()/100, 1) # length of values to sum over
# if range_len == 0: range_len = np.round(x_temp.max()/100, 3)

# # for i in range(0, int(np.ceil(x_temp.max())), range_len):
# for i in np.arange(0, np.ceil(x_temp.max() + range_len), range_len):
#     # low = y_temp[i]
#     # high = y_temp[i + range_len]
#     y.append(sum(y_temp[(x_temp >= i) & (x_temp < i+range_len)]))
#     # x.append(f'{i}-{i+range_len}')
#     x.append((i+(i+range_len))/2)
# print(x, y)

# # plt.bar(x, y)
# # plt.xticks(rotation=90)


# # calc distance threshold and prominence_threshold
# y_range = np.max(y) - np.min(y)
# x_range = x_temp.max() - x_temp.min()
# distance_fraction = 0.0001  # e.g., 0.05% of data range
# prominence_fraction = 0.3  # e.g., 30% of data range
# height_in = 0.05 * y_temp.sum() # e.g., 15% total gain
# distance_threshold = x_range * distance_fraction
# if distance_threshold < 5: distance_threshold = 5
# prominence_threshold = y_range * prominence_fraction
# print(f'\n\ndistance_threshold: {distance_threshold}')
# print(f'\nprominence_threshold: {prominence_threshold}')
# print(f'\nheight: {height_in}')


# # mirror the first 10 values to enable first value to be identified as peak
# # how many values to add to front?
# prpndN = 8

# y = [y[i] for i in range(prpndN-1, 0, -1)] + list(y)

# peaks, peakMetrics = find_peaks(
#     # df_plot['Gain'], 
#     y,
#     distance = distance_threshold, 
#     prominence = prominence_threshold,
#     height = height_in # 5000 # peak must have at least this much gain
#     )
# peaks = peaks - (prpndN-1)

# y = y[prpndN-1:len(y)]

# peakSplits = np.array(x)[peaks]

# # plot
# plt.figure(figsize=(8, 4))

# # Plot the continuous data
# plt.plot(
#     # x_temp, y_temp,
#     x, y,
#     label='Continuous Data', color='b')
# # plt.scatter(x_temp, y_temp, 
# #             c='b', marker='o', label='Peaks')
# # plt.scatter(x_temp.iloc[peaks], 
# #             y_temp.iloc[peaks], 
# #             c='r', marker='o', label='Peaks')
# plt.scatter(x, y, 
#             c='b', marker='o')
# plt.scatter(np.array(x)[peaks], 
#             np.array(y)[peaks], 
#             c='r', marker='o', label='Peaks')


# plt.title(f"Identifying Peaks in Split Values of {voi} Based on Gain")
# plt.xlabel("Split Value")
# plt.ylabel("Gain")
# plt.legend()
# plt.grid(True)
# plt.show()


# %%
# plot tree
##############

# fig, ax = plt.subplots(figsize = (100, 100))
# # xgb.plot_tree(model, num_trees = 89, ax = ax)
# xgb.plot_tree(model, num_trees = 749, ax = ax)
# %%
