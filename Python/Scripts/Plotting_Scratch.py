# Ben Choat 7/17/2022

# some plotting of GAGESii stuff

# %% import libraries

import pandas as pd
import plotnine as p9

# %% import data

# Define working directories and variables

# define which clustering method is being combined. This variable 
# will be used for collecting data from the appropriate directory as well as
# naming the combined file
clust_meth = 'AggEcoregion' #AggEcoregion', 'None'

# define time scale working with. This variable will be used to read and
# write data from and to the correct directories
time_scale = 'monthly' # 'mean_annual', 'annual', 'monthly', 'daily'

# results directory
dir_res = f'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out/{time_scale}'
df_ind_results = pd.read_pickle(f'{dir_res}/combined/All_IndResults_monthly.pkl')
df_sum_results = pd.read_pickle(f'{dir_res}/combined/All_SummaryResults_monthly.pkl')

# df_results = df_results[df_results['parameters'] != 'nc38alpha1.0']
# df_results = df_results[df_results['parameters'] != 'alpha1.0']

# subset df_results to models of interest
#define variable holding call to models of interest (moi)
moi = 'regr_precip|strd_lasso|strd_mlr|PCA|XGBoost'
df_ind_results = df_ind_results[df_ind_results['model'].str.contains(moi)]
df_sum_results = df_sum_results[df_sum_results['model'].str.contains(moi)]


# %% plot some performance plots from mean annual water yield with no clustering
# # adjR2
# p1 = (
#         p9.ggplot(data = df_results) +
#             p9.geom_boxplot(p9.aes(x = 'model', y = 'r2adj')) +
#              p9.theme(axis_text_x = p9.element_text(angle = 45)) # +
#              # p9.ylim(0, 1)
#     )

# # mae
# p2 = (
#         p9.ggplot(data = df_results) +
#             p9.geom_boxplot(p9.aes(x = 'model', y = 'mae')) +
#              p9.theme(axis_text_x = p9.element_text(angle = 45))
#     )

# # rmse
# p3 = (
#         p9.ggplot(data = df_results) +
#             p9.geom_boxplot(p9.aes(x = 'model', y = 'rmse')) +
#              p9.theme(axis_text_x = p9.element_text(angle = 45))
#     )


# # r2adj from linear regression by parameters
# p4 = (
#         p9.ggplot(data = df_results) +
#             p9.geom_boxplot(p9.aes(x = 'model', y = 'r2adj', fill = 'parameters')) +
#              p9.theme(axis_text_x = p9.element_text(angle = 45))
#     )

# r2adj from linear regression by parameters
p5 = (
        p9.ggplot(data = df_ind_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'NSE', fill = 'train_val')) +
            p9.theme(axis_text_x = p9.element_text(angle = 45)) +
            p9.ylim(-2, 1) + # Note that stats for box and whiskers seems to be calced excluding vals outside ylim
            p9.facet_wrap(facets = 'region', ncol = 2, scales = 'free_y')
    )

# r2 from linear regression by parameters
p6 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'r2', fill = 'train_val')) +
            p9.theme(axis_text_x = p9.element_text(angle = 45)) +
            p9.ylim(0, 1) +
            p9.facet_wrap(facets = 'region', ncol = 2, scales = 'free_y')
    )

# mae from linear regression by parameters
p7 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'mae', fill = 'train_val')) +
            p9.theme(axis_text_x = p9.element_text(angle = 45)) +
            # p9.ylim(0, 1) +
            p9.facet_wrap(facets = 'region', ncol = 2, scales = 'free_y')
    )

# rmse from linear regression by parameters
p8 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'rmse', fill = 'train_val')) +
            p9.theme(axis_text_x = p9.element_text(angle = 45)) +
            # p9.ylim(0, 1) +
            p9.facet_wrap(facets = 'region', ncol = 2, scales = 'free_y')
    )

# %%

# p1
# p2
# p3
# p4
p5
p6
p7
p8
# %%
