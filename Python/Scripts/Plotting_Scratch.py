# Ben Choat 7/17/2022

# some plotting of GAGESii stuff

# %% import libraries

import pandas as pd
import plotnine as p9

# %% import data
# explantory var (and other data) directory
dir_expl = 'D:/Projects/GAGESii_ANNstuff/Data_Out'
df_results = pd.read_csv(f'{dir_expl}/Results/Results_NonTimeSeries.csv')

df_results = df_results[df_results['parameters'] != 'nc38alpha1.0']
df_results = df_results[df_results['parameters'] != 'alpha1.0']


# %% plot some performance plots from mean annual water yield with no clustering
# adjR2
p1 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'r2adj')) +
             p9.theme(axis_text_x = p9.element_text(angle = 45))
    )

# mae
p2 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'mae')) +
             p9.theme(axis_text_x = p9.element_text(angle = 45))
    )

# rmse
p3 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'rmse')) +
             p9.theme(axis_text_x = p9.element_text(angle = 45))
    )


# r2adj from linear regression by parameters
p4 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'r2adj', fill = 'parameters')) +
             p9.theme(axis_text_x = p9.element_text(angle = 45))
    )

# r2adj from linear regression by parameters
p5 = (
        p9.ggplot(data = df_results) +
            p9.geom_boxplot(p9.aes(x = 'model', y = 'r2adj', fill = 'train_val')) +
             p9.theme(axis_text_x = p9.element_text(angle = 45)) # +
             # p9.facet_grid(facets = '. ~ parameters')
    )
# %%

p1
p2
p3
p4
p5
# %%
