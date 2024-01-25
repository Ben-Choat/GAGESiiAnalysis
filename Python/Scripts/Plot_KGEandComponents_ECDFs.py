'''
BChoat 2024/1/24

Script to plot ecdfs of the components of KGE, and KGE itself.
'''

# %%
# Import libraries
################


import pandas as pd
import numpy as np
# import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
# from statsmodels.distributions.empirical_distribution import ECDF


# %%
# define directory variables and specify scenario to work with


# Define directory variables
# directory with data to work with
# dir_work = 'D:/Projects/GAGESii_ANNstuff/HPC_Files/GAGES_Work/data_out' 
dir_work = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Results' 

# save figs (True of False?)
save_figs = False
# directory where to write figs
dir_figs = 'D:/Projects/GAGESii_ANNstuff/Data_Out/Figures'


# cluster method to use
# clust_meth_in = ['AggEcoregion']
clust_meth_in = ['None', 'Class', 'AggEcoregion']
# which model to calc shaps for
model_in = 'XGBoost'


# %% load data
###########

# annual and monthy 
df_resind_All = pd.read_csv(
        f'{dir_work}/NSEComponents_KGE.csv',
        dtype = {'STAID': 'string',
                'region': 'string'}
    )

df_resind_annual = df_resind_All[
    df_resind_All['time_scale'] == 'annual'
    ]

df_resind_monthly = df_resind_All[
    df_resind_All['time_scale'] == 'monthly'
    ]


# %% create a custom color map to associate specific colors with regions
############################

color_dict = {
    'CntlPlains': 'blue',
    'EastHghlnds': 'cyan',
    'MxWdShld': 'darkgoldenrod',
    'NorthEast': 'orange',
    'SECstPlain': 'green',
    'SEPlains': 'lime',
    'WestMnts': 'firebrick',
    'WestPlains': 'tomato',
    'WestXeric': 'saddlebrown',
    'Non-ref': 'purple',
    'Ref': 'palevioletred',
    'All': 'dimgray'
}
custom_cmap = ListedColormap([color_dict[key] for key in color_dict])
custom_pallete = [color_dict[key] for key in color_dict]



# %%
# eCDF plots (all three timescales)
##############
# metric_in = 'NSE'

cmap_str = custom_pallete

# annual
data_in_annual = df_resind_annual.copy() # [


data_in_annual.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


# month
data_in_month = df_resind_monthly.copy() # [


data_in_month.sort_values(
    by = ['clust_method', 'region'], inplace = True
    )


####

data_in_annual = data_in_annual[
    (data_in_annual['clust_method'].isin(clust_meth_in)) &
    (data_in_annual['model'] == model_in)
]

data_in_month = data_in_month[
    (data_in_month['clust_method'].isin(clust_meth_in)) &
    (data_in_month['model'] == model_in)
]

####

fig, axs = plt.subplots(4, 4, figsize = (8.5, 11)) # , sharex = True)#, sharex = True)
# ax1, ax2, ax3, ax4, ax5, ax6, = axs.flatten()
axsf = axs.flatten()


# annual training
sns.ecdfplot(
    data = data_in_annual.query('train_val == "train"'),
    x = 'r',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[0],
    legend = False
)
axsf[0].annotate('(a)', xy = (0.05, 0.9)) #(80, 0.02))
axsf[0].grid()
axsf[0].title.set_text('CDF')
axsf[0].set_xlim(0, 1)
axsf[0].set_xticklabels([])
axsf[0].set(xlabel = '', # '|residuals| [cm]', 
        ylabel = 'Annual Training Data') # Non-Exceedence Probability (Training Data)')


sns.kdeplot(
    data = data_in_annual.query('train_val == "train"'),
    x = 'alpha',
    hue = 'region',
    linestyle = '--', # ':',
    palette = cmap_str,
    ax = axsf[1],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[1].annotate('(b)', xy = (0.1, 1.08)) # (0.8, 0.02))
axsf[1].grid()
axsf[1].title.set_text('KDE')
axsf[1].set(xlabel = '', ylabel = '') #metric_in)
axsf[1].set_xlim(0, 2)
axsf[1].set_xticklabels([])


sns.kdeplot(
    data = data_in_annual.query('train_val == "train"'),
    x = 'beta',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[2],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[2].annotate('(c)', xy = (0.1, 6.1)) # (0.8, 0.02))
axsf[2].grid()
axsf[2].title.set_text('KDE')
axsf[2].set(xlabel = '', ylabel = '') # metric_in)
axsf[2].set_xlim(0, 2)
axsf[2].set_xticklabels([])

# testing
sns.ecdfplot(
    data = data_in_annual.query('train_val == "train"'),
    x = 'KGE',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[3],
    legend = False
)
axsf[3].annotate('(d)', xy = (-0.9, 0.9)) #(80, 0.02))
axsf[3].grid()
axsf[3].title.set_text('CDF') # Mean Annual')
axsf[3].set_xlim(-1, 1)
axsf[3].set_xticklabels([])
axsf[3].set(xlabel = '', 
        ylabel = '')



# -------------------------

# Annual testing

# annual training
sns.ecdfplot(
    data = data_in_annual.query('train_val == "valnit"'),
    x = 'r',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[4],
    legend = False
)
axsf[4].annotate('(e)', xy = (0.05, 0.9)) #(80, 0.02))
axsf[4].grid()
axsf[4].title.set_text('')
axsf[4].set_xlim(0, 1)
axsf[4].set_xticklabels([])
axsf[4].set(xlabel = '', # '|residuals| [cm]', 
        ylabel = 'Annual Testing Data') # Non-Exceedence Probability (Training Data)')


sns.kdeplot(
    data = data_in_annual.query('train_val == "valnit"'),
    x = 'alpha',
    hue = 'region',
    linestyle = '--', # ':',
    palette = cmap_str,
    ax = axsf[5],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[5].annotate('(f)', xy = (0.1, 0.29)) # (0.8, 0.02))
axsf[5].grid()
axsf[5].title.set_text('')
axsf[5].set(xlabel = '', ylabel = '') #metric_in)
axsf[5].set_xlim(0, 2)
axsf[5].set_xticklabels([])


sns.kdeplot(
    data = data_in_annual.query('train_val == "valnit"'),
    x = 'beta',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[6],
    clip = (0, 2),
    fill =False,
    legend = False
)
axsf[6].annotate('(g)', xy = (0.1, 0.21)) # (0.8, 0.02))
axsf[6].grid()
axsf[6].title.set_text('')
axsf[6].set(xlabel = '', ylabel = '') # metric_in)
axsf[6].set_xlim(0, 2)
axsf[6].set_xticklabels([])

# testing
sns.ecdfplot(
    data = data_in_annual.query('train_val == "valnit"'),
    x = 'KGE',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[7],
    legend = False
)
axsf[7].annotate('(h)', xy = (-0.9, 0.9)) #(80, 0.02))
axsf[7].grid()
axsf[7].title.set_text('') # Mean Annual')
axsf[7].set_xlim(-1, 1)
axsf[7].set_xticklabels([])
axsf[7].set(xlabel = '', 
        ylabel = '')


# -----------------------------------------

# monthly training
sns.ecdfplot(
    data = data_in_month.query('train_val == "train"'),
    x = 'r',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[8],
    legend = False
)
axsf[8].annotate('(i)', xy = (0.05, 0.9)) #(80, 0.02))
axsf[8].grid()
axsf[8].title.set_text('')
axsf[8].set_xlim(0, 1)
axsf[8].set_xticklabels([])
axsf[8].set(xlabel = '', # '|residuals| [cm]', 
        ylabel = 'Monthly Training Data') # Non-Exceedence Probability (Training Data)')


sns.kdeplot(
    data = data_in_month.query('train_val == "train"'),
    x = 'alpha',
    hue = 'region',
    linestyle = '--', # ':',
    palette = cmap_str,
    ax = axsf[9],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[9].annotate('(j)', xy = (0.1, 0.61)) # (0.8, 0.02))
axsf[9].grid()
axsf[9].title.set_text('')
axsf[9].set(xlabel = '', ylabel = '') #metric_in)
axsf[9].set_xlim(0, 2)
axsf[9].set_xticklabels([])


sns.kdeplot(
    data = data_in_month.query('train_val == "train"'),
    x = 'beta',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[10],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[10].annotate('(k)', xy = (0.1, 5.1)) # (0.8, 0.02))
axsf[10].grid()
axsf[10].title.set_text('')
axsf[10].set(xlabel = '', ylabel = '') # metric_in)
axsf[10].set_xlim(0, 2)
axsf[10].set_xticklabels([])

# testing
sns.ecdfplot(
    data = data_in_month.query('train_val == "train"'),
    x = 'KGE',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[11],
    legend = False
)
axsf[11].annotate('(l)', xy = (-0.9, 0.9)) #(80, 0.02))
axsf[11].grid()
axsf[11].title.set_text('') # Mean Annual')
axsf[11].set_xlim(-1, 1)
axsf[11].set_xticklabels([])
axsf[11].set(xlabel = '', 
        ylabel = '')



# -------------------------

# Annual testing

# annual training
sns.ecdfplot(
    data = data_in_month.query('train_val == "valnit"'),
    x = 'r',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[12],
    legend = False
)
axsf[12].annotate('(m)', xy = (0.05, 0.9)) #(80, 0.02))
axsf[12].grid()
axsf[12].title.set_text('')
axsf[12].set_xlim(0, 1)
# axsf[12].set_xticklabels([])
axsf[12].set(xlabel = 'r', # '|residuals| [cm]', 
        ylabel = 'Monthly Testing Data') # Non-Exceedence Probability (Training Data)')


sns.kdeplot(
    data = data_in_month.query('train_val == "valnit"'),
    x = 'alpha',
    hue = 'region',
    linestyle = '--', # ':',
    palette = cmap_str,
    ax = axsf[13],
    clip = (0, 2),
    fill = False,
    legend = False
)
axsf[13].annotate('(n)', xy = (0.1, 0.355)) # (0.8, 0.02))
axsf[13].grid()
axsf[13].title.set_text('')
axsf[13].set(xlabel = 'alpha', ylabel = '') #metric_in)
axsf[13].set_xlim(0, 2)
# axsf[13].set_xticklabels([])


sns.kdeplot(
    data = data_in_month.query('train_val == "valnit"'),
    x = 'beta',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[14],
    clip = (0, 2),
    fill =False,
    legend = False
)
axsf[14].annotate('(o)', xy = (0.1, 0.205)) # (0.8, 0.02))
axsf[14].grid()
axsf[14].title.set_text('')
axsf[14].set(xlabel = 'beta', ylabel = '') # metric_in)
axsf[14].set_xlim(0, 2)
# axsf[14].set_xticklabels([])

# testing
sns.ecdfplot(
    data = data_in_month.query('train_val == "valnit"'),
    x = 'KGE',
    hue = 'region',
    linestyle = '--',
    palette = cmap_str,
    ax = axsf[15],
    legend = False
)
axsf[15].annotate('(p)', xy = (-0.9, 0.9)) #(80, 0.02))
axsf[15].grid()
axsf[15].title.set_text('') # Mean Annual')
axsf[15].set_xlim(-1, 1)
# axsf[15].set_xticklabels([])
axsf[15].set(xlabel = 'KGE', 
        ylabel = '')


# Add a legend at the bottom
# legend = plt.legend(title='Region', bbox_to_anchor=(0.8, -0.4), loc='right', ncol=6)
legend_labels = [x for x in data_in_month['region'].unique()]
legend_handles = [Line2D([0], [0], linestyle='--', color=color_dict[label]) \
                  for label in legend_labels]
common_legend = fig.legend(title='Region', labels = legend_labels, 
                           handles = legend_handles,
                           bbox_to_anchor=(1, -0.01), loc='right', ncol=6,
                           frameon = False)
plt.subplots_adjust(bottom=0.1)  # Adjust the bottom margin if needed

plt.tight_layout()
# save fig
if save_figs:
    plt.savefig(
        # f'{dir_figs}/ecdfs_{part_in}_{clust_meth_in}_{model_in}.png', 
        f'{dir_figs}/ecdfs_TrainTest_KGE_Components_{clust_meth_in}_{model_in}.png', 
        dpi = 300,
        bbox_inches = 'tight'
        )
else:
    plt.show()