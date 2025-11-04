'''
BChoat 2025/11/2

Script to calculate P:PET using DAYMET mean annual precip and GAGES-II mean annual PET
'''

# %% import libs
#############################
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %% define input vars, dirs, and such
#############################

# base dir for inputs
BASE_IN = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/Share/Publish/",
    "Results_Data_HPCScripts_WRR_Choatetal"
)

# DAYMET dir based off BASE_IN
DAYMET_DIR = Path(BASE_IN, "HPC_Files/GAGES_Work/data_work/Daymet")

# dir with gagesii data
G2_DIR = Path(BASE_IN, "HPC_Files/GAGES_Work/data_work/GAGESiiVariables")

# dir with results
RESULTS_DIR = Path(BASE_IN, "Data_Out/Results")

CLUST_METHODS = [
    'Class', "None", "AggEcoregion", "CAMELS"
]

# %% Read in data
#############################

# read in id data from gagesii
df_id_train = pd.read_csv(Path(G2_DIR, "ID_train.csv"), dtype={'STAID':str})
df_id_test = pd.read_csv(Path(G2_DIR, "ID_valnit.csv"), dtype={'STAID':str})

# read in gagesii explanatory variables (annual)
df_expl_train = pd.read_csv(Path(G2_DIR, "Expl_train.csv"), dtype={'STAID':str})
df_expl_test = pd.read_csv(Path(G2_DIR, "Expl_valnit.csv"), dtype={'STAID':str})

# get average vals across years
df_expl_train = (
    df_expl_train
    .groupby('STAID')
    .mean()
    .drop('year', axis=1)
    .reset_index()
    )
df_expl_test = (
    df_expl_test
    .groupby('STAID')
    .mean()
    .drop('year', axis=1)
    .reset_index()
    )

# read in results
df_results_meanannual = pd.read_csv(Path(RESULTS_DIR, "PerfMetrics_MeanAnnual.csv"),
                        dtype={"STAID": str})
df_results_ts = pd.read_csv(Path(RESULTS_DIR, "NSEComponents_KGE.csv"),
                            dtype={"STAID": str})

# subset to cluster methods of interest
df_results_meanannual = df_results_meanannual.query("clust_method in @CLUST_METHODS")
df_results_ts = df_results_ts.query("clust_method in @CLUST_METHODS")
# subset results to best model for each STAID
df_results_annual = df_results_ts.query("time_scale == 'annual'")
df_results_month = df_results_ts.query("time_scale == 'monthly'")

# mean annual
idx = (
    df_results_meanannual
    .groupby('STAID')['residuals']
    .apply(lambda s: s.abs().idxmin())
)
df_ma_best = (
    df_results_meanannual
    .loc[idx.values]
    .reset_index(drop=True)
)

# annual
idx = (
    df_results_annual
    .groupby('STAID')['NSE']
    .apply(lambda s: s.idxmax())
)
df_annual_best = (
    df_results_annual
    .loc[idx.values]
    .reset_index(drop=True)
)

# month
idx = (
    df_results_month
    .groupby('STAID')['NSE']
    .apply(lambda s: s.idxmax())
)
df_month_best = (
    df_results_month
    .loc[idx.values]
    .reset_index(drop=True)
)

# add id col indicating train or test
df_id_train['partition'] = 'train'
df_id_test['partition'] = 'test'
df_expl_train['partition'] = 'train'
df_expl_test['partition'] = 'test'

# combine into single df to work with more easily
df_id = pd.concat([df_id_train, df_id_test], axis=0)
df_expl = pd.concat([df_expl_train, df_expl_test], axis=0)
df_expl = pd.merge(
    df_expl, df_id[["STAID", "AggEcoregion"]], on="STAID"
)

# add perf metrics to df_expl by STAID
df_ma_best = df_ma_best[['STAID', 'residuals']].rename(columns={'residuals': 'Residual'})
df_annual_best = df_annual_best[['STAID', 'NSE', 'KGE']].rename(
    columns={'NSE': 'NSE_annual', 'KGE': 'KGE_annual'}
)
df_month_best = df_month_best[['STAID', 'NSE', 'KGE']].rename(
    columns={'NSE': 'NSE_monthly', 'KGE': 'KGE_monthly'}
)
df_perf = (
    df_ma_best
    .merge(df_annual_best, on='STAID', how='left')
    .merge(df_month_best, on='STAID', how='left')
)
df_expl = df_expl.merge(df_perf, on='STAID', how='left')
# %% calculate P:PET across all basins (mean annual)
############################
df_p_pet = df_expl.copy()[[
    "STAID", "PPTAVG_BASIN", "PET", "AggEcoregion", "partition",
    "Residual", "NSE_annual", "KGE_annual", "NSE_monthly", "KGE_monthly"
    ]]
# df_p_pet_train = df_expl_train[["STAID", "year", "PPTAVG_BASIN", "PET"]]
# df_p_pet_test = df_expl_test[["STAID", "year", "PPTAVG_BASIN", "PET"]]

# convert precip to mm
df_p_pet['PPTAVG_BASIN'] = df_p_pet["PPTAVG_BASIN"]*10
# df_p_pet_train['PPTAVG_BASIN'] = df_p_pet_train["PPTAVG_BASIN"]*10
# df_p_pet_test['PPTAVG_BASIN'] = df_p_pet_test["PPTAVG_BASIN"]*10

# calc p:pet
df_p_pet['P:PET'] = df_p_pet['PPTAVG_BASIN']/df_p_pet['PET']
# categorize P:PET into rainfall-runoff regime bins for plotting
bins = [-np.inf, 0.8, 1.0, 1.2, np.inf]
labels = ['<0.8', '0.8-1.0', '1.0-1.2', '>1.2']
df_p_pet['P:PET_Range'] = pd.cut(
    df_p_pet['P:PET'],
    bins=bins,
    labels=labels,
    right=False
)
# %% plot map with P:PET values
############################

# %% boxplots or similar for P:PET by ecoregion
############################

# %% HISTOGRAMS
###############################
ax = df_p_pet['P:PET'].hist(bins=50)
ax.axvline(x=1, color='brown', linestyle='--',
           linewidth=2, label='x=1')


fig, ax = plt.subplots()
for name, grp in df_p_pet.groupby('AggEcoregion'):
    ax.hist(grp['P:PET'], bins=50, alpha=0.4, label=name, density=True)
    ax.legend(title='AggEcoregion')
ax.axvline(1, color='brown', linestyle='--', linewidth=2, label='x=1')


# Overlayed with seaborn (simplest)

ax = sns.histplot(data=df_p_pet, x='P:PET', hue='AggEcoregion', bins=50,
        element='step', stat='density', common_norm=False, alpha=0.4)
# ax.legend(title='AggEcoregion')

ax.axvline(1, color='brown', linestyle='--', linewidth=2, label='x=1')


# Small multiples (facets) if many groups

g = sns.displot(data=df_p_pet, x='P:PET', col='AggEcoregion', col_wrap=3, bins=50)
for ax in g.axes.flat:
    ax.axvline(1, color='brown', linestyle='--', linewidth=2)

# %%BOXPLOTS
###############################

ax = sns.boxplot(data=df_p_pet, y="AggEcoregion", x="P:PET", hue="partition")
ax.axvline(x=1, color='brown', linestyle='--', linewidth=2, label='P:PET=1')


# %% kge vs P:PET
###############################

metric_in = "NSE"
# mean annual
df_p_pet['abs(Residual)'] = df_p_pet['Residual'].abs()
ax = sns.scatterplot(data=df_p_pet, x="P:PET", y="abs(Residual)")
ax.axvline(1, color='brown', linestyle='--', linewidth=2)
ax.set_yscale("log")
ax.set_title("Mean Annual Residuals (cm)")
plt.show()

# annual
ax = sns.scatterplot(data=df_p_pet, x="P:PET", y=f"{metric_in}_annual")
ax.axvline(1, color='brown', linestyle='--', linewidth=2)
ax.set_yscale("log")
ax.set_title(f"{metric_in} for annual WY")
plt.show()

# monthly
ax = sns.scatterplot(data=df_p_pet, x="P:PET", y=f"{metric_in}_monthly", hue='partition')
ax.axvline(1, color='brown', linestyle='--', linewidth=2)
# ax.set_yscale("log")
# ax.set_ylim([-1, 1])
ax.set_title(f"{metric_in} for monthly WY")
plt.show()

# %% boxplots of kge/nse vs



# %% consider shap values for basins P:PET<1 versus P:PET>1
############################

