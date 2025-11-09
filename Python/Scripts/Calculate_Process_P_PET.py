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

# dir holding shap values
SHAP_DIR = Path("C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff",
                "Data_Out/SHAP_OUT")

# file holding feature categories
FEAT_CATS_FILE = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/",
    "GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv"
)

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

# read in and prep shap vals
df_shap_ma_train = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_train_mean_annual_normQ.csv")
)
df_shap_ma_test = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_valnit_mean_annual_normQ.csv")
)
df_shap_a_train = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_train_annual_normQ.csv")
)
df_shap_a_test = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_valnit_annual_normQ.csv")
)
df_shap_mo_train = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_train_monthly_normQ.csv")
)
df_shap_mo_test = pd.read_csv(
    Path(SHAP_DIR, "MeanShap_valnit_monthly_normQ.csv")
)
## combine train and test
df_shap_ma_train['partition'] = "train"
df_shap_ma_test['partition'] = 'test'
df_shap_a_train['partition'] = "train"
df_shap_a_test['partition'] = 'test'
df_shap_mo_train['partition'] = "train"
df_shap_mo_test['partition'] = 'test'
df_shap_ma = pd.concat([df_shap_ma_train, df_shap_ma_test])
df_shap_a = pd.concat([df_shap_a_train, df_shap_a_test])
df_shap_mo = pd.concat([df_shap_mo_train, df_shap_mo_test])

# read in features
feat_cats = pd.read_csv(
    'C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories.csv'
    
    # 'D:/Projects/GAGESii_ANNstuff/Data_Out/UMAP_HDBSCAN/FeatureCategories_wUnits.csv'
    )

# %% calculate P:PET across all basins (mean annual)
############################
df_p_pet = df_expl.copy()[[
    "STAID", "PPTAVG_BASIN", "PET", "AggEcoregion", "partition",
    "Residual", "NSE_annual", "KGE_annual", "NSE_monthly", "KGE_monthly"
    ]]
# df_p_pet_train = df_expl_train[[="STAID", "year", "PPTAVG_BASIN", "PET"]]
# df_p_pet_test = df_expl_test[["STAID", "year", "PPTAVG_BASIN", "PET"]]

# convert precip to mm
df_p_pet['PPTAVG_BASIN'] = df_p_pet["PPTAVG_BASIN"]*10
# df_p_pet_train['PPTAVG_BASIN'] = df_p_pet_train["PPTAVG_BASIN"]*10
# df_p_pet_test['PPTAVG_BASIN'] = df_p_pet_test["PPTAVG_BASIN"]*10

# calc p:pet
df_p_pet['P:PET'] = df_p_pet['PPTAVG_BASIN']/df_p_pet['PET']

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

metric_in = "KGE"
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
# categorize P:PET into rainfall-runoff regime bins for plotting
# bins = [-np.inf, 0.8, 1.2, np.inf]
# labels = ['<0.8', '0.8-1.2', '>1.2']
bins = [-np.inf, 0.9, 1.1, np.inf]
labels = ['<0.9', '0.9-1.1', '>1.1']
bins = [-np.inf, 1, np.inf]
labels = ['<1', '>=1']

# categorize P:PET into coarse regimes for later analysis
df_p_pet['P:PET_Category'] = pd.cut(
    df_p_pet['P:PET'],
    bins=bins,
    labels=labels,
    right=False
)

ax = sns.boxplot(
    df_p_pet, x='P:PET_Category',
    y=f"{metric_in}_monthly",
    hue='partition'
)
ax.set_ylim([-1, 1])

# %% ECDFs
#################################

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(7, 5))

linestyles = {'train': '-', 'test': '--'}
hue_col = "P:PET_Category"

# fix hue order & colors so lines and legend stay in sync
hue_order = sorted(df_p_pet[hue_col].dropna().unique())
palette = sns.color_palette("Set2", n_colors=len(hue_order))
color_map = dict(zip(hue_order, palette))

# plot both partitions, suppressing auto legends
for part, ls in linestyles.items():
    sns.ecdfplot(
        data=df_p_pet[df_p_pet['partition'] == part],
        x=f"{metric_in}_annual",
        hue=hue_col,
        hue_order=hue_order,
        palette=color_map,
        linewidth=3,
        linestyle=ls,
        ax=ax,
        legend=False,
    )

# axes formatting
ax.set_xlabel(f"{metric_in} (annual)")
ax.set_ylabel("Empirical CDF")
ax.set_title(f"ECDF of Annual {metric_in} by P:PET Category and Partition")
ax.set_xlim([-1, 1])

# --- Legend 1: P:PET Category (color) ---
color_handles = [
    plt.Line2D([0], [0], color=color_map[k], linestyle='-', linewidth=2, label=k)
    for k in hue_order
]
leg1 = ax.legend(color_handles, hue_order, title="P:PET Category", loc="upper left")
ax.add_artist(leg1)

# --- Legend 2: Partition (linestyle) ---
style_handles = [
    plt.Line2D([0], [0], color='k', linestyle=ls, linewidth=2, label=part)
    for part, ls in linestyles.items()
]
ax.legend(style_handles, list(linestyles.keys()), title="Partition", loc="upper center")

plt.tight_layout()
plt.show()

# %% consider shap values for basins P:PET<1 versus P:PET>1
############################

def summarize_shap(df, feature_names, suffix):
    cols = [col for col in feature_names if col in df.columns]
    if not cols:
        return (
            df[['partition', 'region']]
            .drop_duplicates()
            .assign(**{f'shap_sum_{suffix}': 0})
        )
    abs_sum = df[cols].abs().sum(axis=1)
    return (
        pd.concat(
            [
                df[['partition', 'region']],
                abs_sum.rename(f'shap_sum_{suffix}')
            ],
            axis=1
        )
        .groupby(['partition', 'region'], as_index=False)[f'shap_sum_{suffix}']
        .sum()
    )


# prep shap vals
df_featcats = pd.read_csv(FEAT_CATS_FILE)
feature_sets = {
    'climate': df_featcats.loc[df_featcats['Coarse_Cat'] == 'Climate', 'Features'],
    'physio': df_featcats.loc[df_featcats['Coarse_Cat'] == 'Physiography', 'Features'],
    'anthro_hydro': df_featcats.loc[df_featcats['Coarse_Cat'] == 'Anthro_Hydro', 'Features'],
    'anthro_land': df_featcats.loc[df_featcats['Coarse_Cat'] == 'Anthro_Land', 'Features'],
}
shap_frames = {
    'ma': df_shap_ma,
    'a': df_shap_a,
    'mo': df_shap_mo,
}

# calculate summed absolute shap values for each frame/category combo
shap_summaries = {
    f'{frame_key}_{feature_key}': summarize_shap(df, features, f'{frame_key}_{feature_key}')
    for frame_key, df in shap_frames.items()
    for feature_key, features in feature_sets.items()
}

# ma_cols = [col for col in shap_summaries.keys() if "ma_" in col]
# a_cols = [col for col in shap_summaries.keys() if "a_" in col]
# a_cols = [col for col in a_cols if "ma_" not in col]
# mo_cols = [col for col in shap_summaries.keys() if "mo_" in col]

# ma_clim_sum = shap_summaries['ma_climate']
# a_clim_sum = shap_summaries['a_climate']
# mo_clim_sum = shap_summaries['mo_climate']

# mean annual
df_shap_ma_summ = shap_summaries['ma_climate']
df_shap_ma_summ = pd.merge(
    df_shap_ma_summ, shap_summaries['ma_physio'],
    on=['region', 'partition']
)
df_shap_ma_summ = pd.merge(
    df_shap_ma_summ, shap_summaries['ma_anthro_hydro'],
    on=['region', 'partition']
)
df_shap_ma_summ = pd.merge(
    df_shap_ma_summ, shap_summaries['ma_anthro_land'],
    on=['region', 'partition']
)
# annual
df_shap_a_summ = shap_summaries['ma_climate']
df_shap_a_summ = pd.merge(
    df_shap_a_summ, shap_summaries['ma_physio'],
    on=['region', 'partition']
)
df_shap_a_summ = pd.merge(
    df_shap_a_summ, shap_summaries['ma_anthro_hydro'],
    on=['region', 'partition']
)
df_shap_a_summ = pd.merge(
    df_shap_a_summ, shap_summaries['ma_anthro_land'],
    on=['region', 'partition']
)
# monthly
df_shap_mo_summ = shap_summaries['ma_climate']
df_shap_mo_summ = pd.merge(
    df_shap_mo_summ, shap_summaries['ma_physio'],
    on=['region', 'partition']
)
df_shap_mo_summ = pd.merge(
    df_shap_mo_summ, shap_summaries['ma_anthro_hydro'],
    on=['region', 'partition']
)
df_shap_mo_summ = pd.merge(
    df_shap_mo_summ, shap_summaries['ma_anthro_land'],
    on=['region', 'partition']
)

# normalize to 1
df_shap_ma_summ.iloc[:, 2:6] = df_shap_ma_summ.iloc[:, 2:6].div(
    df_shap_ma_summ.iloc[:, 2:6].sum(axis=1, skipna=True),
    axis=0,
)
df_shap_a_summ.iloc[:, 2:6] = df_shap_a_summ.iloc[:, 2:6].div(
    df_shap_a_summ.iloc[:, 2:6].sum(axis=1, skipna=True),
    axis=0,
)
df_shap_mo_summ.iloc[:, 2:6] = df_shap_mo_summ.iloc[:, 2:6].div(
    df_shap_mo_summ.iloc[:, 2:6].sum(axis=1, skipna=True),
    axis=0,
)


# shap_ma = {k: shap_summaries.get(k) for k in shap_summaries.keys()}

# %%
