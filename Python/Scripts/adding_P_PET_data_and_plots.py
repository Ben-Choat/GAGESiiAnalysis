"""
BChoat 2025/11/08

Add binary P:PET classes (<1, >=1) to the core ID tables, then
use those classes to (1) plot NSE/KGE eCDFs and (2) aggregate
SHAP values for downstream visualization.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


# %% define dirs, vars, and such -------------------------------------------------
BASE_SHARE = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/Share/Publish/",
    "Results_Data_HPCScripts_WRR_Choatetal",
)
GAGES_DIR = Path(BASE_SHARE, "HPC_Files/GAGES_Work/data_work/GAGESiiVariables")
RESULTS_DIR = Path(BASE_SHARE, "Data_Out/Results")
SHAP_DIR = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff",
    "Data_Out/SHAP_OUT",
)
FEAT_CATS_FILE = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/"
    "Data_Out/UMAP_HDBSCAN/FeatureCategories.csv"
)
FIG_DIR = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/Figures/Manuscript", "P_PET_classes"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLUST_METHODS = ["Class", "None", "AggEcoregion", "CAMELS"]
MODEL_IN = "XGBoost"
P_PET_BINS = [-np.inf, 1, np.inf]
P_PET_LABELS = ["<1", ">=1"]
ECDF_METRICS = ("NSE", "KGE")
ECDF_TIME_SCALES = ("annual", "monthly")
PLOT_ECDFS = True
SAVE_ECDF_PLOT_DEFAULT = False
WRITE_UPDATED_IDS = False
WRITE_SHAP_SUMMARY = False
# SHAP_FILE_PATTERN = "MeanShap_{partition}_{suffix}_normQ.csv"
SHAP_FILE_PATTERN = "MeanShap_BestGrouping_All_{suffix}_normQ.csv"
sns.set_theme(style="whitegrid", context="talk")


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Thin wrapper that validates file existence before reading."""
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


# %% load and prep data ----------------------------------------------------------
def load_id_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read training and validation ID tables and tag partitions."""
    df_id_train = _read_csv(Path(GAGES_DIR, "ID_train.csv"), dtype={"STAID": str})
    df_id_val = _read_csv(Path(GAGES_DIR, "ID_valnit.csv"), dtype={"STAID": str})
    df_id_train["partition"] = "train"
    df_id_val["partition"] = "valnit"
    return df_id_train, df_id_val


def load_explanatory_tables() -> pd.DataFrame:
    """
    Load explanatory variables, average across years, and attach AggEcoregion.
    """
    df_expl_train = _read_csv(Path(GAGES_DIR, "Expl_train.csv"), dtype={"STAID": str})
    df_expl_val = _read_csv(Path(GAGES_DIR, "Expl_valnit.csv"), dtype={"STAID": str})

    def _prep(df: pd.DataFrame, partition: str) -> pd.DataFrame:
        df = df.copy()
        df["STAID"] = df["STAID"].astype(str)
        grouped = (
            df.groupby("STAID")
            .mean(numeric_only=True)
            .reset_index()
        )
        grouped = grouped.drop(columns=["year"], errors="ignore")
        grouped["partition"] = partition
        return grouped

    df_expl_train = _prep(df_expl_train, "train")
    df_expl_val = _prep(df_expl_val, "valnit")

    df_expl = pd.concat([df_expl_train, df_expl_val], ignore_index=True)
    df_id_train, df_id_val = load_id_tables()
    df_id = pd.concat([df_id_train, df_id_val], ignore_index=True)
    df_expl = df_expl.merge(
        df_id[["STAID", "AggEcoregion"]],
        on="STAID",
        how="left",
    )
    return df_expl


def calculate_p_pet(df_expl: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate mean annual P:PET per basin and classify into binary categories.
    """
    if {"PPTAVG_BASIN", "PET"} - set(df_expl.columns):
        raise KeyError("Missing PPTAVG_BASIN or PET in explanatory data.")
    df_ppet = df_expl[["STAID", "PPTAVG_BASIN", "PET", "partition"]].copy()
    # Convert precip from cm to mm to match PET units.
    df_ppet["PPT_mm"] = df_ppet["PPTAVG_BASIN"] * 10
    df_ppet["P_to_PET"] = df_ppet["PPT_mm"] / df_ppet["PET"]
    df_ppet["P_PET_Class"] = pd.cut(
        df_ppet["P_to_PET"],
        bins=P_PET_BINS,
        labels=P_PET_LABELS,
        right=False,
    )
    return df_ppet[["STAID", "partition", "P_to_PET", "P_PET_Class"]]


def update_id_tables(df_ppet: pd.DataFrame) -> pd.DataFrame:
    """Attach P:PET metrics to ID tables and optionally persist the updates."""
    df_id_train, df_id_val = load_id_tables()
    df_id = pd.concat([df_id_train, df_id_val], ignore_index=True)

    if "P_to_PET" not in df_id.columns:
        df_id = df_id.merge(df_ppet, on=["STAID", "partition"], how="left")

    if WRITE_UPDATED_IDS:
        for part, path in (("train", "ID_train.csv"), ("valnit", "ID_valnit.csv")):
            df_part = df_id[df_id["partition"] == part].copy()
            df_part.to_csv(Path(GAGES_DIR, path), index=False)
    return df_id


def load_performance_results() -> pd.DataFrame:
    """Read NSE/KGE component results and subset to desired model/methods."""
    df_results = _read_csv(
        Path(RESULTS_DIR, "NSEComponents_KGE.csv"),
        dtype={"STAID": str, "region": str},
    )
    df_results = df_results.loc[
        (df_results["model"] == MODEL_IN)
        & (df_results["clust_method"].isin(CLUST_METHODS))
    ].copy()
    return df_results


# %% add P:PET class to df_ID ----------------------------------------------------
def plot_metric_ecdf(
    df_metrics: pd.DataFrame,
    df_classes: pd.DataFrame,
    metric: str,
    time_scale: str,
    save_plot: bool = SAVE_ECDF_PLOT_DEFAULT,
) -> None:
    """Plot eCDFs split by partition (line style) and P:PET class (color)."""
    print("plotting ecdfs")
    df_plot = (
        df_metrics[df_metrics["time_scale"] == time_scale]
        .merge(df_classes[["STAID", "P_PET_Class"]], on="STAID", how="left")
        .dropna(subset=["P_PET_Class"])
    )
    if df_plot.empty:
        return

    linestyles = {"train": "-", "valnit": "--"}
    style_labels = {"train": "Train", "valnit": "Test"}
    present_parts = [
        part for part in linestyles if part in df_plot["train_val"].unique()
    ]
    hue_order = sorted(df_plot["P_PET_Class"].unique())
    palette = sns.color_palette("Set2", n_colors=len(hue_order))
    fig, ax = plt.subplots(figsize=(7, 5))
    print(f"present_parts: {present_parts}")
    for part in present_parts:
        ls = linestyles[part]
        data_part = df_plot[df_plot["train_val"] == part]
        if data_part.empty:
            continue
        sns.ecdfplot(
            data=data_part,
            x=metric,
            hue="P_PET_Class",
            hue_order=hue_order,
            palette=palette,
            linewidth=2.5,
            linestyle=ls,
            ax=ax,
            legend=False,
        )

    ax.set_xlabel(metric)
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"{time_scale.capitalize()} {metric} by P:PET class")
    ax.set_xlim([-1, 1])

    color_lookup = {cls: palette[i] for i, cls in enumerate(hue_order)}
    combo_handles = []
    for part in present_parts:
        for cls in hue_order:
            combo_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color_lookup[cls],
                    linewidth=2,
                    linestyle=linestyles[part],
                    label=f"{style_labels.get(part, part.title())} ({cls})",
                )
            )
    if combo_handles:
        ax.legend(
            handles=combo_handles,
            title="Partition (P:PET class)",
            loc="upper left",
        )
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    plt.tight_layout()
    plt.show()

    if save_plot:
        fig_path = FIG_DIR / f"ECDF_{metric}_{time_scale}.png"
        fig.savefig(fig_path, dpi=300)
    plt.close(fig)


# %% plot NSE and KGE eCDFs ------------------------------------------------------
def drive_ecdf_plots(
    df_metrics: pd.DataFrame,
    df_id: pd.DataFrame,
    save_plots: bool = SAVE_ECDF_PLOT_DEFAULT,
) -> None:
    """Iterate through metrics/time-scales and trigger plotting."""
    if not PLOT_ECDFS:
        return
    df_classes = df_id[["STAID", "P_PET_Class"]].dropna()
    for metric in ECDF_METRICS:
        if metric not in df_metrics.columns:
            continue
        for time_scale in ECDF_TIME_SCALES:
            plot_metric_ecdf(
                df_metrics,
                df_classes,
                metric,
                time_scale,
                save_plot=save_plots,
            )


# %% aggregate shap values -------------------------------------------------------
def load_shap_frames() -> Dict[str, pd.DataFrame]:
    """Read train/val SHAP data for each time scale and concatenate."""
    shap_frames: Dict[str, pd.DataFrame] = {}
    scale_suffix = {
        "mean_annual": "mean_annual",
        "annual": "annual",
        "monthly": "monthly",
    }
    partitions = ("train", "valnit")
    for scale_key, suffix in scale_suffix.items():
        stacked: List[pd.DataFrame] = []
        for part in partitions:
            fn = Path(
                SHAP_DIR,
                SHAP_FILE_PATTERN.format(partition=part, suffix=suffix),
            )
            try:
                df = _read_csv(
                    fn,
                    dtype={
                        "STAID": str,
                        "site_no": str,
                        "siteid": str,
                        "SiteID": str,
                    },
                )
            except FileNotFoundError:
                continue
            df = df.copy()
            # Normalize partition naming for downstream merges.
            if "train_val" in df.columns:
                df = df.rename(columns={"train_val": "partition"})
            df["partition"] = part
            stacked.append(df)
        if stacked:
            shap_frames[scale_key] = pd.concat(stacked, ignore_index=True)
    return shap_frames


def get_feature_sets() -> Dict[str, List[str]]:
    """Return mapping of coarse categories to the relevant features."""
    feat_cats = _read_csv(FEAT_CATS_FILE)
    feature_sets: Dict[str, List[str]] = {}
    cat_mapping = {
        "Climate": "climate",
        "Physiography": "physio",
        "Anthro_Hydro": "anthro_hydro",
        "Anthro_Land": "anthro_land",
    }
    for coarse_label, set_name in cat_mapping.items():
        mask = feat_cats["Coarse_Cat"] == coarse_label
        if not mask.any():
            continue
        features = (
            feat_cats.loc[mask, "Features"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if features:
            feature_sets[set_name] = sorted(features)
    return feature_sets


def summarize_shap_by_category(
    df_shap: pd.DataFrame,
    feature_sets: Dict[str, Iterable[str]],
    df_classes: pd.DataFrame,
    scale_key: str,
) -> pd.DataFrame:
    """Sum absolute SHAP values by category/P:PET class and normalize."""
    id_col = None
    for candidate in ("STAID", "site_no", "siteid", "SiteID"):
        if candidate in df_shap.columns:
            id_col = candidate
            break
    if id_col is None:
        print(
            f"Skipping SHAP summary for {scale_key}: no STAID-like column present."
        )
        return pd.DataFrame()
    if id_col != "STAID":
        df_shap = df_shap.rename(columns={id_col: "STAID"})

    if "partition" not in df_shap.columns and "train_val" in df_shap.columns:
        df_shap = df_shap.rename(columns={"train_val": "partition"})

    if "partition" not in df_shap.columns:
        df_shap = df_shap.copy()
        df_shap["partition"] = "all"
        df_classes = df_classes.copy()
        df_classes["partition"] = "all"

    merge_keys = ["STAID", "partition"]
    df = df_shap.merge(df_classes, on=merge_keys, how="left")
    df = df.dropna(subset=["P_PET_Class"])
    if df.empty:
        return pd.DataFrame()

    cat_cols = []
    for cat, features in feature_sets.items():
        cols = [col for col in features if col in df.columns]
        if not cols:
            continue
        col_name = f"abs_shap_{cat}"
        df[col_name] = df[cols].abs().sum(axis=1)
        cat_cols.append(col_name)

    if not cat_cols:
        return pd.DataFrame()

    group_fields = ["Class", "P_PET_Class", "partition"]
    grouped = (
        df[group_fields + cat_cols]
        .groupby(group_fields, dropna=False)
        .sum()
        .reset_index()
    )
    total = grouped[cat_cols].sum(axis=1)
    total = total.replace(0, np.nan)
    grouped[cat_cols] = grouped[cat_cols].div(total, axis=0)
    grouped["time_scale"] = scale_key
    return grouped


def aggregate_all_shap(
    shap_frames: Dict[str, pd.DataFrame], df_classes: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate SHAP summaries for every time scale and optionally persist."""
    feature_sets = get_feature_sets()
    summaries = []
    for scale_key, df_shap in shap_frames.items():
        summ = summarize_shap_by_category(df_shap, feature_sets, df_classes, scale_key)
        if not summ.empty:
            summaries.append(summ)
    if not summaries:
        return pd.DataFrame()
    df_summary = pd.concat(summaries, ignore_index=True)
    if WRITE_SHAP_SUMMARY:
        df_summary.to_csv(Path(FIG_DIR, "shap_summary_by_P_PET.csv"), index=False)
    return df_summary


def plot_shap_category_bars(df_summary: pd.DataFrame) -> None:
    """Plot stacked bars of normalized SHAP sums by time scale/class/partition."""
    if df_summary.empty:
        return

    cat_cols = [col for col in df_summary.columns if col.startswith("abs_shap_")]
    if not cat_cols:
        return

    # Fixed order/color mapping for manuscript styling.
    ordered_cols = [
        "abs_shap_climate",
        "abs_shap_physio",
        "abs_shap_anthro_hydro",
        "abs_shap_anthro_land",
    ]
    cat_cols = [col for col in ordered_cols if col in cat_cols]
    if not cat_cols:
        return
    cat_labels = ["Climate", "Physiographic", "AnthroHydro", "AnthroLand"]
    color = ["blue", "saddlebrown", "red", "black"]
    color_map = {col: color[i] for i, col in enumerate(cat_cols)}
    label_map = {col: cat_labels[i] for i, col in enumerate(cat_cols)}

    default_part_order = ["train", "valnit", "test", "all"]
    time_order = ["mean_annual", "annual", "monthly"]

    for basin_class, df_bc in df_summary.groupby("Class"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        for idx, time_scale in enumerate(time_order):
            ax = axes[idx]
            df_ts = df_bc[df_bc["time_scale"] == time_scale].copy()
            if df_ts.empty:
                ax.set_visible(False)
                continue

            class_order = [
                cls for cls in P_PET_LABELS if cls in df_ts["P_PET_Class"].unique()
            ]
            class_order += [
                cls
                for cls in df_ts["P_PET_Class"].unique()
                if cls not in class_order
            ]
            part_order = [
                p for p in default_part_order if p in df_ts["partition"].unique()
            ]
            part_order += [
                p for p in df_ts["partition"].unique() if p not in part_order
            ]

            combos: List[Tuple[str, str]] = []
            for cls in class_order:
                for part in part_order:
                    mask = (
                        (df_ts["P_PET_Class"] == cls)
                        & (df_ts["partition"] == part)
                    )
                    if mask.any():
                        combos.append((cls, part))
            if not combos:
                ax.set_visible(False)
                continue

            x = np.arange(len(combos))
            bottoms = np.zeros(len(combos))
            for col in cat_cols:
                heights = []
                for cls, part in combos:
                    mask = (
                        (df_ts["P_PET_Class"] == cls)
                        & (df_ts["partition"] == part)
                    )
                    heights.append(df_ts.loc[mask, col].iloc[0] if mask.any() else 0)
                heights = np.array(heights)
                ax.bar(
                    x,
                    heights,
                    bottom=bottoms,
                    color=color_map[col],
                    label=label_map[col],
                    width=0.8,
                    edgecolor="none",
                    linewidth=0,
                )
                bottoms += heights

            x_labels = [f"{part}\n{cls}" for cls, part in combos]
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.set_title(time_scale.replace("_", " ").title())
            ax.set_ylim(0, 1.05)
            ax.annotate(
                f"({chr(97 + idx)})",
                xy=(1.02, 1.01),
                xycoords="axes fraction",
                ha="right",
                va="top",
                clip_on=False,
            )

            if idx == 0:
                ax.set_ylabel("Relative contribution of each variable type")
            else:
                ax.legend_.remove() if ax.legend_ else None

        axes[0].legend(
            title="Feature group",
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
        )
        plt.tight_layout()
        plt.show()


# %% orchestration ---------------------------------------------------------------
# def main() -> None:
df_expl = load_explanatory_tables()
df_ppet = calculate_p_pet(df_expl)
df_id = update_id_tables(df_ppet)
df_metrics = load_performance_results()
shap_frames = load_shap_frames()
drive_ecdf_plots(df_metrics, df_id, save_plots=False)
df_summary = aggregate_all_shap(shap_frames, df_id)

if df_summary.empty:
    print("No SHAP summaries generated.")
else:
    cols = [
        'abs_shap_climate',
        'abs_shap_physio',
        'abs_shap_anthro_hydro',
        'abs_shap_anthro_land',
    ]
    if all(col in df_summary for col in cols):
        assert np.isclose(df_summary[cols].sum(axis=1), 1).all(), "not all rows sum to 1"
    print("Generated SHAP summary table with shape:", df_summary.shape)
    plot_shap_category_bars(df_summary)


# if __name__ == "__main__":
#     main()
