"""
BChoat 2026/05

Plot SHAP stacked bar products from Build_SHAP_Products.py outputs.

Products:
1. Regional stacked bars, using either XGBoost-only regional summaries or
   best-regional-model summaries.
2. P:PET stacked bars, using the by-prediction SHAP table and basin-level
   P:PET classes.
"""

# %%
import re
import glob
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from Load_Data import load_data_fun


# %% define dirs, vars, and such -------------------------------------------------
dir_shap = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/SHAP_OUT"
)
dir_results = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results"
)
dir_workHPC = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/"
    "HPC_Files/GAGES_Work/data_out"
)
dir_vars = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/"
    "Data_Out/AllVars_Partitioned"
)
feat_cats_file = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/"
    "Data_Out/UMAP_HDBSCAN/FeatureCategories.csv"
)
dir_figs = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/Figures/Manuscript/SHAP_Products"
)
dir_figs.mkdir(parents=True, exist_ok=True)

OUT_TAG = "_202605"

time_scales = ["mean_annual", "annual", "monthly"]
part_in = ["train", "valnit"]

# "XGBoostOnly" or "BestRegionalModel_medianNSE"
REGION_PRODUCT = "XGBoostOnly"

# P:PET plots are built from by-prediction rows. Use "XGBoost" for the
# XGBoost-only manuscript product. Use "best_median_NSE" to plot rows from the
# best regional model where NSE exists.
P_PET_MODEL_SCOPE = "XGBoost"
P_PET_CLUST_METHODS = ["AggEcoregion"]

WRITE_SUMMARY_TABLES = True
SAVE_FIGS = True
RELOAD_SHAP_INPUTS = True
BAR_WIDTH = 0.62
BAR_X_STEP = 0.68
P_PET_CLASS_GAP = 0.22
ANNOTATE_BAR_PERCENTAGES = True
ANNOTATE_FONTSIZE_REGION = 8
ANNOTATE_FONTSIZE_P_PET = 12
ANNOTATE_FONTSIZE_REGION_P_PET = 12
PANEL_TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE_P_PET = 10
TICK_LABEL_FONTSIZE_REGION_P_PET = 15
BAR_X_LABEL_ROTATION = 60
PANEL_TITLE_FONTSIZE_REGION_P_PET = 18
AXIS_LABEL_FONTSIZE_REGION_P_PET = 16
LEGEND_FONTSIZE = 9
LEGEND_FONTSIZE_REGION_P_PET = 13
LEGEND_NCOL = 2
LEGEND_NCOL_REGION_P_PET = 4
ANNOTATE_MIN_FRACTION = 0.04  # 0.02
ANNOTATE_SMALL_SEGMENTS_OUTSIDE = False
ANNOTATE_OUTSIDE_MIN_FRACTION = 0.01
ANNOTATE_OUTSIDE_X_OFFSET = 0.2

P_PET_BINS = [-np.inf, 1, np.inf]
P_PET_LABELS = ["P:PET<1", "P:PET≥1"]
HEATMAP_PRECIP_VMIN = 0
HEATMAP_PRECIP_VMAX = 0.64
HEATMAP_OTHER_VMIN = -0.15
HEATMAP_OTHER_VMAX = 0.15
HEATMAP_N_FEATURES = 30
HEATMAP_ANTHRO_VMIN = -0.025
HEATMAP_ANTHRO_VMAX = 0.025
HEATMAP_ANTHRO_N_FEATURES = 40
PLOT_ALL_VARIABLE_HEATMAPS = True
PLOT_ANTHRO_HEATMAPS = True
HEATMAP_VARIABLE_LABEL_FONTSIZE = 11
HEATMAP_CATEGORY_LABEL_FONTSIZE = 11
HEATMAP_AXIS_TITLE_FONTSIZE = 11
HEATMAP_CATEGORY_LABEL_ROTATION = 90
# Use [False], [True], or [False, True].
HEATMAP_INCLUDE_P_PET_OPTIONS = [True]  # [False, True]
# Options: "regression_slope" or "spearman"
HEATMAP_DIRECTION_METHOD = "spearman"
USE_HEATMAP_DIRECTION_CACHE = True
WRITE_HEATMAP_DIRECTION_CACHE = True

sns.set_theme(style="whitegrid", context="talk")

feature_colors = {
    "Climate": "blue",
    "Physiography": "saddlebrown",
    "Anthro_Hydro": "red",
    "Anthro_Land": "black",
}

feature_label = {
    "Climate": "Climate",
    "Physiography": "Physiographic",
    "Anthro_Hydro": "AnthroHydro",
    "Anthro_Land": "AnthroLand",
}

lookback_feature_labels = {
    "prcp": "Ant Precip",
    "swe": "Ant SWE",
    "vp": "Ant Vapor Pressure",
    "tmax": "Ant Max Temp",
    "tmin": "Ant Min Temp",
}

metadata_cols = {
    "STAID",
    "year",
    "month",
    "date",
    "WY_cm",
    "y_obs",
    "y_pred",
    "shap_norm",
    "n_predictions",
    "NSE",
    "KGE",
    "r",
    "alpha",
    "beta",
    "residuals",
    "|residuals|",
    "RMSE",
    "PercBias",
    "model",
    "best_model",
    "median_NSE",
    "metric_in",
    "metric_summary",
    "aggregation",
    "time_scale",
    "train_val",
    "clust_method",
    "clust_meth",
    "region",
    "P_to_PET",
    "P_PET_Class",
    "partition",
    "Class",
    "AggEcoregion",
}


# %% helpers --------------------------------------------------------------------
def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def get_feature_sets() -> Dict[str, List[str]]:
    """Return generic feature-category mappings from FeatureCategories.csv."""
    feat_cats = _read_csv(feat_cats_file)
    feature_sets: Dict[str, List[str]] = {}
    for cat in feature_colors:
        features: List[str] = []
        for col in ["Features", "Alias"]:
            if col in feat_cats.columns:
                features.extend(
                    feat_cats.loc[feat_cats["Coarse_Cat"] == cat, col]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
        feature_sets[cat] = sorted(set(features))
    return feature_sets


def expand_feature_sets_to_columns(
    feature_sets: Dict[str, Iterable[str]],
    available_cols: Iterable[str],
) -> Dict[str, List[str]]:
    """
    Expand generic feature names to matching lookback columns.

    For example, if FeatureCategories.csv lists `prcp`, and the SHAP table has
    `prcp`, `prcp_1`, and `prcp_2`, all three columns are assigned to the same
    feature category.
    """
    available_cols = list(available_cols)
    expanded: Dict[str, List[str]] = {}
    for cat, features in feature_sets.items():
        cols = []
        for feature in features:
            feature = str(feature)
            if feature in available_cols:
                cols.append(feature)

            # Only expand generic base names. If the category file ever contains
            # an explicit lookback name like prcp_1, do not expand it again.
            if re.search(r"_\d+$", feature):
                continue

            lookback_pattern = re.compile(rf"^{re.escape(feature)}_\d+$")
            cols.extend(
                col for col in available_cols if lookback_pattern.match(str(col))
            )

        expanded[cat] = sorted(set(cols))
    return expanded


def feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns that look like SHAP feature columns."""
    return [
        col
        for col in df.columns
        if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
    ]


def add_feature_group_sums(
    df: pd.DataFrame,
    feature_sets: Dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Sum absolute SHAP values into broad feature categories."""
    df = df.copy()
    shap_cols = set(feature_columns(df))
    feature_sets = expand_feature_sets_to_columns(feature_sets, shap_cols)
    group_cols = []
    for cat, features in feature_sets.items():
        cols = [col for col in features if col in shap_cols]
        if not cols:
            continue
        out_col = f"shap_{cat}"
        df[out_col] = df[cols].abs().sum(axis=1)
        group_cols.append(out_col)
    if not group_cols:
        raise ValueError("No SHAP feature columns matched FeatureCategories.csv")
    return df


def normalize_group_contrib(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize shap_* columns to row sums of one."""
    df = df.copy()
    cat_cols = [col for col in df.columns if col.startswith("shap_")]
    total = df[cat_cols].sum(axis=1).replace(0, np.nan)
    df[cat_cols] = df[cat_cols].div(total, axis=0)
    return df


def normalize_bar_columns(df_bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize selected bar columns to row sums of one."""
    total = df_bars.sum(axis=1).replace(0, np.nan)
    return df_bars.div(total, axis=0)


def contrast_text_color(color: str) -> str:
    """Choose black or white text based on fill-color luminance."""
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.55 else "white"


def annotate_stacked_percentages(
    ax: plt.Axes,
    df_bars: pd.DataFrame,
    colors: List[str],
    outside_small_segments: bool = True,
    x_positions: List[float] = None,
    fontsize: int = ANNOTATE_FONTSIZE_REGION,
) -> None:
    """Annotate stacked bars, moving small segments outside the bars."""
    if not ANNOTATE_BAR_PERCENTAGES:
        return

    small_labels = {}
    bar_width = BAR_WIDTH
    if x_positions is None:
        x_positions = list(range(df_bars.shape[0]))
    for bar_idx, (_, row) in enumerate(df_bars.fillna(0).iterrows()):
        x_pos = x_positions[bar_idx]
        y_bottom = 0.0
        for idx, col in enumerate(df_bars.columns):
            value = row[col]
            if value < ANNOTATE_MIN_FRACTION:
                if (
                    ANNOTATE_SMALL_SEGMENTS_OUTSIDE
                    and outside_small_segments
                    and ANNOTATE_OUTSIDE_MIN_FRACTION <= value
                ):
                    small_labels.setdefault(bar_idx, []).append(
                        {
                            "x_right": x_pos + bar_width / 2,
                            "y_mid": y_bottom + value / 2,
                            "label": f"{value * 100:.0f}%",
                            "color": colors[idx],
                        }
                    )
                y_bottom += value
                continue

            ax.text(
                x_pos,
                y_bottom + value / 2,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                color=contrast_text_color(colors[idx]),
                fontsize=fontsize,
                zorder=5,
                clip_on=False,
            )
            y_bottom += value

    for labels_for_bar in small_labels.values():
        labels_for_bar = sorted(labels_for_bar, key=lambda item: item["y_mid"])
        min_gap = 0.055
        last_y = -np.inf
        for item in labels_for_bar:
            y_text = max(item["y_mid"], last_y + min_gap)
            y_text = min(y_text, 1.02)
            last_y = y_text
            ax.annotate(
                item["label"],
                xy=(item["x_right"], item["y_mid"]),
                xytext=(item["x_right"] + ANNOTATE_OUTSIDE_X_OFFSET, y_text),
                textcoords="data",
                ha="right",
                va="center",
                fontsize=fontsize,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": item["color"],
                    "linewidth": 0.7,
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": item["color"],
                    "linewidth": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                clip_on=False,
            )

    x_min, x_max = ax.get_xlim()
    if small_labels:
        ax.set_xlim(x_min, x_max + 0.04)


def plot_stacked_bars(
    ax: plt.Axes,
    df_bars: pd.DataFrame,
    colors: List[str],
    x_positions: List[float] = None,
) -> List[float]:
    """Plot stacked bars with explicit x positions for tighter spacing."""
    if x_positions is None:
        x_positions = [i * BAR_X_STEP for i in range(df_bars.shape[0])]

    bottoms = np.zeros(df_bars.shape[0])
    for idx, col in enumerate(df_bars.columns):
        values = df_bars[col].fillna(0).to_numpy()
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            color=colors[idx],
            edgecolor="none",
            width=BAR_WIDTH,
        )
        bottoms = bottoms + values

    ax.set_xticks(x_positions)
    ax.set_xticklabels(df_bars.index)
    ax.set_xlim(
        min(x_positions) - BAR_WIDTH / 2 - 0.08,
        max(x_positions) + BAR_WIDTH / 2 + 0.08,
    )
    return x_positions


def rotate_bar_xticklabels(ax: plt.Axes, fontsize: int) -> None:
    """Rotate bar x tick labels while keeping each label anchored to its tick."""
    ax.tick_params(axis="x", labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_rotation(BAR_X_LABEL_ROTATION)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("top")


def rotate_heatmap_xticklabels(ax: plt.Axes) -> None:
    """Rotate heatmap x tick labels while keeping labels anchored to columns."""
    ax.tick_params(axis="x", labelsize=HEATMAP_CATEGORY_LABEL_FONTSIZE)
    for label in ax.get_xticklabels():
        label.set_rotation(HEATMAP_CATEGORY_LABEL_ROTATION)
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("top")


def read_region_summary_files() -> pd.DataFrame:
    """Read regional SHAP summary files for all configured partitions/times."""
    frames = []
    for timescale in time_scales:
        for partition in part_in:
            path = Path(
                dir_shap,
                f"MeanShap_{REGION_PRODUCT}_{partition}_{timescale}_normQ"
                f"{OUT_TAG}.csv",
            )
            try:
                df = _read_csv(path, dtype={"region": str})
            except FileNotFoundError:
                print(f"Missing regional SHAP file: {path}")
                continue
            df["time_scale"] = timescale
            df["train_val"] = partition
            frames.append(df)
            print(f"Loaded regional SHAP file: {path}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_region_categories(df_region: pd.DataFrame) -> pd.DataFrame:
    """Convert regional feature SHAP columns to normalized category columns."""
    feature_sets = get_feature_sets()
    df = add_feature_group_sums(df_region, feature_sets)
    keep_cols = [
        "time_scale",
        "train_val",
        "clust_method",
        "region",
        "n_predictions",
        "best_model",
        "median_NSE",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    cat_cols = [col for col in df.columns if col.startswith("shap_")]
    df = df[keep_cols + cat_cols].copy()
    return normalize_group_contrib(df)


def plot_region_bars(df_summary: pd.DataFrame) -> None:
    """Plot one combined regional stacked-bar figure for each partition."""
    if df_summary.empty:
        print("No regional summary rows to plot.")
        return

    cat_cols = [col for col in df_summary.columns if col.startswith("shap_")]
    colors = [feature_colors[col.replace("shap_", "")] for col in cat_cols]
    labels = [feature_label[col.replace("shap_", "")] for col in cat_cols]
    clust_order = ["None", "Class", "AggEcoregion"]

    for partition in part_in:
        df_part = df_summary[df_summary["train_val"] == partition].copy()
        if df_part.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(10, 6), sharey=True)
        for idx, timescale in enumerate(time_scales):
            ax = axes[idx]
            df_plot = df_part[df_part["time_scale"] == timescale].copy()
            if df_plot.empty:
                ax.set_visible(False)
                continue

            df_plot["clust_order"] = df_plot["clust_method"].map(
                {clust: i for i, clust in enumerate(clust_order)}
            )
            df_plot["clust_order"] = df_plot["clust_order"].fillna(99)
            df_plot = df_plot.sort_values(["clust_order", "region"])
            df_plot["plot_label"] = df_plot["region"].astype(str)

            df_bars = df_plot.set_index("plot_label")[cat_cols]
            x_positions = plot_stacked_bars(ax, df_bars, colors)
            annotate_stacked_percentages(
                ax,
                df_bars,
                colors,
                outside_small_segments=True,
                x_positions=x_positions,
                fontsize=ANNOTATE_FONTSIZE_REGION,
            )
            ax.set_title(
                timescale.replace("_", " ").title(),
                fontsize=PANEL_TITLE_FONTSIZE,
            )
            ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylim(0, 1.05)
            rotate_bar_xticklabels(ax, TICK_LABEL_FONTSIZE)
            ax.tick_params(
                axis="y",
                labelsize=TICK_LABEL_FONTSIZE,
            )
            if idx == 0:
                ax.set_ylabel(
                    "Relative contribution",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            else:
                ax.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)

            # Add light separators between None, Class, and AggEcoregion bars.
            method_counts = df_plot.groupby("clust_method", sort=False).size()
            xpos = 0
            for _, count in method_counts.iloc[:-1].items():
                xpos += count
                boundary = (x_positions[xpos - 1] + x_positions[xpos]) / 2
                ax.axvline(boundary, color="0.7", linewidth=0.8)

        handles = [
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(colors, labels)
        ]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=LEGEND_NCOL,
            fontsize=LEGEND_FONTSIZE,
            columnspacing=0.9,
            handletextpad=0.5,
            borderpad=0.3,
        )
        # fig.suptitle(f"{REGION_PRODUCT}: {partition}")
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        if SAVE_FIGS:
            fig.savefig(
                dir_figs / f"SHAP_region_{REGION_PRODUCT}_{partition}_combined.png",
                dpi=300,
            )
        plt.show()


def load_p_pet_classes() -> pd.DataFrame:
    """Calculate mean annual P:PET class for train and validation basins."""
    frames = []
    for partition in part_in:
        df_expl = _read_csv(
            Path(dir_vars, f"Expl_{partition}.csv"),
            dtype={"STAID": str},
        )
        df_id = _read_csv(
            Path(dir_vars, f"ID_{partition}.csv"),
            dtype={"STAID": str},
        )
        df_expl = (
            df_expl.groupby("STAID")
            .mean(numeric_only=True)
            .reset_index()
            .drop(columns=["year"], errors="ignore")
        )
        df_ppet = df_expl[["STAID", "PPTAVG_BASIN", "PET"]].copy()
        df_ppet["P_to_PET"] = (df_ppet["PPTAVG_BASIN"] * 10) / df_ppet["PET"]
        df_ppet["P_PET_Class"] = pd.cut(
            df_ppet["P_to_PET"],
            bins=P_PET_BINS,
            labels=P_PET_LABELS,
            right=False,
        )
        df_ppet["train_val"] = partition
        id_cols = [col for col in ["STAID", "Class", "AggEcoregion"] if col in df_id]
        df_ppet = df_ppet.merge(df_id[id_cols], on="STAID", how="left")
        frames.append(df_ppet)
    return pd.concat(frames, ignore_index=True)


def read_by_prediction_files() -> pd.DataFrame:
    """Read by-prediction SHAP files, projecting/filtering parquet when possible."""
    frames = []
    feature_sets = get_feature_sets()
    meta_cols_needed = [
        "STAID",
        "year",
        "month",
        "date",
        "train_val",
        "model",
        "time_scale",
        "clust_method",
        "region",
        "NSE",
    ]
    parquet_filters = None
    if P_PET_MODEL_SCOPE == "XGBoost":
        parquet_filters = [("model", "==", "XGBoost")]

    for timescale in time_scales:
        parquet_path = Path(
            dir_shap,
            f"SHAP_ByPrediction_{timescale}_allModels_normQ{OUT_TAG}.parquet",
        )
        csv_path = Path(
            dir_shap,
            f"SHAP_ByPrediction_{timescale}_allModels_normQ{OUT_TAG}.csv",
        )

        if parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                file_cols = set(pq.read_schema(parquet_path).names)
                feature_cols_needed = sorted(
                    set(
                        feature
                        for features in expand_feature_sets_to_columns(
                            feature_sets, file_cols
                        ).values()
                        for feature in features
                    )
                )
                cols_needed = [
                    col for col in meta_cols_needed + feature_cols_needed
                    if col in file_cols
                ]
                df = pd.read_parquet(
                    parquet_path,
                    columns=cols_needed,
                    filters=parquet_filters,
                )
            except (KeyError, ValueError):
                # Some files may not contain every alias/feature listed in
                # FeatureCategories.csv. Fall back to reading the file schema,
                # then project only columns that are actually present.
                try:
                    import pyarrow.parquet as pq

                    file_cols = set(pq.read_schema(parquet_path).names)
                    cols_existing = [col for col in cols_needed if col in file_cols]
                    df = pd.read_parquet(
                        parquet_path,
                        columns=cols_existing,
                        filters=parquet_filters,
                    )
                except ImportError:
                    df = pd.read_parquet(parquet_path, filters=parquet_filters)
            except ImportError:
                df = pd.read_parquet(parquet_path, filters=parquet_filters)
            for col in ["STAID", "region"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            print(f"Loaded by-prediction SHAP parquet: {parquet_path}")
        elif csv_path.exists():
            file_cols = set(pd.read_csv(csv_path, nrows=0).columns)
            feature_cols_needed = sorted(
                set(
                    feature
                    for features in expand_feature_sets_to_columns(
                        feature_sets, file_cols
                    ).values()
                    for feature in features
                )
            )
            cols_needed = [
                col for col in meta_cols_needed + feature_cols_needed
                if col in file_cols
            ]
            df = _read_csv(
                csv_path,
                dtype={"STAID": str, "region": str},
                usecols=lambda col: col in cols_needed,
            )
            if P_PET_MODEL_SCOPE == "XGBoost" and "model" in df.columns:
                df = df[df["model"] == "XGBoost"].copy()
            print(f"Loaded by-prediction SHAP CSV: {csv_path}")
        else:
            print(f"Missing by-prediction SHAP file: {parquet_path}")
            print(f"Missing by-prediction SHAP file: {csv_path}")
            continue

        df["time_scale"] = timescale
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def filter_p_pet_model_scope(df: pd.DataFrame) -> pd.DataFrame:
    """Filter by-prediction rows to the requested model scope."""
    if P_PET_MODEL_SCOPE == "XGBoost":
        return df[df["model"] == "XGBoost"].copy()

    if P_PET_MODEL_SCOPE != "best_median_NSE":
        raise ValueError(
            "P_PET_MODEL_SCOPE must be 'XGBoost' or 'best_median_NSE'. "
            f"Got: {P_PET_MODEL_SCOPE}"
        )

    if "NSE" not in df.columns:
        raise ValueError("Cannot use best_median_NSE scope without an NSE column.")

    score_cols = ["time_scale", "train_val", "clust_method", "region", "model"]
    df_scores = (
        df.dropna(subset=["NSE"])
        .groupby(score_cols, as_index=False)["NSE"]
        .median()
        .rename(columns={"NSE": "median_NSE"})
    )
    idx_best = df_scores.groupby(score_cols[:-1])["median_NSE"].idxmax()
    df_best = df_scores.loc[idx_best, score_cols].reset_index(drop=True)
    return df.merge(df_best, on=score_cols, how="inner")


def summarize_p_pet_categories(df_by_pred: pd.DataFrame) -> pd.DataFrame:
    """Summarize by-prediction SHAP values by P:PET class."""
    if df_by_pred.empty:
        return pd.DataFrame()

    df_classes = load_p_pet_classes()
    df = filter_p_pet_model_scope(df_by_pred)
    if P_PET_CLUST_METHODS:
        df = df[df["clust_method"].isin(P_PET_CLUST_METHODS)].copy()
    df = df.merge(
        df_classes[
            ["STAID", "train_val", "P_to_PET", "P_PET_Class", "Class", "AggEcoregion"]
        ],
        on=["STAID", "train_val"],
        how="left",
    )
    df = df.dropna(subset=["P_PET_Class"]).copy()
    if df.empty:
        return pd.DataFrame()

    feature_sets = get_feature_sets()
    df = add_feature_group_sums(df, feature_sets)
    cat_cols = [col for col in df.columns if col.startswith("shap_")]

    group_cols = ["time_scale", "train_val", "P_PET_Class"]
    rows = []
    for keys, df_group in df.groupby(group_cols, dropna=False):
        out = dict(zip(group_cols, keys))
        out["n_predictions"] = df_group.shape[0]
        for col in cat_cols:
            out[col] = df_group[col].sum()
        rows.append(out)
    df_summary = pd.DataFrame(rows)
    return normalize_group_contrib(df_summary)


def plot_p_pet_bars(df_summary: pd.DataFrame) -> None:
    """Plot one P:PET stacked-bar figure with train and validation together."""
    if df_summary.empty:
        print("No P:PET summary rows to plot.")
        return

    cat_cols = [col for col in df_summary.columns if col.startswith("shap_")]
    colors = [feature_colors[col.replace("shap_", "")] for col in cat_cols]
    labels = [feature_label[col.replace("shap_", "")] for col in cat_cols]

    fig, axes = plt.subplots(1, 3, figsize=(8, 6), sharey=True)
    partition_label = {"train": "train", "valnit": "test"}
    combo_order = [
        (partition, ppet_class)
        for ppet_class in P_PET_LABELS
        for partition in part_in
    ]

    for idx, timescale in enumerate(time_scales):
        ax = axes[idx]
        df_plot = df_summary[df_summary["time_scale"] == timescale].copy()
        if df_plot.empty:
            ax.set_visible(False)
            continue

        df_plot = df_plot.set_index(["train_val", "P_PET_Class"])
        df_plot = df_plot.reindex(combo_order)
        df_bars = df_plot[cat_cols].fillna(0)
        df_bars.index = [
            f"{partition_label.get(partition, partition)}\n{ppet_class}"
            for partition, ppet_class in combo_order
        ]
        x_positions = [
            0,
            BAR_X_STEP,
            (2 * BAR_X_STEP) + P_PET_CLASS_GAP,
            (3 * BAR_X_STEP) + P_PET_CLASS_GAP,
        ]
        x_positions = plot_stacked_bars(ax, df_bars, colors, x_positions=x_positions)
        annotate_stacked_percentages(
            ax,
            df_bars,
            colors,
            outside_small_segments=True,
            x_positions=x_positions,
            fontsize=ANNOTATE_FONTSIZE_P_PET,
        )
        ax.set_title(
            timescale.replace("_", " ").title(),
            fontsize=PANEL_TITLE_FONTSIZE,
        )
        ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylim(0, 1.05)
        rotate_bar_xticklabels(ax, TICK_LABEL_FONTSIZE_P_PET)
        ax.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE_P_PET)
        ax.axvline((x_positions[1] + x_positions[2]) / 2, color="0.7", linewidth=0.8)
        if idx == 0:
            ax.set_ylabel(
                "Relative contribution",
                fontsize=AXIS_LABEL_FONTSIZE,
            )
        else:
            ax.set_ylabel("", fontsize=AXIS_LABEL_FONTSIZE)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(colors, labels)
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=LEGEND_NCOL,
        fontsize=LEGEND_FONTSIZE,
        columnspacing=0.9,
        handletextpad=0.5,
        borderpad=0.3,
    )
    # fig.suptitle(f"P:PET SHAP categories: {P_PET_MODEL_SCOPE}")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    if SAVE_FIGS:
        fig.savefig(
            dir_figs / f"SHAP_PPET_{P_PET_MODEL_SCOPE}_train_valnit_combined.png",
            dpi=300,
        )
    plt.show()


def summarize_p_pet_for_combined_region_plot(
    df_p_pet_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Convert P:PET summary rows to region-summary-like rows."""
    if df_p_pet_summary.empty:
        return pd.DataFrame()

    df = df_p_pet_summary.copy()
    df["clust_method"] = "P:PET"
    df["region"] = df["P_PET_Class"].astype(str)
    df["source_group"] = "P:PET"
    return df


def plot_region_and_p_pet_bars(
    df_region_summary: pd.DataFrame,
    df_p_pet_summary: pd.DataFrame,
    exclude_climate: bool = False,
) -> None:
    """
    Plot regional and P:PET stacked bars together.

    One figure is produced for each partition. Time scales are stacked in one
    column to improve label legibility.
    """
    if df_region_summary.empty or df_p_pet_summary.empty:
        print("Combined region/P:PET plot skipped: missing summary rows.")
        return

    cat_cols = [col for col in df_region_summary.columns if col.startswith("shap_")]
    if exclude_climate:
        cat_cols = [col for col in cat_cols if col != "shap_Climate"]
    if not cat_cols:
        print("Combined region/P:PET plot skipped: no category columns to plot.")
        return
    colors = [feature_colors[col.replace("shap_", "")] for col in cat_cols]
    labels = [feature_label[col.replace("shap_", "")] for col in cat_cols]
    clust_order = ["None", "Class", "AggEcoregion", "P:PET"]

    df_region = df_region_summary.copy()
    df_region["source_group"] = df_region["clust_method"]
    df_ppet = summarize_p_pet_for_combined_region_plot(df_p_pet_summary)
    df_combined = pd.concat(
        [df_region, df_ppet[df_region.columns.intersection(df_ppet.columns)]],
        ignore_index=True,
    )

    for partition in part_in:
        df_part = df_combined[df_combined["train_val"] == partition].copy()
        if df_part.empty:
            continue

        fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, sharey=True)
        for idx, timescale in enumerate(time_scales):
            ax = axes[idx]
            df_plot = df_part[df_part["time_scale"] == timescale].copy()
            if df_plot.empty:
                ax.set_visible(False)
                continue

            df_plot["clust_order"] = df_plot["clust_method"].map(
                {clust: i for i, clust in enumerate(clust_order)}
            )
            df_plot["clust_order"] = df_plot["clust_order"].fillna(99)
            df_plot = df_plot.sort_values(["clust_order", "region"])
            df_plot["plot_label"] = df_plot["region"].astype(str)

            df_bars = normalize_bar_columns(df_plot.set_index("plot_label")[cat_cols])
            x_positions = plot_stacked_bars(ax, df_bars, colors)
            annotate_stacked_percentages(
                ax,
                df_bars,
                colors,
                outside_small_segments=False,
                x_positions=x_positions,
                fontsize=ANNOTATE_FONTSIZE_REGION_P_PET,
            )
            ax.set_title(
                timescale.replace("_", " ").title(),
                fontsize=PANEL_TITLE_FONTSIZE_REGION_P_PET,
            )
            ax.set_xlabel("", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylim(0, 1.05)
            rotate_bar_xticklabels(ax, TICK_LABEL_FONTSIZE_REGION_P_PET)
            ax.tick_params(
                axis="y",
                labelsize=TICK_LABEL_FONTSIZE_REGION_P_PET,
            )
            ax.set_ylabel(
                "Relative contribution" if idx == 1 else "",
                fontsize=AXIS_LABEL_FONTSIZE_REGION_P_PET,
            )

            method_counts = df_plot.groupby("clust_method", sort=False).size()
            xpos = 0
            for _, count in method_counts.iloc[:-1].items():
                xpos += count
                boundary = (x_positions[xpos - 1] + x_positions[xpos]) / 2
                ax.axvline(boundary, color="0.7", linewidth=0.8)

        handles = [
            Patch(facecolor=color, edgecolor="none", label=label)
            for color, label in zip(colors, labels)
        ]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=LEGEND_NCOL_REGION_P_PET,
            fontsize=LEGEND_FONTSIZE_REGION_P_PET,
            columnspacing=0.9,
            handletextpad=0.5,
            borderpad=0.3,
        )
        fig.tight_layout(rect=[0, 0.075, 1, 1])
        if SAVE_FIGS:
            climate_tag = "_noClimate" if exclude_climate else ""
            fig.savefig(
                dir_figs
                / (
                    f"SHAP_region_PPET_{REGION_PRODUCT}_{partition}"
                    f"_vertical{climate_tag}.png"
                ),
                dpi=300,
            )
        plt.show()


def feature_aliases_and_categories() -> tuple[Dict[str, str], Dict[str, str]]:
    """Return feature aliases and alias-level feature categories."""
    feat_cats = _read_csv(feat_cats_file)
    alias_map: Dict[str, str] = {}
    category_map: Dict[str, str] = {}
    for _, row in feat_cats.iterrows():
        feature = str(row.get("Features", "")).replace("TS_", "")
        alias = str(row.get("Alias", feature)).replace("TS_", "")
        category = row.get("Coarse_Cat")
        if feature and feature != "nan":
            alias_map[feature] = alias
        if alias and alias != "nan" and category in feature_colors:
            category_map[alias] = category
    for label in lookback_feature_labels.values():
        category_map[label] = "Climate"
    return alias_map, category_map


def collapse_lookback_heatmap_columns(df_values: pd.DataFrame) -> pd.DataFrame:
    """Collapse lookback SHAP columns into the antecedent variables used in plots."""
    df = df_values.copy()
    drop_cols = []
    for base_feature, label in lookback_feature_labels.items():
        pattern = re.compile(rf"^{re.escape(base_feature)}_\d+$")
        cols = [col for col in df.columns if pattern.match(str(col))]
        if cols:
            df[label] = df[cols].sum(axis=1, min_count=1)
            drop_cols.extend(cols)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def prep_xy_for_heatmap_direction(
    timescale: str,
    partition: str,
    clust_meth: str,
    region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load explanatory values and ID columns for old-style SHAP direction signs."""
    df_expl, df_wy, _ = load_data_fun(
        dir_work=str(dir_results),
        time_scale=timescale,
        train_val=partition,
        clust_meth=clust_meth,
        region=str(region),
        standardize=False,
    )
    df_expl = df_expl.copy()
    if "SQKM_x" in df_expl.columns:
        df_expl.columns = df_expl.columns.str.replace("SQKM_x", "SQKM")

    if timescale == "mean_annual":
        id_cols = ["STAID"]
    elif timescale == "annual":
        id_cols = ["STAID", "year"]
    elif timescale == "monthly":
        id_cols = ["STAID", "year", "month"]
    else:
        raise ValueError(f"Unsupported time scale: {timescale}")

    df_ids = df_expl[id_cols].copy()
    df_ids["STAID"] = df_ids["STAID"].astype(str)
    df_expl = df_expl.drop(columns=id_cols)

    files = glob.glob(
        str(Path(dir_workHPC, timescale, "VIF_Removed", f"*{clust_meth}_{region}.csv"))
    )
    if files:
        df_vif = pd.read_csv(files[0])
        vif_col = "columns_Removed" if "columns_Removed" in df_vif else "Columns_Removed"
        vif_removed = df_vif[vif_col].astype(str).str.replace(
            "DRAIN_SQKM_x",
            "DRAIN_SQKM",
        )
        df_expl = df_expl.drop(
            columns=[col for col in vif_removed if col in df_expl.columns],
            errors="ignore",
        )
    return df_ids.reset_index(drop=True), df_expl.reset_index(drop=True)


def heatmap_direction_cache_path() -> Path:
    """Return the cache path for heatmap direction signs."""
    ppet_tag = "_withPPET" if any(HEATMAP_INCLUDE_P_PET_OPTIONS) else ""
    return Path(
        dir_shap,
        (
            f"SHAP_HeatmapDirection_{REGION_PRODUCT}_{P_PET_MODEL_SCOPE}_"
            f"{HEATMAP_DIRECTION_METHOD}{ppet_tag}_normQ{OUT_TAG}.csv"
        ),
    )


def direction_lookup_to_dataframe(
    direction_lookup: Dict[tuple, Dict[str, float]],
) -> pd.DataFrame:
    """Convert nested direction lookup to a long table."""
    rows = []
    for keys, directions in direction_lookup.items():
        timescale, partition, clust_meth, region = keys
        for feature, direction in directions.items():
            rows.append(
                {
                    "time_scale": timescale,
                    "train_val": partition,
                    "clust_method": clust_meth,
                    "region": region,
                    "feature": feature,
                    "direction": direction,
                    "direction_method": HEATMAP_DIRECTION_METHOD,
                    "region_product": REGION_PRODUCT,
                    "model_scope": P_PET_MODEL_SCOPE,
                    "out_tag": OUT_TAG,
                }
            )
    return pd.DataFrame(rows)


def dataframe_to_direction_lookup(df_direction: pd.DataFrame) -> Dict[tuple, Dict[str, float]]:
    """Convert a long direction table to the nested lookup used by heatmaps."""
    direction_lookup: Dict[tuple, Dict[str, float]] = {}
    group_cols = ["time_scale", "train_val", "clust_method", "region"]
    for keys, df_group in df_direction.groupby(group_cols, dropna=False):
        direction_lookup[keys] = dict(zip(df_group["feature"], df_group["direction"]))
    return direction_lookup


def dependence_direction(
    df_pair: pd.DataFrame,
    x_col: str,
    shap_col: str,
) -> float | None:
    """Return direction from the configured dependence measure."""
    df_pair = df_pair[[x_col, shap_col]].dropna()
    if df_pair.shape[0] < 2 or df_pair[x_col].nunique() < 2:
        return None
    if HEATMAP_DIRECTION_METHOD == "spearman":
        rho = df_pair[x_col].corr(df_pair[shap_col], method="spearman")
        if pd.isna(rho) or rho == 0:
            return None
        return -1 if rho < 0 else 1

    x = df_pair[x_col].to_numpy().reshape(-1, 1)
    y = df_pair[shap_col].to_numpy()
    x = StandardScaler().fit_transform(x)
    lm = LinearRegression()
    lm.fit(x, y)
    return -1 if lm.coef_[0] < 0 else 1


def regression_direction_lookup(df_by_pred: pd.DataFrame) -> Dict[tuple, Dict[str, float]]:
    """Match feature values to SHAP rows and infer direction from dependence."""
    if HEATMAP_DIRECTION_METHOD not in ["regression_slope", "spearman"]:
        raise ValueError(
            "HEATMAP_DIRECTION_METHOD must be 'regression_slope' or 'spearman'. "
            f"Got: {HEATMAP_DIRECTION_METHOD}"
        )

    cache_path = heatmap_direction_cache_path()
    if USE_HEATMAP_DIRECTION_CACHE and cache_path.exists():
        df_direction = _read_csv(cache_path, dtype={"region": str})
        print(f"Loaded heatmap direction cache: {cache_path}")
        return dataframe_to_direction_lookup(df_direction)

    direction_lookup: Dict[tuple, Dict[str, float]] = {}
    if df_by_pred.empty:
        return direction_lookup

    df = df_by_pred.copy()
    if P_PET_MODEL_SCOPE == "XGBoost" and "model" in df:
        df = df[df["model"] == "XGBoost"].copy()

    group_cols = ["time_scale", "train_val", "clust_method", "region"]
    ppet_joined_frames = []
    for keys, df_group in df.groupby(group_cols, dropna=False):
        timescale, partition, clust_meth, region = keys
        try:
            df_ids, df_expl = prep_xy_for_heatmap_direction(
                timescale,
                partition,
                clust_meth,
                region,
            )
        except Exception as exc:
            print(
                "Heatmap direction fallback: could not load explanatory data for "
                f"{timescale}-{partition}-{clust_meth}-{region}: {exc}"
            )
            continue

        if timescale == "mean_annual":
            id_cols = ["STAID"]
        elif timescale == "annual":
            id_cols = ["STAID", "year"]
        elif timescale == "monthly":
            id_cols = ["STAID", "year", "month"]
        else:
            continue
        df_x = pd.concat([df_ids[id_cols], df_expl], axis=1)
        df_group = df_group.copy()
        df_group["STAID"] = df_group["STAID"].astype(str)
        df_x["STAID"] = df_x["STAID"].astype(str)
        df_join = df_group.merge(df_x, on=id_cols, how="left", suffixes=("_shap", "_x"))
        if any(HEATMAP_INCLUDE_P_PET_OPTIONS) and clust_meth in P_PET_CLUST_METHODS:
            ppet_joined_frames.append(df_join)

        directions: Dict[str, float] = {}
        for col in feature_columns(df_group):
            x_col = f"{col}_x" if f"{col}_x" in df_join.columns else col
            shap_col = f"{col}_shap" if f"{col}_shap" in df_join.columns else col
            if x_col not in df_join or shap_col not in df_join:
                continue
            direction = dependence_direction(df_join, x_col, shap_col)
            if direction is not None:
                directions[col] = direction
        direction_lookup[keys] = directions

    if any(HEATMAP_INCLUDE_P_PET_OPTIONS) and ppet_joined_frames:
        df_classes = load_p_pet_classes()
        df_ppet = pd.concat(ppet_joined_frames, ignore_index=True)
        df_ppet = df_ppet.merge(
            df_classes[["STAID", "train_val", "P_PET_Class"]],
            on=["STAID", "train_val"],
            how="left",
        ).dropna(subset=["P_PET_Class"])
        ppet_group_cols = ["time_scale", "train_val", "P_PET_Class"]
        for keys, df_group in df_ppet.groupby(ppet_group_cols, dropna=False):
            timescale, partition, ppet_class = keys
            directions = {}
            for col in feature_columns(df):
                x_col = f"{col}_x" if f"{col}_x" in df_group.columns else col
                shap_col = f"{col}_shap" if f"{col}_shap" in df_group.columns else col
                if x_col not in df_group or shap_col not in df_group:
                    continue
                direction = dependence_direction(df_group, x_col, shap_col)
                if direction is not None:
                    directions[col] = direction
            direction_lookup[
                (timescale, partition, "P:PET", str(ppet_class))
            ] = directions

    if WRITE_HEATMAP_DIRECTION_CACHE:
        df_direction = direction_lookup_to_dataframe(direction_lookup)
        df_direction.to_csv(cache_path, index=False)
        print(f"Wrote heatmap direction cache: {cache_path}")
    return direction_lookup


def summarize_signed_regional_shap(
    df_region: pd.DataFrame,
    df_by_pred: pd.DataFrame,
    direction_lookup: Dict[tuple, Dict[str, float]] | None = None,
) -> pd.DataFrame:
    """
    Summarize regional SHAP values for heatmaps.

    Magnitudes come from the regional mean-absolute SHAP products, which are
    already normalized by regional mean Q. Signs come from the configured
    feature-value/SHAP dependence method.
    """
    if df_region.empty:
        return pd.DataFrame()

    group_cols = ["time_scale", "train_val", "clust_method", "region"]
    df_mag = df_region.copy()
    feature_cols_in = feature_columns(df_mag)
    if not feature_cols_in:
        return pd.DataFrame()

    if direction_lookup is None:
        direction_lookup = regression_direction_lookup(df_by_pred)

    rows = []
    for _, row in df_mag.iterrows():
        key = tuple(row[col] for col in group_cols)
        row_out = {col: row[col] for col in group_cols if col in row}
        directions = direction_lookup.get(key, {})
        for col in feature_cols_in:
            value = row[col]
            if pd.isna(value):
                row_out[col] = np.nan
                continue
            if str(col) == "prcp" or re.match(r"^prcp_\d+$", str(col)):
                row_out[col] = value
                continue
            direction = directions.get(col, np.nan)
            row_out[col] = value * direction if pd.notna(direction) else np.nan
        rows.append(row_out)

    df_summary = pd.DataFrame(rows)
    alias_map, _ = feature_aliases_and_categories()
    rename_map = {
        col: alias_map.get(str(col).replace("TS_", ""), str(col).replace("TS_", ""))
        for col in feature_cols_in
    }
    meta_cols = [col for col in group_cols if col in df_summary]
    value_cols = [col for col in df_summary.columns if col not in meta_cols]
    df_values = df_summary[value_cols].T.groupby(level=0).sum(min_count=1).T
    df_values = collapse_lookback_heatmap_columns(df_values)
    df_values = df_values.rename(columns=rename_map)
    df_values = df_values.T.groupby(level=0).sum(min_count=1).T
    return pd.concat([df_summary[meta_cols].reset_index(drop=True), df_values], axis=1)


def summarize_signed_p_pet_shap(
    df_by_pred: pd.DataFrame,
    direction_lookup: Dict[tuple, Dict[str, float]],
) -> pd.DataFrame:
    """Summarize feature-level SHAP values for optional P:PET heatmap columns."""
    if df_by_pred.empty:
        return pd.DataFrame()

    df_classes = load_p_pet_classes()
    df = filter_p_pet_model_scope(df_by_pred)
    if P_PET_CLUST_METHODS:
        df = df[df["clust_method"].isin(P_PET_CLUST_METHODS)].copy()
    df = df.merge(
        df_classes[["STAID", "train_val", "P_PET_Class"]],
        on=["STAID", "train_val"],
        how="left",
    ).dropna(subset=["P_PET_Class"])
    if df.empty:
        return pd.DataFrame()

    feature_cols_in = feature_columns(df)
    group_cols = ["time_scale", "train_val", "P_PET_Class"]
    rows = []
    for keys, df_group in df.groupby(group_cols, dropna=False):
        timescale, partition, ppet_class = keys
        direction_key = (timescale, partition, "P:PET", str(ppet_class))
        directions = direction_lookup.get(direction_key, {})
        row_out = {
            "time_scale": timescale,
            "train_val": partition,
            "clust_method": "P:PET",
            "region": str(ppet_class),
        }
        for col in feature_cols_in:
            values = df_group[col]
            if not values.notna().any():
                row_out[col] = np.nan
                continue
            magnitude = values.abs().fillna(0).mean()
            if str(col) == "prcp" or re.match(r"^prcp_\d+$", str(col)):
                row_out[col] = magnitude
                continue
            direction = directions.get(col, np.nan)
            row_out[col] = magnitude * direction if pd.notna(direction) else np.nan
        rows.append(row_out)

    df_summary = pd.DataFrame(rows)
    alias_map, _ = feature_aliases_and_categories()
    rename_map = {
        col: alias_map.get(str(col).replace("TS_", ""), str(col).replace("TS_", ""))
        for col in feature_cols_in
    }
    meta_cols = ["time_scale", "train_val", "clust_method", "region"]
    value_cols = [col for col in df_summary.columns if col not in meta_cols]
    df_values = df_summary[value_cols].T.groupby(level=0).sum(min_count=1).T
    df_values = collapse_lookback_heatmap_columns(df_values)
    df_values = df_values.rename(columns=rename_map)
    df_values = df_values.T.groupby(level=0).sum(min_count=1).T
    return pd.concat([df_summary[meta_cols].reset_index(drop=True), df_values], axis=1)


def prepare_heatmap_matrix(
    df_summary: pd.DataFrame,
    partition: str,
    timescale: str,
    category_map: Dict[str, str],
) -> pd.DataFrame:
    """Return the top heatmap variables for one partition and time scale."""
    clust_order = ["None", "Class", "AggEcoregion"]
    df_plot = df_summary[
        (df_summary["train_val"] == partition)
        & (df_summary["time_scale"] == timescale)
    ].copy()
    if df_plot.empty:
        return pd.DataFrame()

    df_plot["clust_order"] = df_plot["clust_method"].map(
        {clust: i for i, clust in enumerate(clust_order)}
    )
    df_plot["clust_order"] = df_plot["clust_order"].fillna(99)
    df_plot["p_pet_order"] = df_plot["region"].map(
        {label: i for i, label in enumerate(P_PET_LABELS)}
    )
    df_plot["p_pet_order"] = df_plot["p_pet_order"].fillna(-1)
    df_plot = df_plot.sort_values(["clust_order", "p_pet_order", "region"])

    meta_cols = {
        "time_scale",
        "train_val",
        "clust_method",
        "region",
        "clust_order",
        "p_pet_order",
    }
    value_cols = [
        col
        for col in df_plot.columns
        if col not in meta_cols and pd.api.types.is_numeric_dtype(df_plot[col])
    ]
    df_values = df_plot.set_index("region")[value_cols]
    col_order = df_values.fillna(0).abs().mean().sort_values(ascending=False).index
    if timescale == "mean_annual" and "Ant Precip" in df_values.columns:
        df_values = df_values.drop(columns=["Ant Precip"])
        col_order = [col for col in col_order if col != "Ant Precip"]
    forced_candidates = ["Precip"] if timescale == "mean_annual" else ["Precip", "Ant Precip"]
    forced_cols = [col for col in forced_candidates if col in df_values.columns]
    remaining_cols = [col for col in col_order if col not in forced_cols]
    ordered_cols = forced_cols + remaining_cols
    return df_values.reindex(ordered_cols, axis=1).iloc[:, :HEATMAP_N_FEATURES].T


def prepare_anthro_heatmap_matrix(
    df_summary: pd.DataFrame,
    partition: str,
    timescale: str,
    category_map: Dict[str, str],
) -> pd.DataFrame:
    """Return the top anthropogenic variables for one partition and time scale."""
    clust_order = ["None", "Class", "AggEcoregion"]
    df_plot = df_summary[
        (df_summary["train_val"] == partition)
        & (df_summary["time_scale"] == timescale)
    ].copy()
    if df_plot.empty:
        return pd.DataFrame()

    df_plot["clust_order"] = df_plot["clust_method"].map(
        {clust: i for i, clust in enumerate(clust_order)}
    )
    df_plot["clust_order"] = df_plot["clust_order"].fillna(99)
    df_plot["p_pet_order"] = df_plot["region"].map(
        {label: i for i, label in enumerate(P_PET_LABELS)}
    )
    df_plot["p_pet_order"] = df_plot["p_pet_order"].fillna(-1)
    df_plot = df_plot.sort_values(["clust_order", "p_pet_order", "region"])

    meta_cols = {
        "time_scale",
        "train_val",
        "clust_method",
        "region",
        "clust_order",
        "p_pet_order",
    }
    value_cols = [
        col
        for col in df_plot.columns
        if col not in meta_cols
        and pd.api.types.is_numeric_dtype(df_plot[col])
        and category_map.get(col) in ["Anthro_Hydro", "Anthro_Land"]
    ]
    df_values = df_plot.set_index("region")[value_cols]
    col_order = df_values.fillna(0).abs().mean().sort_values(ascending=False).index
    return (
        df_values.reindex(col_order, axis=1)
        .iloc[:, :HEATMAP_ANTHRO_N_FEATURES]
        .T
    )


def color_heatmap_tick_labels(ax: plt.Axes, category_map: Dict[str, str]) -> None:
    """Color y-axis variable labels by broad feature category."""
    for label in ax.get_yticklabels():
        category = category_map.get(label.get_text())
        if category in feature_colors:
            label.set_color(feature_colors[category])


def plot_heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    df: pd.DataFrame,
    title_in: str | None,
    min_max: List[float],
    plot_nonprecip_cbar: bool = False,
    plot_precip_cbar: bool = False,
    cmap_in: str = "coolwarm_r",
    xlab_in: str | None = "Region",
    ylab_in: str | None = "Explanatory Variables",
    cmap_title: str = "",
    category_map: Dict[str, str] | None = None,
) -> None:
    """Draw one heatmap panel using the legacy GAGESii_Plotting.py layout."""
    if df.empty:
        ax.set_visible(False)
        return

    cmap_in = sns.color_palette(cmap_in, 100)
    center_in = (min_max[0] + min_max[1]) / 2
    vmin_in = min_max[0]
    vmax_in = min_max[1]

    ax.title.set_text(title_in)
    heatmap = sns.heatmap(
        df,
        linewidth=0.03,
        linecolor="black",
        ax=ax,
        cmap=cmap_in,
        center=center_in,
        vmin=vmin_in,
        vmax=vmax_in,
        cbar=False,
        yticklabels=True,
    )
    ax.set_xlabel(xlab_in, fontsize=HEATMAP_AXIS_TITLE_FONTSIZE)
    ax.set_ylabel(ylab_in, fontsize=HEATMAP_AXIS_TITLE_FONTSIZE)
    # rotate_heatmap_xticklabels(ax)
    ax.tick_params(
        axis="y",
        labelrotation=0,
        labelsize=HEATMAP_VARIABLE_LABEL_FONTSIZE,
    )
    if category_map is not None:
        color_heatmap_tick_labels(ax, category_map)

    ll_anchor = 0.04
    y_pos = -0.075

    if plot_precip_cbar:
        cbar_ax = fig.add_axes([(ll_anchor + 0.18), y_pos, 0.3, 0.02])
        cbar = plt.colorbar(
            heatmap.collections[0],
            cax=cbar_ax,
            orientation="horizontal",
            extend="max",
            label=cmap_title,
            ticks=[vmin_in, center_in, vmax_in],
        )
        cbar.ax.tick_params(labelsize=8)
        label_ax = fig.add_axes([0.05, y_pos, 0.05, 0.02])
        label_ax.axis("off")
        label_ax.text(
            1,
            0.5,
            "Impact on WY prediction;\nmean(SHAP)/mean(Q) [cm/cm]",
            rotation=0,
            ha="center",
            va="center",
            transform=label_ax.transAxes,
            fontsize=8,
        )
    if plot_nonprecip_cbar:
        cbar_ax = fig.add_axes([(ll_anchor + 0.53), y_pos, 0.3, 0.02])
        cbar = plt.colorbar(
            heatmap.collections[0],
            cax=cbar_ax,
            orientation="horizontal",
            extend="both",
            label=cmap_title,
            ticks=[vmin_in, center_in, vmax_in],
        )
        cbar.ax.tick_params(labelsize=8)


def plot_shap_heatmap_summary(
    df_summary: pd.DataFrame,
    include_p_pet: bool,
) -> None:
    """Plot one regional heatmap product, optionally including P:PET columns."""
    _, category_map = feature_aliases_and_categories()
    vars_short = ["Precip", "Ant Precip"]
    for partition in part_in:
        matrices = {
            timescale: prepare_heatmap_matrix(
                df_summary,
                partition,
                timescale,
                category_map,
            )
            for timescale in time_scales
        }
        if all(matrix.empty for matrix in matrices.values()):
            continue

        mannual_data_plot = matrices["mean_annual"]
        annual_data_plot = matrices["annual"]
        monthly_data_plot = matrices["monthly"]
        n_in = mannual_data_plot.shape[0] or HEATMAP_N_FEATURES
        height_short1 = 1 / n_in
        height_short2 = 2 / n_in

        with sns.plotting_context("notebook"):
            fig_width = 12 if include_p_pet else 10
            fig = plt.figure(figsize=(fig_width, 8), constrained_layout=True)
            gs = fig.add_gridspec(n_in, 3, height_ratios=np.repeat(1, n_in))

            ax1 = fig.add_subplot(gs[0 : int(height_short1 * n_in), 0])
            ax2 = fig.add_subplot(gs[0 : int(height_short2 * n_in), 1])
            ax3 = fig.add_subplot(gs[0 : int(height_short2 * n_in), 2])
            ax4 = fig.add_subplot(gs[int(height_short1 * n_in) : n_in, 0])
            ax5 = fig.add_subplot(gs[int(height_short2 * n_in) : n_in, 1])
            ax6 = fig.add_subplot(gs[int(height_short2 * n_in) : n_in, 2])

            plot_heatmap_panel(
                fig,
                ax1,
                mannual_data_plot.query("index in @vars_short"),
                "Mean Annual",
                [HEATMAP_PRECIP_VMIN, HEATMAP_PRECIP_VMAX],
                plot_nonprecip_cbar=False,
                plot_precip_cbar=True,
                cmap_in="BuPu",
                cmap_title="Precip and Ant Precip",
                xlab_in=None,
                ylab_in=None,
                category_map=category_map,
            )
            plot_heatmap_panel(
                fig,
                ax2,
                annual_data_plot.query("index in @vars_short"),
                "Annual",
                [HEATMAP_PRECIP_VMIN, HEATMAP_PRECIP_VMAX],
                plot_nonprecip_cbar=False,
                plot_precip_cbar=False,
                cmap_in="BuPu",
                xlab_in=None,
                ylab_in=None,
                category_map=category_map,
            )
            plot_heatmap_panel(
                fig,
                ax3,
                monthly_data_plot.query("index in @vars_short"),
                "Monthly",
                [HEATMAP_PRECIP_VMIN, HEATMAP_PRECIP_VMAX],
                plot_nonprecip_cbar=False,
                plot_precip_cbar=False,
                cmap_in="BuPu",
                xlab_in=None,
                ylab_in=None,
                category_map=category_map,
            )
            plot_heatmap_panel(
                fig,
                ax4,
                mannual_data_plot.query("index not in @vars_short"),
                None,
                [HEATMAP_OTHER_VMIN, HEATMAP_OTHER_VMAX],
                plot_nonprecip_cbar=True,
                plot_precip_cbar=False,
                cmap_in="coolwarm_r",
                cmap_title="Other Variables",
                category_map=category_map,
            )
            plot_heatmap_panel(
                fig,
                ax5,
                annual_data_plot.query("index not in @vars_short"),
                None,
                [HEATMAP_OTHER_VMIN, HEATMAP_OTHER_VMAX],
                plot_nonprecip_cbar=False,
                plot_precip_cbar=False,
                cmap_in="coolwarm_r",
                ylab_in=None,
                category_map=category_map,
            )
            plot_heatmap_panel(
                fig,
                ax6,
                monthly_data_plot.query("index not in @vars_short"),
                None,
                [HEATMAP_OTHER_VMIN, HEATMAP_OTHER_VMAX],
                plot_nonprecip_cbar=False,
                plot_precip_cbar=False,
                cmap_in="coolwarm_r",
                ylab_in=None,
                category_map=category_map,
            )
            ax1.set_xticklabels([])
            ax2.set_xticklabels([])
            ax3.set_xticklabels([])
            plt.subplots_adjust(hspace=0.09, wspace=1.2)

            if SAVE_FIGS:
                ppet_tag = "_withPPET" if include_p_pet else ""
                fig.savefig(
                    dir_figs
                    / (
                        f"SHAP_heatmap_{REGION_PRODUCT}_{partition}"
                        f"_AllVars_AllTimescales{ppet_tag}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()


def plot_anthro_heatmap_summary(
    df_summary: pd.DataFrame,
    include_p_pet: bool,
) -> None:
    """Plot anthropogenic-only SHAP heatmaps for each partition."""
    _, category_map = feature_aliases_and_categories()
    for partition in part_in:
        matrices = {
            timescale: prepare_anthro_heatmap_matrix(
                df_summary,
                partition,
                timescale,
                category_map,
            )
            for timescale in time_scales
        }
        if all(matrix.empty for matrix in matrices.values()):
            continue

        fig_width = 12 if include_p_pet else 10
        with sns.plotting_context("notebook"):
            fig, axes = plt.subplots(
                ncols=3,
                nrows=1,
                layout="constrained",
                figsize=(fig_width, 6.3),
            )
            for idx, timescale in enumerate(time_scales):
                ax = axes[idx]
                df_plot = matrices[timescale]
                if df_plot.empty:
                    ax.set_visible(False)
                    continue
                sns.heatmap(
                    df_plot,
                    linewidth=0.05,
                    linecolor="black",
                    ax=ax,
                    cmap=sns.color_palette("coolwarm_r", 100),
                    center=0,
                    vmin=HEATMAP_ANTHRO_VMIN,
                    vmax=HEATMAP_ANTHRO_VMAX,
                    cbar=idx == 1,
                    cbar_kws={
                        "label": (
                            "Impact on WY prediction; "
                            "mean(SHAP)/mean(Q) [cm/cm]"
                        ),
                        "extend": "both",
                        "ticks": [
                            HEATMAP_ANTHRO_VMIN,
                            0,
                            HEATMAP_ANTHRO_VMAX,
                        ],
                        "location": "bottom",
                    }
                    if idx == 1
                    else None,
                )
                ax.title.set_text(timescale.replace("_", " ").title())
                ax.set_xlabel("Region", fontsize=HEATMAP_AXIS_TITLE_FONTSIZE)
                ax.set_ylabel(
                    "Explanatory Variables" if idx == 0 else "",
                    fontsize=HEATMAP_AXIS_TITLE_FONTSIZE,
                )
                rotate_heatmap_xticklabels(ax)
                ax.tick_params(
                    axis="y",
                    labelrotation=0,
                    labelsize=HEATMAP_VARIABLE_LABEL_FONTSIZE,
                )
                color_heatmap_tick_labels(ax, category_map)

            if SAVE_FIGS:
                ppet_tag = "_withPPET" if include_p_pet else ""
                fig.savefig(
                    dir_figs
                    / (
                        f"SHAP_heatmap_AnthroOnly_{REGION_PRODUCT}_{partition}"
                        f"_AllTimescales{ppet_tag}.png"
                    ),
                    dpi=300,
                    bbox_inches="tight",
                )
            plt.show()


def plot_shap_heatmaps(df_region: pd.DataFrame, df_by_pred: pd.DataFrame) -> None:
    """Plot regional signed SHAP heatmaps from the new by-prediction products."""
    direction_lookup = regression_direction_lookup(df_by_pred)
    df_region_summary = summarize_signed_regional_shap(
        df_region,
        df_by_pred,
        direction_lookup=direction_lookup,
    )
    if df_region_summary.empty:
        print("Regional SHAP heatmaps skipped: no by-prediction rows available.")
        return

    df_p_pet_heatmap = pd.DataFrame()
    if any(HEATMAP_INCLUDE_P_PET_OPTIONS):
        df_p_pet_heatmap = summarize_signed_p_pet_shap(
            df_by_pred,
            direction_lookup,
        )

    for include_p_pet in HEATMAP_INCLUDE_P_PET_OPTIONS:
        if include_p_pet:
            if df_p_pet_heatmap.empty:
                print("P:PET heatmap columns skipped: no P:PET rows available.")
                continue
            df_summary = pd.concat(
                [df_region_summary, df_p_pet_heatmap],
                ignore_index=True,
                sort=False,
            )
        else:
            df_summary = df_region_summary
        if PLOT_ALL_VARIABLE_HEATMAPS:
            plot_shap_heatmap_summary(df_summary, include_p_pet)
        if PLOT_ANTHRO_HEATMAPS:
            plot_anthro_heatmap_summary(df_summary, include_p_pet)


# %% initialize vars to None to only load once
if "df_region" not in globals():
    df_region = None
if "df_by_prediction" not in globals():
    df_by_prediction = None
# %% run ------------------------------------------------------------------------
if RELOAD_SHAP_INPUTS or df_region is None or df_region.empty:
    df_region = read_region_summary_files()
df_region_summary = pd.DataFrame()
if not df_region.empty:
    df_region_summary = summarize_region_categories(df_region)
    if WRITE_SUMMARY_TABLES:
        out = Path(
            dir_shap,
            f"SHAP_CategorySummary_region_{REGION_PRODUCT}_normQ{OUT_TAG}.csv",
        )
        df_region_summary.to_csv(out, index=False)
        print(f"Wrote: {out}")
    plot_region_bars(df_region_summary)
else:
    print("No regional SHAP summaries loaded.")

if RELOAD_SHAP_INPUTS or df_by_prediction is None or df_by_prediction.empty:
    df_by_prediction = read_by_prediction_files()
df_p_pet_summary = pd.DataFrame()
if not df_by_prediction.empty:
    df_p_pet_summary = summarize_p_pet_categories(df_by_prediction)
    if WRITE_SUMMARY_TABLES and not df_p_pet_summary.empty:
        out = Path(
            dir_shap,
            f"SHAP_CategorySummary_PPET_{P_PET_MODEL_SCOPE}_normQ{OUT_TAG}.csv",
        )
        df_p_pet_summary.to_csv(out, index=False)
        print(f"Wrote: {out}")
    plot_p_pet_bars(df_p_pet_summary)
else:
    print("No by-prediction SHAP rows loaded.")

plot_shap_heatmaps(df_region, df_by_prediction)
plot_region_and_p_pet_bars(df_region_summary, df_p_pet_summary)
plot_region_and_p_pet_bars(
    df_region_summary,
    df_p_pet_summary,
    exclude_climate=True,
)
