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
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch


# %% define dirs, vars, and such -------------------------------------------------
dir_shap = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/SHAP_OUT"
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

metadata_cols = {
    "STAID",
    "year",
    "month",
    "date",
    "WY_cm",
    "shap_norm",
    "n_predictions",
    "NSE",
    "KGE",
    "r",
    "alpha",
    "beta",
    "residuals",
    "|residuals|",
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
            ax.tick_params(
                axis="x",
                rotation=90,
                labelsize=TICK_LABEL_FONTSIZE,
            )
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
        ax.tick_params(axis="x", rotation=0, labelsize=TICK_LABEL_FONTSIZE_P_PET)
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

            df_bars = df_plot.set_index("plot_label")[cat_cols]
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
            ax.tick_params(
                axis="x",
                rotation=90,
                labelsize=TICK_LABEL_FONTSIZE_REGION_P_PET,
            )
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
            fig.savefig(
                dir_figs
                / f"SHAP_region_PPET_{REGION_PRODUCT}_{partition}_vertical.png",
                dpi=300,
            )
        plt.show()


# %% inialize vars to None to only load once
if "df_region" not in globals():
    df_region = None
if "df_by_prediction" not in globals():
    df_by_prediction = None
# %% run ------------------------------------------------------------------------
if df_region is None:
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

if df_by_prediction is None:
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

plot_region_and_p_pet_bars(df_region_summary, df_p_pet_summary)
