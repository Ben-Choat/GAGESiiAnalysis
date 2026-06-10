"""
BChoat 2026/05

Build the SHAP products used by the manuscript plotting scripts.

Products written by this script:
1. SHAP_ByPrediction_{timescale}_allModels_normQ_202605.csv
   - one row per prediction, with STAID/time metadata and SHAP values.
2. MeanShap_XGBoostOnly_{partition}_{timescale}_normQ_202605.csv
   - regional mean absolute SHAP values for XGBoost only.
3. MeanShap_BestRegionalModel_medianNSE_{partition}_{timescale}_normQ_202605.csv
   - regional mean absolute SHAP values for the model with the best
     median NSE in that region.

The regional products are intentionally derived from the by-prediction product
so the plots all trace back to the same SHAP calculations.
"""

# %%
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from Load_Data import load_data_fun
from sklearn.linear_model import LinearRegression


# %% define dirs, vars, and such -------------------------------------------------
dir_results = (
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/Results"
)
dir_workHPC = (
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/"
    "HPC_Files/GAGES_Work/data_out"
)
dir_shapout = (
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/SHAP_OUT"
)

OUT_TAG = "_202605"

time_scales = ["mean_annual", "annual", "monthly"]
part_in = ["train", "valnit"]
clust_meth_in = ["None", "Class", "AggEcoregion"]

# Candidate models to calculate for the by-prediction product. The XGBoost-only
# regional product is filtered from these rows; the best-regional product uses
# all of them.
models_in = ["regr_precip", "strd_mlr", "XGBoost"]

WRITE_BY_PREDICTION = True
WRITE_XGBOOST_REGIONAL = True
WRITE_BEST_MEDIAN_NSE_REGIONAL = True
WRITE_PARQUET_BY_PREDICTION = True
WRITE_CSV_BY_PREDICTION = True
OVERWRITE_OUTPUTS = True

# Existing files are named normQ. This uses the same convention as CalcAll.py:
# feature SHAP values are divided by mean water yield within the loaded group.
NORMALIZE_SHAP_BY_MEAN_WY = True

names_drop = ["STAID", "year", "month", "day", "date"]

regional_group_cols = [
    "time_scale",
    "train_val",
    "clust_method",
    "region",
]

by_prediction_meta_cols = [
    "STAID",
    "year",
    "month",
    "date",
    "WY_cm",
    "shap_norm",
    "NSE",
    "KGE",
    "r",
    "alpha",
    "beta",
    "residuals",
    "|residuals|",
    "model",
    "time_scale",
    "train_val",
    "clust_method",
    "region",
]


# %% helper functions ------------------------------------------------------------
def read_results(timescale: str) -> pd.DataFrame:
    """Read catchment-level performance results for a single time scale."""
    if timescale == "mean_annual":
        df_results = pd.read_csv(
            f"{dir_results}/PerfMetrics_MeanAnnual.csv",
            dtype={"STAID": "string", "region": "string"},
        )
        df_results = df_results.copy()
        df_results["|residuals|"] = df_results["residuals"].abs()
    else:
        df_results = pd.read_csv(
            f"{dir_results}/NSEComponents_KGE.csv",
            dtype={"STAID": "string", "region": "string"},
        )
        df_results = df_results[df_results["time_scale"] == timescale].copy()

    df_results["STAID"] = df_results["STAID"].astype(str)
    df_results["region"] = df_results["region"].astype(str)
    df_results = df_results[
        df_results["train_val"].isin(part_in)
        & df_results["clust_method"].isin(clust_meth_in)
        & df_results["model"].isin(models_in)
    ].copy()
    return df_results


def prep_xy(
    df_expl: pd.DataFrame,
    df_WY: pd.DataFrame,
    timescale: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Split loaded data into explanatory vars, response, and ID/time columns."""
    df_expl = df_expl.copy()
    df_WY = df_WY.copy()

    if "SQKM_x" in df_expl.columns.values:
        df_expl.columns = df_expl.columns.str.replace("SQKM_x", "SQKM")

    if timescale == "mean_annual":
        df_ids = df_expl[["STAID"]].copy()
        df_expl = df_expl.drop("STAID", axis=1)
        y = df_WY["Ann_WY_cm"].copy()
    elif timescale == "annual":
        df_ids = df_expl[["STAID", "year"]].copy()
        df_expl = df_expl.drop(["STAID", "year"], axis=1)
        y = df_WY["Ann_WY_cm"].copy()
    elif timescale == "monthly":
        df_ids = df_expl[["STAID", "year", "month"]].copy()
        df_expl = df_expl.drop(["STAID", "year", "month"], axis=1)
        y = df_WY["Mnth_WY_cm"].copy()
    else:
        raise ValueError(f"Unsupported time scale: {timescale}")

    df_ids["STAID"] = df_ids["STAID"].astype(str)
    return df_expl, y, df_ids


def read_vif_removed(timescale: str, clust_meth: str, region: str) -> List[str]:
    """Read variables removed for VIF and return columns present in the data."""
    files = glob.glob(
        f"{dir_workHPC}/{timescale}/VIF_Removed/*{clust_meth}_{region}.csv"
    )
    if not files:
        print(f"No VIF_Removed file found for {timescale}-{clust_meth}-{region}")
        return []

    df_vif = pd.read_csv(files[0])
    if "columns_Removed" in df_vif.columns:
        vif_removed = df_vif["columns_Removed"]
    else:
        vif_removed = df_vif["Columns_Removed"]

    vif_removed = vif_removed.astype(str)
    vif_removed = vif_removed.str.replace("DRAIN_SQKM_x", "DRAIN_SQKM")
    return vif_removed.tolist()


def read_mlr_features(timescale: str, clust_meth: str, region: str) -> List[str]:
    """Read variables retained in the final standardized MLR model."""
    files = glob.glob(
        f"{dir_workHPC}/{timescale}/VIF_dfs/"
        f"{clust_meth}_{region}_strd_mlr*.csv"
    )
    if not files:
        raise FileNotFoundError(
            "No VIF_dfs file found for "
            f"{timescale}-{clust_meth}-{region}-strd_mlr"
        )
    vars_keep = pd.read_csv(files[0])["feature"].astype(str)
    vars_keep = vars_keep.str.replace("DRAIN_SQKM_x", "DRAIN_SQKM")
    return vars_keep.tolist()


def xgboost_model_path(timescale: str, clust_meth: str, region: str) -> Path:
    """Return the trained XGBoost model path used by the HPC scripts."""
    temp_time = timescale.replace("_", "")
    return Path(
        dir_workHPC,
        timescale,
        "Models",
        f"XGBoost_{temp_time}_{clust_meth}_{region}_model.json",
    )


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate columns created by historical feature-name fixes.

    Numeric duplicates are summed because they represent duplicate SHAP
    contributions to the same cleaned feature name. Non-numeric duplicates keep
    the first non-missing value by row.
    """
    if df.columns.is_unique:
        return df

    out_cols = []
    out_names = []
    for col in pd.Index(df.columns).unique():
        df_sub = df.loc[:, df.columns == col]
        if df_sub.shape[1] == 1:
            out_cols.append(df_sub.iloc[:, 0])
        elif all(pd.api.types.is_numeric_dtype(df_sub[x]) for x in df_sub.columns):
            out_cols.append(df_sub.sum(axis=1))
        else:
            out_cols.append(df_sub.bfill(axis=1).iloc[:, 0])
        out_names.append(col)

    df_out = pd.concat(out_cols, axis=1)
    df_out.columns = out_names
    return df_out


def shap_for_combo(
    timescale: str,
    partition: str,
    clust_meth: str,
    region: str,
    model_name: str,
    staid_keep: pd.Series,
) -> pd.DataFrame:
    """Calculate SHAP values for a model/grouping/partition combination."""
    standardize = model_name != "XGBoost"
    df_expl, df_WY, _ = load_data_fun(
        dir_work=dir_results,
        time_scale=timescale,
        train_val=partition,
        clust_meth=clust_meth,
        region=region,
        standardize=np.array(standardize),
    )
    df_expl, y, df_ids = prep_xy(df_expl, df_WY, timescale)

    vif_removed = read_vif_removed(timescale, clust_meth, region)
    vif_removed = [col for col in vif_removed if col in df_expl.columns]
    if vif_removed:
        df_expl = df_expl.drop(vif_removed, axis=1)

    staid_keep = staid_keep.astype(str)
    row_mask = df_ids["STAID"].isin(staid_keep)
    X_group = df_expl.reset_index(drop=True)
    y_group = y.reset_index(drop=True)
    ids_group = df_ids.reset_index(drop=True)
    X_pred = X_group.loc[row_mask].reset_index(drop=True)
    y_pred = y_group.loc[row_mask].reset_index(drop=True)
    ids_pred = ids_group.loc[row_mask].reset_index(drop=True)

    if X_pred.empty:
        return pd.DataFrame()

    if model_name == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(str(xgboost_model_path(timescale, clust_meth, region)))
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_pred, check_additivity=False)
        df_shap = pd.DataFrame(shap_values.values, columns=X_pred.columns)
    elif model_name == "regr_precip":
        model = LinearRegression()
        X_fit = X_group[["prcp"]]
        X_out = X_pred[["prcp"]]
        model.fit(X_fit, y_group)
        explainer = shap.LinearExplainer(model, X_fit)
        shap_values = explainer(X_out)
        df_shap = pd.DataFrame(shap_values.values, columns=X_out.columns)
    elif model_name in ["strd_mlr", "strd_lasso"]:
        vars_keep = read_mlr_features(timescale, clust_meth, region)
        vars_keep = [col for col in vars_keep if col in X_group.columns]
        if not vars_keep:
            raise ValueError(
                f"No MLR variables found in data for {timescale}-{clust_meth}-{region}"
            )
        model = LinearRegression()
        X_fit = X_group[vars_keep]
        X_out = X_pred[vars_keep]
        model.fit(X_fit, y_group)
        explainer = shap.LinearExplainer(model, X_fit)
        shap_values = explainer(X_out)
        df_shap = pd.DataFrame(shap_values.values, columns=X_out.columns)
    else:
        raise ValueError(f"Unsupported model for SHAP product: {model_name}")

    df_shap = collapse_duplicate_columns(df_shap)

    shap_norm = "none"
    if NORMALIZE_SHAP_BY_MEAN_WY:
        mean_wy = y_group.mean()
        if pd.notna(mean_wy) and mean_wy != 0:
            df_shap = df_shap / mean_wy
            shap_norm = "mean_WY_cm"
        else:
            shap_norm = "mean_WY_cm_not_applied"

    df_shap["WY_cm"] = y_pred
    df_shap["shap_norm"] = shap_norm
    df_out = pd.concat([ids_pred, df_shap.reset_index(drop=True)], axis=1)
    return df_out


def output_path_by_prediction(timescale: str) -> Path:
    return Path(
        dir_shapout,
        f"SHAP_ByPrediction_{timescale}_allModels_normQ{OUT_TAG}.csv",
    )


def output_path_by_prediction_parquet(timescale: str) -> Path:
    return Path(
        dir_shapout,
        f"SHAP_ByPrediction_{timescale}_allModels_normQ{OUT_TAG}.parquet",
    )


def output_path_xgb(partition: str, timescale: str) -> Path:
    return Path(
        dir_shapout,
        f"MeanShap_XGBoostOnly_{partition}_{timescale}_normQ{OUT_TAG}.csv",
    )


def output_path_best(partition: str, timescale: str) -> Path:
    return Path(
        dir_shapout,
        f"MeanShap_BestRegionalModel_medianNSE_{partition}_{timescale}_normQ"
        f"{OUT_TAG}.csv",
    )


def feature_cols_from_by_prediction(df: pd.DataFrame) -> List[str]:
    """Return likely SHAP feature columns from a by-prediction table."""
    meta = set(by_prediction_meta_cols + regional_group_cols)
    cols = []
    for col in df.columns:
        if col in meta:
            continue
        if col in names_drop:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def append_or_write(df: pd.DataFrame, path: Path) -> None:
    """Write a CSV using the notebook-script append/overwrite style."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if OVERWRITE_OUTPUTS and path.exists():
        os.remove(path)
    df.to_csv(path, index=False)
    print(f"Wrote: {path}")


def write_by_prediction(df: pd.DataFrame, timescale: str) -> None:
    """Write by-prediction SHAP outputs, preferring parquet for later reads."""
    if WRITE_CSV_BY_PREDICTION:
        append_or_write(df, output_path_by_prediction(timescale))

    if WRITE_PARQUET_BY_PREDICTION:
        path = output_path_by_prediction_parquet(timescale)
        path.parent.mkdir(parents=True, exist_ok=True)
        if OVERWRITE_OUTPUTS and path.exists():
            os.remove(path)
        try:
            df.to_parquet(path, index=False)
            print(f"Wrote: {path}")
        except ImportError as exc:
            print(
                "Could not write parquet by-prediction file. Install pyarrow "
                f"or fastparquet to enable parquet output. Original error: {exc}"
            )


def build_by_prediction_for_timescale(timescale: str) -> pd.DataFrame:
    """Calculate and write the per-prediction SHAP table for one time scale."""
    df_results = read_results(timescale)
    if df_results.empty:
        print(f"No results found for {timescale}")
        return pd.DataFrame()

    df_results = df_results.sort_values(
        ["train_val", "clust_method", "region", "model", "STAID"]
    ).reset_index(drop=True)
    shap_frames = []

    group_cols = ["train_val", "clust_method", "region", "model"]
    for (partition, clust_meth, region, model_name), df_group in df_results.groupby(
        group_cols, dropna=False
    ):
        print(f"\nProcessing {timescale}-{partition}-{clust_meth}-{region}-{model_name}")
        df_shap = shap_for_combo(
            timescale,
            partition,
            clust_meth,
            str(region),
            model_name,
            df_group["STAID"],
        )
        if df_shap.empty:
            print("No SHAP rows returned for this combination.")
            continue

        keep_meta = [col for col in df_group.columns if col not in df_shap.columns]
        df_perf = df_group[["STAID"] + keep_meta].copy()
        df_out = df_shap.merge(df_perf, on="STAID", how="left")
        df_out = collapse_duplicate_columns(df_out)
        shap_frames.append(df_out)

    if not shap_frames:
        return pd.DataFrame()

    df_by_pred = pd.concat(shap_frames, ignore_index=True)
    if WRITE_BY_PREDICTION:
        write_by_prediction(df_by_pred, timescale)
    return df_by_pred


def summarize_region_shap(df_by_pred: pd.DataFrame) -> pd.DataFrame:
    """Summarize by-prediction SHAP values to mean absolute SHAP by region."""
    feature_cols = feature_cols_from_by_prediction(df_by_pred)
    if not feature_cols:
        return pd.DataFrame()

    rows = []
    for keys, df_group in df_by_pred.groupby(regional_group_cols, dropna=False):
        out = dict(zip(regional_group_cols, keys))
        out["n_predictions"] = df_group.shape[0]
        for col in feature_cols:
            out[col] = df_group[col].abs().mean()
        rows.append(out)
    return pd.DataFrame(rows)


def write_xgboost_regional(df_by_pred: pd.DataFrame, timescale: str) -> None:
    """Write XGBoost-only regional SHAP summaries."""
    if not WRITE_XGBOOST_REGIONAL:
        return
    df_xgb = df_by_pred[df_by_pred["model"] == "XGBoost"].copy()
    if df_xgb.empty:
        print(f"No XGBoost rows available for {timescale}")
        return

    df_summary = summarize_region_shap(df_xgb)
    df_summary["model"] = "XGBoost"
    df_summary["aggregation"] = "mean_abs_SHAP"
    for partition, df_part in df_summary.groupby("train_val"):
        append_or_write(df_part, output_path_xgb(partition, timescale))


def write_best_median_nse_regional(df_by_pred: pd.DataFrame, timescale: str) -> None:
    """Write best-regional-model SHAP summaries using median NSE."""
    if not WRITE_BEST_MEDIAN_NSE_REGIONAL:
        return
    if "NSE" not in df_by_pred.columns:
        print(
            f"Skipping best-regional-model summary for {timescale}: "
            "NSE column is not present."
        )
        return

    score_cols = regional_group_cols + ["model"]
    df_scores = (
        df_by_pred.dropna(subset=["NSE"])
        .groupby(score_cols, as_index=False)["NSE"]
        .median()
        .rename(columns={"NSE": "median_NSE"})
    )
    if df_scores.empty:
        print(f"Skipping best-regional-model summary for {timescale}: no NSE values.")
        return

    idx_best = df_scores.groupby(regional_group_cols)["median_NSE"].idxmax()
    df_best = df_scores.loc[idx_best].reset_index(drop=True)
    print("\nBest regional models based on median NSE:")
    print(df_best[regional_group_cols + ["model", "median_NSE"]])

    df_keep = df_by_pred.merge(
        df_best[regional_group_cols + ["model", "median_NSE"]],
        on=regional_group_cols + ["model"],
        how="inner",
    )
    df_summary = summarize_region_shap(df_keep)
    df_summary = df_summary.merge(
        df_best[regional_group_cols + ["model", "median_NSE"]],
        on=regional_group_cols,
        how="left",
    )
    df_summary = df_summary.rename(columns={"model": "best_model"})
    df_summary["metric_in"] = "NSE"
    df_summary["metric_summary"] = "median"
    df_summary["aggregation"] = "mean_abs_SHAP"

    for partition, df_part in df_summary.groupby("train_val"):
        append_or_write(df_part, output_path_best(partition, timescale))


# %% main workflow ---------------------------------------------------------------
for timescale in time_scales:
    print(f"\n\n========== {timescale} ==========")
    df_by_prediction = build_by_prediction_for_timescale(timescale)
    if df_by_prediction.empty:
        print(f"No by-prediction SHAP rows created for {timescale}.")
        continue

    write_xgboost_regional(df_by_prediction, timescale)
    write_best_median_nse_regional(df_by_prediction, timescale)

print("\n\n---------------COMPLETE----------------\n\n")
