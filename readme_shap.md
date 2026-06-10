# SHAP Workflow Notes

This document summarizes the SHAP-related scripts, the files they create/read,
and the main consistency concerns. The key point is that the current workflow has
multiple SHAP CSV families that are related but not interchangeable.

## Current Canonical Workflow

Use these two scripts for the new workflow:

```text
Python/Scripts/Build_SHAP_Products.py
Python/Scripts/Plot_SHAP_Products.py
```

`Build_SHAP_Products.py` is the producer. It first writes one canonical
by-prediction SHAP table, then derives regional summaries from that same table.
That makes the region plots and P:PET plots trace back to the same SHAP values.

Outputs:

```text
SHAP_ByPrediction_{timescale}_allModels_normQ_202605.csv
MeanShap_XGBoostOnly_{partition}_{timescale}_normQ_202605.csv
MeanShap_BestRegionalModel_medianNSE_{partition}_{timescale}_normQ_202605.csv
```

`Plot_SHAP_Products.py` is the consumer. It reads the new files above, collapses
feature columns into broad categories from `FeatureCategories.csv`, and makes:

```text
SHAP_CategorySummary_region_{REGION_PRODUCT}_normQ_202605.csv
SHAP_CategorySummary_PPET_{P_PET_MODEL_SCOPE}_normQ_202605.csv
```

The plotting script can use `REGION_PRODUCT = "XGBoostOnly"` or
`REGION_PRODUCT = "BestRegionalModel_medianNSE"` for region stacked bars. For
P:PET stacked bars, it reads the by-prediction file because that is the file
family with `STAID`.

## Main SHAP Output Families

### Partitioned regional SHAP files

Produced by:

- `Python/Scripts/GAGESii_SHAP_CalcAll.py`

Current output pattern:

```text
MeanShap_{part_in}_{timescale}_normQ_202605.csv
PCA95_{part_in}_{timescale}_202605.csv
```

Examples:

```text
MeanShap_valnit_mean_annual_normQ_202605.csv
MeanShap_valnit_annual_normQ_202605.csv
MeanShap_valnit_monthly_normQ_202605.csv
```

These files are partition-specific (`train` or `valnit`) and region/group level.
Each SHAP row represents a `clust_meth` / `region` / `time_scale` group, not an
individual catchment. They contain `clust_meth` and `region`, but not `STAID`.

Current model-selection behavior:

- For each `time_scale`, `clust_meth`, and `region`, the script chooses the best
  model type across catchments in that group.
- For `mean_annual`, the score metric is always `|residuals|`; lower is better.
- For `annual` and `monthly`, the score metric is `metric_in`, currently `KGE`;
  higher is better.
- The metric is summarized according to `metric_summary`, currently `median`,
  with options `mean`, `median`, or `qmean`.
- The output records `best_model`, `best_score`, and `metric_summary`.

Current SHAP calculation behavior:

- `regr_precip`: uses the fitted precipitation coefficient.
- `strd_mlr` / `strd_lasso`: uses `shap.LinearExplainer`, then stores mean
  absolute SHAP values with sign from model coefficients.
- `XGBoost`: uses `shap.TreeExplainer`, then stores mean absolute SHAP values
  with sign assigned by the slope of standardized feature values against SHAP
  values. Values are normalized by mean water yield.

Intended use:

- Region-level heatmaps and stacked bars by `clust_meth` / `region`.
- This family cannot be split by basin-level `P:PET` class because it has no
  `STAID`.

### STAID-preserving individual SHAP files

Produced by:

- `Python/Scripts/GAGESii_SHAP_CalcAll_SaveIndividualPreds.py`

Current output pattern:

```text
MeanShap_BestGrouping_All_{timescale}_normQ_202605.csv
```

Examples:

```text
MeanShap_BestGrouping_All_mean_annual_normQ_202605.csv
MeanShap_BestGrouping_All_annual_normQ_202605.csv
MeanShap_BestGrouping_All_monthly_normQ_202605.csv
```

These files preserve catchment metadata and now repeat metadata for each
individual SHAP row. Annual/monthly/date identifiers are retained when present.

Current model-selection behavior:

- `models_in = ['XGBoost']`, so this script currently selects the best XGBoost
  grouping, not the best model across all model types.
- `part_in = ['train', 'valnit']`, so best rows are selected from both
  partitions.
- For `mean_annual`, lower `|residuals|` is better.
- For `annual` and `monthly`, higher `metric_in` is better; currently
  `metric_in = 'NSE'`.
- Selection is per `STAID`, not per region.

Current SHAP calculation behavior:

- In practice this script is currently XGBoost-only because of `models_in`.
- It computes individual-row XGBoost SHAP values.
- It adds `WY_cm` to the output.
- Important concern: despite the filename containing `normQ`, the current
  XGBoost branch in this script does not divide SHAP values by water yield. This
  should be reconciled before using these files as normalized SHAP values.

Intended use:

- Catchment-level or time-step-level analysis where `STAID` is required.
- This is the file family needed for basin-level `P:PET` grouping.

### Legacy/best-model non-partitioned SHAP files

Produced by:

- `Python/Scripts/GAGESii_SHAP_CalcAll_bestModel.py`

Current output pattern:

```text
MeanShap_BestGrouping_All_{timescale}_normQ.csv
```

Current concerns:

- It is not tagged with `_202605`.
- It currently loops only `time_scale[1:2]`, which means annual only.
- `models_in = ['XGBoost']`, so it is not selecting among all model types.
- It writes the same base filename family as `SaveIndividualPreds.py`, but with
  different behavior.
- It should be treated as legacy or experimental unless explicitly brought into
  alignment with the chosen canonical workflow.

## SHAP Consumers

### `Python/Scripts/GAGESii_Plotting.py`

Reads:

```text
MeanShap_{part_in}_mean_annual_normQ.csv
MeanShap_{part_in}_annual_normQ.csv
MeanShap_{part_in}_monthly_normQ.csv
```

This script expects partitioned regional SHAP files. Its stacked barplots are
region-level plots based on `clust_meth` / `region`.

Current concern:

- It reads untagged files, while the modified producer writes `_202605` files.
  To use the modified outputs, this script must be updated to read the tagged
  filenames.

### `Python/Scripts/GAGESii_Plotting2_best.py`

Reads:

```text
MeanShap_BestGrouping_All_mean_annual_normQ.csv
MeanShap_BestGrouping_All_annual_normQ.csv
MeanShap_BestGrouping_All_monthly_normQ.csv
```

This script is aimed at best-grouping/catchment-preserving files and then
aggregates them by region for plotting.

Current concerns:

- It reads untagged files.
- It may be using files from `GAGESii_SHAP_CalcAll_bestModel.py` or older
  outputs rather than the tagged `SaveIndividualPreds.py` outputs.
- Its results should not be compared directly to `GAGESii_Plotting.py` unless
  both are pointed at outputs produced by a consistent SHAP workflow.

### `Python/Scripts/adding_P_PET_data_and_plots.py`

Current behavior:

```python
USE_PARTITIONED_SHAP = True
```

When `USE_PARTITIONED_SHAP = True`, it reads:

```text
MeanShap_train_{timescale}_normQ_202605.csv
MeanShap_valnit_{timescale}_normQ_202605.csv
```

These partitioned files do not contain `STAID`, so the script correctly skips
SHAP summaries with:

```text
Skipping SHAP summary ... no STAID-like column present.
```

When `USE_PARTITIONED_SHAP = False`, it reads:

```text
MeanShap_BestGrouping_All_{timescale}_normQ.csv
```

Current concern:

- The non-partitioned pattern is currently untagged in the script:

  ```python
  SHAP_FILE_PATTERN_ALL = "MeanShap_BestGrouping_All_{suffix}_normQ.csv"
  ```

  If using modified `SaveIndividualPreds.py` output, this should be changed to:

  ```python
  SHAP_FILE_PATTERN_ALL = "MeanShap_BestGrouping_All_{suffix}_normQ{OUT_TAG}.csv"
  ```

Intended use:

- P:PET-based SHAP summaries require `STAID`, so this script should use the
  STAID-preserving `BestGrouping_All` files, not the partitioned regional files.

### `Python/Scripts/Calculate_Process_P_PET.py`

This appears to be an exploratory precursor for calculating P:PET and trying
SHAP summaries. It reads both partitioned SHAP files and performance data, then
experiments with P:PET categories and SHAP aggregation.

Current concern:

- It should not be treated as the canonical SHAP/P:PET workflow unless it is
  updated to use the same tagged/canonical SHAP outputs as the other scripts.

## Consistency Check

The current scripts do not all use the same SHAP values calculated the same way.

Main inconsistencies:

1. Regional vs STAID-level grain
   - `GAGESii_SHAP_CalcAll.py` outputs one row per region/group.
   - `GAGESii_SHAP_CalcAll_SaveIndividualPreds.py` outputs individual
     catchment/time-step rows.

2. Different model-selection logic
   - `GAGESii_SHAP_CalcAll.py` selects the best model type per region/group.
   - `GAGESii_SHAP_CalcAll_SaveIndividualPreds.py` currently selects the best
     XGBoost grouping per catchment.
   - `GAGESii_SHAP_CalcAll_bestModel.py` is also XGBoost-only in its current
     state and appears stale.

3. Different metric choices
   - `GAGESii_SHAP_CalcAll.py`: annual/monthly use `metric_in = 'KGE'`, with
     `metric_summary = 'median'`.
   - `GAGESii_SHAP_CalcAll_SaveIndividualPreds.py`: annual/monthly use
     `metric_in = 'NSE'`.

4. Different SHAP aggregation
   - `GAGESii_SHAP_CalcAll.py` stores regional mean absolute SHAP values with a
     direction sign and normalizes by mean water yield.
   - `GAGESii_SHAP_CalcAll_SaveIndividualPreds.py` stores individual XGBoost
     SHAP rows and currently does not normalize by water yield despite the
     filename.

5. Tagged vs untagged filenames
   - Modified producers write `_202605` outputs.
   - Several consumers still read untagged outputs.

## Target SHAP Products

The desired workflow should produce four canonical products. These products
should be treated as separate outputs with explicit filenames, because they have
different grains and answer different questions.

### 1. SHAP values for each prediction

Purpose:

- Preserve the SHAP contribution for every prediction row.
- XGBoost should be supported first.
- Other model types should be allowed by design, rather than blocked by hard-coded
  `models_in = ['XGBoost']`.

Recommended producer:

- `Python/Scripts/Build_SHAP_Products.py`

Current output pattern:

```text
SHAP_ByPrediction_{timescale}_allModels_normQ_202605.csv
```

Minimum required columns:

```text
STAID
train_val
time_scale
clust_method
region
model
year/month/date where applicable
observed WY column or WY_cm
SHAP feature columns
```

Important implementation requirements:

- Metadata must be repeated for every prediction row.
- Annual/monthly rows should keep `year` and/or `month`.
- The file should make clear whether SHAP values are raw or normalized by water
  yield. Do not use `normQ` in the filename unless values are actually
  normalized by discharge/water yield.
- If non-XGBoost models are included, record the explainer/approach used so
  values are interpretable across model types.

### 2. Regional SHAP summaries for all-XGBoost models

Purpose:

- Produce region-level SHAP summaries using XGBoost only.
- This supports apples-to-apples comparison of feature groups across regions
  without mixing model families.

Recommended producer:

- `Python/Scripts/Build_SHAP_Products.py`

Current output pattern:

```text
MeanShap_XGBoostOnly_{part_in}_{timescale}_normQ_202605.csv
```

Minimum required columns:

```text
clust_meth
region
train_val
time_scale
model = XGBoost
metric_summary = not_applicable or fixed
SHAP feature columns
```

Important implementation requirements:

- Do not perform best-model selection across model families.
- Keep the XGBoost SHAP calculation and normalization identical to any other
  XGBoost summary product.
- Record enough metadata to make it clear these are regional summaries, not
  STAID-level values.

### 3. Regional SHAP summaries for the best model by median NSE

Purpose:

- Produce region-level SHAP summaries using the best model type in each region.
- The best regional model should be identified by `median(NSE)`.

Recommended producer:

- `Python/Scripts/Build_SHAP_Products.py`

Required settings:

```python
metric_in = 'NSE'
metric_summary = 'median'
```

Current output pattern:

```text
MeanShap_BestRegionalModel_medianNSE_{part_in}_{timescale}_normQ_202605.csv
```

Minimum required columns:

```text
clust_meth
region
train_val
time_scale
best_model
best_score
metric_in = NSE
metric_summary = median
SHAP feature columns
```

Important implementation requirements:

- Best model selection must be done within each `clust_meth` / `region` /
  `time_scale` / `train_val` subset.
- For `mean_annual`, decide explicitly whether to keep the current behavior
  (`|residuals|`) or force the same `median(NSE)` rule. If the desired product is
  literally "best by median(NSE)", then mean annual must have an NSE-like score
  available or be excluded from this product.
- If different model types are mixed, document the SHAP calculation differences.
  Linear-model SHAP and XGBoost SHAP should not be treated as identical without
  caveats.

### 4. SHAP stacked-bar plotting scripts

Desired plots:

1. Ecoregion/region stacked SHAP barplots.
2. P:PET-partition stacked SHAP barplots.

Recommended plotting inputs:

- `Python/Scripts/Plot_SHAP_Products.py`

- Ecoregion barplots should use regional summary files:

  ```text
  MeanShap_XGBoostOnly_...
  MeanShap_BestRegionalModel_medianNSE_...
  ```

- P:PET-partition barplots should use STAID-preserving by-prediction files:

  ```text
  SHAP_ByPrediction_...
  ```

Important implementation requirements:

- Do not use regional summary files for basin-level P:PET partitions because
  regional files do not contain `STAID`.
- Do not use by-prediction files for regional bars without first defining how to
  aggregate prediction rows to catchments and then regions.
- The plotting scripts should expose a single `SHAP_INPUT_FILE` or
  `SHAP_INPUT_PATTERN` setting so the source file family is obvious.

## Recommended Cleanup

Choose one canonical producer for each target product and make the filenames
distinct enough that scripts cannot accidentally read the wrong SHAP family.

Immediate changes recommended:

1. Rename output files so the grain and selection rule are explicit:
   - `SHAP_ByPrediction_*`
   - `MeanShap_XGBoostOnly_*`
   - `MeanShap_BestRegionalModel_medianNSE_*`
2. Update consumers to read only the intended canonical file family.
3. Decide whether by-prediction SHAP values should be raw or normalized. Make the
   filename match that decision.
4. Align `metric_in` and `metric_summary` with the desired product:
   - XGBoost-only regional summaries: no cross-model selection.
   - Best-regional-model summaries: `metric_in = 'NSE'` and
     `metric_summary = 'median'`.
5. Keep `GAGESii_SHAP_CalcAll_bestModel.py` out of the canonical workflow unless
   it is refactored and renamed, because its current behavior overlaps with
   `SaveIndividualPreds.py` but is not equivalent.

Avoid comparing outputs across regional and STAID-level file families unless the
model-selection metric, model candidates, partition handling, SHAP aggregation,
and normalization are intentionally aligned.
