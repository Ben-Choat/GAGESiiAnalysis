"""
BChoat 2026/05

Convert existing SHAP CSV files to parquet so plotting can load them faster.
This does not recalculate SHAP values.
"""

# %%
from pathlib import Path

import pandas as pd


# %% define dirs, vars, and such -------------------------------------------------
dir_shap = Path(
    "C:/Users/bench/OneDrive/ML_DriversOfWY/GAGESii_ANNstuff/Data_Out/SHAP_OUT"
)

OUT_TAG = "_202605"
time_scales = ["mean_annual", "annual", "monthly"]

# Convert the large by-prediction files by default.
CSV_PATTERNS = [
    f"SHAP_ByPrediction_{{timescale}}_allModels_normQ{OUT_TAG}.csv",
]

OVERWRITE_PARQUET = True


# %% convert --------------------------------------------------------------------
for timescale in time_scales:
    for pattern in CSV_PATTERNS:
        csv_path = Path(dir_shap, pattern.format(timescale=timescale))
        parquet_path = csv_path.with_suffix(".parquet")

        if not csv_path.exists():
            print(f"Missing CSV, skipping: {csv_path}")
            continue

        if parquet_path.exists() and not OVERWRITE_PARQUET:
            print(f"Parquet already exists, skipping: {parquet_path}")
            continue

        print(f"Reading: {csv_path}")
        df = pd.read_csv(
            csv_path,
            dtype={
                "STAID": str,
                "region": str,
            },
        )

        print(f"Writing: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
        print(f"Finished: {parquet_path} {df.shape}")

print("\n\n---------------COMPLETE----------------\n\n")
