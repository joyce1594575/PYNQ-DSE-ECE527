#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

# --------------------------------
# 0) Config
# --------------------------------
# Change this if your CSV has a different name
CSV_PATH = "Train_data.csv"
# e.g. CSV_PATH = "Train_data.csv"


# --------------------------------
# 1) Helper: pretty prediction function
# --------------------------------
def predict_for_unroll_k(model, target_cols, unroll_k_values):
    """
    Print model predictions for one or more unroll_k values.

    Parameters
    ----------
    model : trained scikit-learn regressor
    target_cols : list of strings, column names of the outputs
    unroll_k_values : int or list-like of ints
    """
    import numpy as np
    import pandas as pd

    # Allow a single int or a list of ints
    if isinstance(unroll_k_values, (int, float)):
        unroll_k_values = [int(unroll_k_values)]

    # Build feature dataframe
    X_new = pd.DataFrame({"unroll_k": unroll_k_values})

    preds = model.predict(X_new)

    print("\n=== Predictions ===")
    for k, pred in zip(unroll_k_values, preds):
        print(f"\nConfig: unroll_k = {k}")
        for col, val in zip(target_cols, pred):
            print(f"  {col:20s} ≈ {val:.1f}")
    print("===================\n")


# --------------------------------
# 2) Load & clean data
# --------------------------------
df = pd.read_csv(CSV_PATH)

df = df.drop(
    columns=[
        "config_name",
        "interval_min_ii",
        "interval_max_ii",
        "STD",
        "Unnamed: 11",
        "pipeline_ii",
    ],
    errors="ignore",
)

if "PYNQ_Latency (us)" in df.columns:
    df = df.rename(columns={"PYNQ_Latency (us)": "pynq_latency_us"})

print("Columns after cleaning:")
print(df.columns.tolist())
print("\nData preview:")
print(df.head())

# --------------------------------
# 3) Features (X) and targets (y)
# --------------------------------
feature_cols = ["unroll_k"]
X = df[feature_cols].copy()

target_cols = [
    "latency_min_cycles",
    "latency_max_cycles",
    "LUT",
    "FF",
    "BRAM_18K",
    "DSP48E",
]
if "pynq_latency_us" in df.columns:
    target_cols.append("pynq_latency_us")

y = df[target_cols]

# --------------------------------
# 4) Train model on ALL data
# --------------------------------
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42,
)

model.fit(X, y)

# --------------------------------
# 5) Training-set MAE (sanity check)
# --------------------------------
y_pred = model.predict(X)

print("\nTraining-set MAE (on all points):")
from sklearn.metrics import mean_absolute_error

for i, col in enumerate(target_cols):
    mae = mean_absolute_error(y.iloc[:, i], y_pred[:, i])
    print(f"{col:20s}  MAE = {mae:10.2f}")

print(
    "\n Note: Only ~6 samples. This is an interpolator over unroll_k, "
    "not a strongly generalizable model."
)

# --------------------------------
# 6) Example usage of helper
# --------------------------------
example_unrolls = [1, 2, 4, 6, 8, 16]
predict_for_unroll_k(model, target_cols, example_unrolls)

# --------------------------------
# 7) Save model
# --------------------------------
dump(model, "gemm_regressor_unroll_only.joblib")
print("Model saved to gemm_regressor_unroll_only.joblib")
