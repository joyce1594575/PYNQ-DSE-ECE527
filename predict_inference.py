import pandas as pd
from joblib import load

# 1) Reload model
model = load(r"D:\MP4_starter\MP4\mp4\mp4\accelerator_hls\accelerator_hls\gemm_regressor_unroll_only.joblib")

# 2) Recreate target_cols (must match training)
target_cols = [
    "latency_min_cycles",
    "latency_max_cycles",
    "LUT",
    "FF",
    "BRAM_18K",
    "DSP48E",
    "pynq_latency_us",   # include only if your CSV had it
]

# 3) Paste the helper definition (same as in the script) and use it:
def predict_for_unroll_k(model, target_cols, unroll_k_values):
    import numpy as np
    import pandas as pd
    if isinstance(unroll_k_values, (int, float)):
        unroll_k_values = [int(unroll_k_values)]
    X_new = pd.DataFrame({"unroll_k": unroll_k_values})
    preds = model.predict(X_new)
    print("\n=== Predictions ===")
    for k, pred in zip(unroll_k_values, preds):
        print(f"\nConfig: unroll_k = {k}")
        for col, val in zip(target_cols, pred):
            print(f"  {col:20s} ≈ {val:.1f}")
    print("===================\n")

# 4) Call it
predict_for_unroll_k(model, target_cols, [3, 6, 12])
