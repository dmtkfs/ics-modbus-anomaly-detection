# scripts/smoke_dataset.py
from __future__ import annotations
import random
import numpy as np
from collections import Counter

from src.utils.data_prep import load_dataset, load_config, sha256_file
from src.utils.ml_data_prep import load_ml_config, make_supervised_xy


def main():
    # Fixed seed per EIP
    random.seed(42)
    np.random.seed(42)

    # Load dataset + ML config
    df = load_dataset("configs/dataset.yaml")
    ml_cfg = load_ml_config("configs/ml.yaml")

    print("[smoke] Loaded master.csv successfully")
    print(f"[smoke] Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Show hashes
    ds_cfg = load_config("configs/dataset.yaml")
    expected = ds_cfg.get("sha256")
    actual = sha256_file(ds_cfg["dataset_path"])
    print(f"[checksum] Expected: {expected}")
    print(f"[checksum] Actual:   {actual}")
    if expected and actual == expected:
        print("[checksum] Dataset checksum verified, EIP integrity confirmed.")

    # Verify label distribution (text labels)
    y_text = df[ml_cfg["label_column"]]
    counts = Counter(y_text)
    total = sum(counts.values())
    pct = {k: f"{(v/total)*100:.2f}%" for k, v in counts.items()}
    print(f"[smoke] Label split: {counts} ({pct})")

    # Quick X,y extraction
    X, y = make_supervised_xy(df, ml_cfg)
    print(f"[smoke] X columns: {list(X.columns)}")
    print(f"[smoke] X shape: {X.shape}, y shape: {y.shape}")
    assert set(y.unique()) <= {0, 1}, "Label encoding must be {0,1}"

    # Attack Family presence (for LOAO)
    fam_col = ml_cfg["attack_family_column"]
    fam_counts = df[fam_col].value_counts(dropna=False).to_dict()
    print(f"[smoke] Attack families present: {fam_counts}")

    print("[smoke] Dataset wiring looks good. You can proceed to Step 2.")


if __name__ == "__main__":
    main()
