# scripts/smoke_heuristics.py
from __future__ import annotations
import os
import time
import random
import pickle
import numpy as np
import pandas as pd

from src.heuristics import ModbusHeuristicsDetector
from src.utils.data_prep import load_config, sha256_file  # keep for EIP checksum

SMOKE_ROWS = 1_000_000  # cap for slice

USECOLS_TARGET = [
    "Source",
    "Label",
    "AttackFamily",  # may be missing in some exports – handled below
    "FunctionCodeNum",
    "Destination Port",
    "Source Port",
]

DTYPES_BASE = {
    "Source": "category",
    "Label": "category",
    "AttackFamily": "category",
    "FunctionCodeNum": "int16",  # fits Modbus function codes
    "Destination Port": "Int32",  # allow NA with nullable Int32
    "Source Port": "Int32",
}


def load_light_dataset():
    cfg = load_config("configs/dataset.yaml")
    path = cfg["dataset_path"]

    # 1) Sniff header to know what actually exists
    header = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()

    usecols = [c for c in USECOLS_TARGET if c in available]
    dtypes = {k: v for k, v in DTYPES_BASE.items() if k in usecols}

    # 2) Read with minimal columns + tight dtypes
    try:
        df = pd.read_csv(path, usecols=usecols, dtype=dtypes, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path, usecols=usecols, dtype=dtypes, low_memory=False)

    # 3) If AttackFamily missing, add for schema parity
    if "AttackFamily" not in df.columns:
        df["AttackFamily"] = "Unknown"
        df["AttackFamily"] = df["AttackFamily"].astype("category")

    return df, cfg


def stratified_sample(df, label_col="Label", max_rows=SMOKE_ROWS, rng=42):
    classes = df[label_col].unique().tolist()
    per_class = max(1, max_rows // max(1, len(classes)))
    parts = []
    for c in classes:
        sub = df[df[label_col] == c]
        n = min(per_class, len(sub))
        parts.append(sub.sample(n=n, random_state=rng) if n < len(sub) else sub)
    out = pd.concat(parts, ignore_index=True).copy() if parts else pd.DataFrame()
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=rng)
    return out


def main():
    random.seed(42)
    np.random.seed(42)
    t0 = time.time()

    # Load minimal columns + checksum
    df, ds_cfg = load_light_dataset()
    expected = ds_cfg.get("sha256")
    actual = sha256_file(ds_cfg["dataset_path"])
    print(
        "[smoke] Dataset checksum verified (EIP integrity confirmed)"
        if expected and actual == expected
        else "[smoke] WARNING: Dataset checksum mismatch"
    )
    print(f"[smoke] Loaded dataset (light): {len(df):,} rows, {df.shape[1]} cols")

    # Stratified slice
    df_small = stratified_sample(df, label_col="Label", max_rows=SMOKE_ROWS, rng=42)
    print(f"[smoke] Using stratified slice of {len(df_small):,} rows")

    det = ModbusHeuristicsDetector()
    print("[smoke] Computing baselines…")
    det.compute_baselines(df_small)
    print("[smoke] Running H1…")
    h1 = det.detect_h1_write_spikes(df_small)
    print("[smoke] Running H2…")
    h2 = det.detect_h2_variant_f(df_small)
    print("[smoke] Evaluating…")
    results = det.evaluate_heuristics(df_small, h1, h2)
    print("[smoke] Visualizing…")
    det.visualize_results(df_small, h1, h2, results)
    print("[smoke] Exporting…")
    det.export_results(df_small, h1, h2)

    # File existence
    for f in [
        "data/baselines/baseline_write_ratios.pkl",
        "data/baselines/baseline_fc_distributions.pkl",
        "figures/heuristics/performance_comparison.png",
        "figures/heuristics/confusion_combined.png",
        "results/heuristics_metrics.csv",
        "data/processed/heuristics_results.csv",
    ]:
        assert os.path.exists(f), f"{f} missing"

    with open("data/baselines/baseline_write_ratios.pkl", "rb") as f:
        wr = pickle.load(f)
    with open("data/baselines/baseline_fc_distributions.pkl", "rb") as f:
        fc = pickle.load(f)
    assert isinstance(wr, dict) and isinstance(fc, dict)
    print(
        f"[smoke] Baselines OK: write_ratios={len(wr)} sources, fc_dists={len(fc)} sources"
    )

    print(f"[smoke] Heuristics smoke completed in {time.time()-t0:.1f}s")
    print(f"[smoke] Results: {results}")


if __name__ == "__main__":
    main()
