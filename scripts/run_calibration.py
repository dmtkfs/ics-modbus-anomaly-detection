# Calibrates LogReg/RF with threshold sweeps, saves models + plots.
# Outputs -> results/ml/feat-ml-calibration-loao/

import argparse
import csv
import pathlib
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from src.utils.ml_data_prep import load_ml_config, make_supervised_xy
from src.utils.data_prep import load_config, check_schema, verify_checksum, sha256_file
from src.ml.calibration import (
    calibrate,
)  # uses RF with n_jobs=1, max_samples subsampling


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _light_load(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    """Checksum + header schema + read only needed cols with tight dtypes."""
    ds_cfg = load_config(ds_yaml_path)
    csv_path = ds_cfg["dataset_path"]

    exp = ds_cfg.get("sha256")
    if exp and not verify_checksum(csv_path, exp):
        actual = sha256_file(csv_path)
        raise ValueError(f"Checksum mismatch.\nExpected: {exp}\nActual:   {actual}")

    # header-only schema check
    check_schema(pd.read_csv(csv_path, nrows=0))

    needed = list(ml_cfg["features"]) + [
        ml_cfg["label_column"],
        ml_cfg["attack_family_column"],
    ]
    dtype_map = {
        "Length": "int32",
        "Source Port": "int32",
        "Destination Port": "int32",
        "FunctionCodeNum": "int16",
        ml_cfg["label_column"]: "string",
        ml_cfg["attack_family_column"]: "string",
    }
    return pd.read_csv(
        csv_path,
        usecols=needed,
        dtype=dtype_map,
        engine="c",
        low_memory=True,
        memory_map=True,
    )


def stratified_cap(X: np.ndarray, y: np.ndarray, cap: int | None, seed: int = 42):
    """Return up to `cap` rows, stratified by y (keeps class ratio)."""
    n = len(y)
    if cap is None or n <= cap:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-calibration-loao/calib_{_ts()}",
        help="Prefix for outputs (CSV, models, plots).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    # caps to keep memory in check on 8–16 GB machines
    ap.add_argument("--cap-train", type=int, default=2_000_000)
    ap.add_argument("--cap-val", type=int, default=500_000)
    args = ap.parse_args()

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load("configs/dataset.yaml", ml_cfg)

    X_df, y_ser = make_supervised_xy(df, ml_cfg)
    X = X_df.values.astype(np.float32, copy=False)
    y = y_ser.values.astype(np.int8, copy=False)

    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    # memory-safe caps (stratified)
    Xtr, ytr = stratified_cap(Xtr, ytr, cap=args.cap_train, seed=args.seed)
    Xva, yva = stratified_cap(Xva, yva, cap=args.cap_val, seed=args.seed)

    pathlib.Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)

    rows = calibrate(Xtr, ytr, Xva, yva, args.out_prefix, seed=args.seed)

    csv_path = f"{args.out_prefix}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "threshold",
                "date_utc",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[calibration] wrote {csv_path}")


if __name__ == "__main__":
    main()
