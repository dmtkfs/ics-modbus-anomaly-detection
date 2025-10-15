# ML-only LOAO: per-family recall for LogReg/RF (+ optional IF reference).
# Outputs -> results/ml/feat-ml-calibration-loao/

import argparse
import pathlib
from datetime import datetime, UTC

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils.data_prep import load_config, check_schema, verify_checksum, sha256_file
from src.utils.ml_data_prep import load_ml_config, make_supervised_xy


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _light_load_with_family(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    cfg = load_config(ds_yaml_path)
    csv_path = cfg["dataset_path"]

    exp = cfg.get("sha256")
    if exp and not verify_checksum(csv_path, exp):
        actual = sha256_file(csv_path)
        raise ValueError(f"Checksum mismatch.\nExpected: {exp}\nActual:   {actual}")

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
    n = len(y)
    if cap is None or n <= cap:
        return X, y
    sss = StratifiedShuffleSplit(n_splits=1, train_size=cap, random_state=seed)
    idx, _ = next(sss.split(X, y))
    return X[idx], y[idx]


def _models(seed: int, with_if: bool):
    models = {
        "LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=20,
            min_samples_leaf=2,
            bootstrap=True,
            max_samples=1_000_000,  # per-tree subsample to save RAM
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,  # avoid multi-proc duplication
        ),
    }
    if with_if:
        models["IsolationForest"] = IsolationForest(
            n_estimators=200,
            contamination="auto",
            max_samples="auto",
            bootstrap=False,
            random_state=seed,
            n_jobs=1,
        )
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-calibration-loao/loao_{_ts()}",
        help="Prefix for outputs (CSV).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-col", default="Attack Family")
    ap.add_argument(
        "--with-if",
        action="store_true",
        help="Include Isolation Forest reference in LOAO.",
    )
    # caps to keep memory in check
    ap.add_argument("--cap-train", type=int, default=1_000_000)
    ap.add_argument("--cap-test", type=int, default=500_000)
    args = ap.parse_args()

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load_with_family("configs/dataset.yaml", ml_cfg)
    X_df, y_ser = make_supervised_xy(df, ml_cfg)

    X_all = X_df.values.astype(np.float32, copy=False)
    y_all = y_ser.values.astype(np.int8, copy=False)

    families = sorted(df[args.family_col].unique())
    models = _models(args.seed, with_if=args.with_if)
    rows = []

    for model_name, model in models.items():
        for fam in families:
            train_idx = (df[args.family_col] != fam).values
            test_idx = (df[args.family_col] == fam).values

            X_train = X_all[train_idx]
            y_train = y_all[train_idx]
            X_test = X_all[test_idx]
            y_test = y_all[test_idx]

            # ensure we actually have attacks in held-out slice
            if not (y_test == 1).any():
                continue

            # memory-safe caps (stratified) — not applied to IF fitting semantics
            X_train, y_train = stratified_cap(
                X_train, y_train, cap=args.cap_train, seed=args.seed
            )
            X_test, y_test = stratified_cap(
                X_test, y_test, cap=args.cap_test, seed=args.seed
            )

            if model_name == "IsolationForest":
                model.fit(X_train)  # unsupervised
                yhat = (model.predict(X_test) == -1).astype(
                    np.int8
                )  # -1 => anomaly => attack(1)
            else:
                model.fit(X_train, y_train)
                yhat = model.predict(X_test).astype(np.int8, copy=False)

            tp = int(((y_test == 1) & (yhat == 1)).sum())
            fn = int(((y_test == 1) & (yhat == 0)).sum())
            recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)

            rows.append({"family": fam, "model": model_name, "recall": recall})

    out_path = f"{args.out_prefix}_metrics.csv"
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[loao] wrote {out_path}")


if __name__ == "__main__":
    main()
