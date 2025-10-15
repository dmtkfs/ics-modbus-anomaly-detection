import argparse
import datetime
import pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.data_prep import load_config, check_schema, verify_checksum, sha256_file
from src.utils.ml_data_prep import load_ml_config, make_supervised_xy


def _ts():
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _light_load_with_family(ds_yaml_path: str, ml_cfg: dict) -> pd.DataFrame:
    ds_cfg = load_config(ds_yaml_path)
    csv_path = ds_cfg["dataset_path"]
    exp = ds_cfg.get("sha256")
    if exp and not verify_checksum(csv_path, exp):
        raise ValueError(f"Checksum mismatch: {sha256_file(csv_path)} != {exp}")
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


def _models(seed=42):
    return {
        "LogReg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=None, n_jobs=-1, random_state=seed
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-prefix",
        default=f"results/ml/feat-ml-calibration-loao/loao_{_ts()}",
        help="Prefix for outputs (CSV, plots).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--family-col", default="Attack Family")
    args = ap.parse_args()

    ml_cfg = load_ml_config("configs/ml.yaml")
    df = _light_load_with_family("configs/dataset.yaml", ml_cfg)
    X_df, y_ser = make_supervised_xy(df, ml_cfg)

    families = sorted(df[args.family_col].unique())
    results = []
    for name, model in _models(args.seed).items():
        for fam in families:
            train_idx = df[args.family_col] != fam
            test_idx = df[args.family_col] == fam
            # ensure held-out has attacks; if not, skip
            # (we assume LabelNum=1 is Attack after encoding in make_supervised_xy)
            y_test_slice = y_ser.values[test_idx]
            if not ((y_test_slice == 1).any()):
                continue
            model.fit(X_df.values[train_idx], y_ser.values[train_idx])
            yhat = model.predict(X_df.values[test_idx])
            tp = int(((y_test_slice == 1) & (yhat == 1)).sum())
            fn = int(((y_test_slice == 1) & (yhat == 0)).sum())
            rec = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
            results.append({"family": fam, "model": name, "recall": rec})

    out_dir = pathlib.Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = f"{args.out_prefix}_metrics.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[loao] wrote {out_csv}")


if __name__ == "__main__":
    main()
