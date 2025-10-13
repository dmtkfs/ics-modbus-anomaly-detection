from __future__ import annotations
import hashlib
import os
import yaml
import pandas as pd

REQUIRED_COLS = [
    "Time",
    "Source",
    "Destination",
    "Length",
    "Source Port",
    "Destination Port",
    "Function Code",
    "Label",
    "Attack Family",
    "FunctionCodeNum",
]


def load_config(path="configs/dataset.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(csv_path: str, expected_sha256: str) -> bool:
    if not expected_sha256:
        return False
    return sha256_file(csv_path) == expected_sha256


def check_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset schema mismatch. Missing columns: {missing}")


def encode_labels(df: pd.DataFrame, mapping: dict[str, int]) -> pd.Series:
    unknown = set(df["Label"].unique()) - set(mapping.keys())
    if unknown:
        raise ValueError(f"Unknown labels present: {unknown}")
    return df["Label"].map(mapping).astype(int)


def load_dataset(cfg_path="configs/dataset.yaml") -> pd.DataFrame:
    cfg = load_config(cfg_path)
    csv_path = cfg["dataset_path"]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Update configs/dataset.yaml.")
    if cfg.get("sha256"):
        if not verify_checksum(csv_path, cfg["sha256"]):
            raise ValueError(
                "Dataset checksum mismatch. Run scripts/compute_checksum.py to refresh."
            )
    df = pd.read_csv(csv_path)
    check_schema(df)
    return df
