# Lightweight EIP compliance audit
# Checks: configs/dataset.yaml keys, matplotlibrc, dataset checksum+schema, label map, families order.
import os
import sys
from src.utils.data_prep import load_dataset, load_config, check_schema


def green(msg):
    print("\033[92m" + msg + "\033[0m")


def red(msg):
    print("\033[91m" + msg + "\033[0m")


def main():
    ok = True
    if not os.path.exists("configs/dataset.yaml"):
        red("[FAIL] configs/dataset.yaml missing")
        ok = False
    else:
        cfg = load_config("configs/dataset.yaml")
        for key in [
            "dataset_path",
            "sha256",
            "columns",
            "label_encoding",
            "families_order",
        ]:
            if key not in cfg:
                red(f"[FAIL] dataset.yaml missing key: {key}")
                ok = False
        le = cfg.get("label_encoding", {})
        if "Attack" not in le or "Benign" not in le:
            red("[FAIL] label_encoding must include Attack and Benign")
            ok = False

    if not os.path.exists("matplotlibrc"):
        red("[FAIL] matplotlibrc missing in repo root")
        ok = False

    try:
        df = load_dataset("configs/dataset.yaml")
        check_schema(df)
        green("[OK] dataset loaded & schema verified")
    except Exception as e:
        red(f"[FAIL] dataset/schema check: {e}")
        ok = False

    if ok:
        green("EIP AUDIT: ALL GREEN")
        sys.exit(0)
    else:
        red("EIP AUDIT: FAIL (see messages above)")
        sys.exit(2)


if __name__ == "__main__":
    main()
