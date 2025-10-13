# Lightweight EIP compliance audit
# Checks: configs/dataset.yaml keys, matplotlibrc, dataset checksum+schema (unless light mode), label map, families order.
import os
import sys

from src.utils.data_prep import load_dataset, load_config, check_schema


def green(msg):
    print("\033[92m" + msg + "\033[0m")


def red(msg):
    print("\033[91m" + msg + "\033[0m")


def main():
    ok = True

    # --- Config checks ---
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

        # label encoding must have both classes
        le = cfg.get("label_encoding", {})
        if "Attack" not in le or "Benign" not in le:
            red("[FAIL] label_encoding must include Attack and Benign")
            ok = False

        # checksum presence (string recorded), even if we skip loading in CI
        if not cfg.get("sha256"):
            red(
                "[FAIL] sha256 is empty in configs/dataset.yaml (run compute_checksum.py)"
            )
            ok = False

    # --- Matplotlib config present ---
    if not os.path.exists("matplotlibrc"):
        red("[FAIL] matplotlibrc missing in repo root")
        ok = False

    # --- Dataset/schema check (full vs light mode) ---
    light_mode = os.getenv("EIP_LIGHT_AUDIT", "0") == "1"
    if light_mode:
        green("[OK] light audit mode: skipping dataset load/schema check")
    else:
        try:
            df = load_dataset("configs/dataset.yaml")
            check_schema(df)
            green("[OK] dataset loaded & schema verified")
        except Exception as e:
            red(f"[FAIL] dataset/schema check: {e}")
            ok = False

    # --- Final status ---
    if ok:
        green("EIP AUDIT: ALL GREEN")
        sys.exit(0)
    else:
        red("EIP AUDIT: FAIL (see messages above)")
        sys.exit(2)


if __name__ == "__main__":
    main()
