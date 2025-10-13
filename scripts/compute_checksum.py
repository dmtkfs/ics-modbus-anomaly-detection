import hashlib
import sys
import os
import yaml

CFG = "configs/dataset.yaml"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(CFG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    csv_path = cfg["dataset_path"]
    if not os.path.exists(csv_path):
        print(
            f"[ERROR] {csv_path} not found. Update configs/dataset.yaml and place your master CSV."
        )
        sys.exit(1)
    digest = sha256_file(csv_path)
    cfg["sha256"] = digest
    with open(CFG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[OK] Updated SHA-256 for {csv_path}: {digest}")


if __name__ == "__main__":
    main()
