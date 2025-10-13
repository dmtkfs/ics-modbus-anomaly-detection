# ICS Modbus Anomaly Detection
![EIP Audit](https://github.com/dmtkfs/ics-modbus-anomaly-detection/actions/workflows/eip-audit.yml/badge.svg)

This project implements **baseline anomaly detection for Industrial Control System (ICS) Modbus traffic**, combining **rule-based heuristics** and **supervised machine-learning baselines** under a unified **Evaluation Integrity Protocol (EIP)** for reproducibility and fair comparison.

The work is based on the **CIC Modbus 2023 dataset**, and evaluates both detection families: **Heuristics** (domain logic) and **ML models** (data-driven) under the same dataset, metrics and evaluation procedures.

## Project Overview

### Goals
- Detect anomalous Modbus traffic using two complementary approaches:
  - **Heuristic detectors**
    - **H1 – Write-Rate Spike:** detects sudden surges in write operations.
    - **H2 – Function-Code Anomalies:** detects abnormal protocol behavior.
  - **Machine-Learning baselines**
    - Logistic Regression
    - Random Forest
    - Unsupervised Isolation Forest (optional)
- Evaluate both heuristics and ML models using a **Leave-One-Attack-Out (LOAO)** methodology across attack families.
- Maintain full cross-team integrity using our **Evaluation Integrity Protocol (EIP)**, a fixed standard for dataset versioning, metrics, figure formats and random seeds.

## Repository Structure

```

ICS-MODBUS-ANOMALY-DETECTION/
│
├── .github/
│   └── workflows/
│       └── eip-audit.yml         # CI gate: runs EIP audit on push/PR
│
├── configs/
│   └── dataset.yaml              # Dataset path, SHA-256, schema, label map, family order
│
├── data/
│   ├── raw/                      # Original CIC Modbus 2023 PCAPs [not committed]
│   └── processed/
│       └── master.csv            # Merged and labeled master dataset (attack + benign) [not committed]
│
├── docs/
│   ├── EIP_Checklist.md          # Tick-before-merge reproducibility checklist
│   └── Evaluation_Integrity_Protocol.md  # Definition and compliance description
│
├── figures/                      # Auto-saved plots and metric figures
│
├── scripts/
│   ├── __init__.py
│   ├── compute_checksum.py       # Pins dataset SHA-256 in configs/dataset.yaml
│   ├── eip_audit.py              # Verifies schema, checksum, label encoding, matplotlibrc
│   ├── proc_dataset_audit.py     # Dataset-level preprocessing audit (independent)
│   ├── run_baselines.py          # Runs ML baselines (LogReg, RF) and writes results CSV
│   └── run_loao.py               # LOAO evaluation across attack families
│
├── src/
│   ├── utils/
│   │   ├── data_prep.py          # Config/dataset loaders, checksum, schema checks
│   │   ├── metrics.py            # Shared metric computation, 3-decimal CSV writer, plotting
│   │   ├── plot_utils.py         # Standardized figure naming helpers
│   │   └── __init__.py
│   ├── heuristics.py             # (planned) H1/H2 rule-based detectors
│   ├── ml_baselines.py           # (planned) Model definitions separate from scripts
│   ├── loao.py                   # (planned) reusable LOAO routines
│   └── data_processing.py        # (planned) preprocessing utilities for dataset refinement
│
├── matplotlibrc                  # Global plotting style (DPI, fonts, sizes)
├── requirements.txt              # Stable dependencies (latest verified releases)
├── LICENSE
└── README.md

````

## Stable Dependency Versions

All packages are pinned to the **latest stable (non-beta)** versions as of October 2025:

| Package | Version | Purpose |
|----------|----------|----------|
| `numpy` | **2.3.3** | numeric array operations |
| `pandas` | **2.3.3** | dataset loading and manipulation |
| `scikit-learn` | **1.7.2** | ML algorithms and metrics |
| `matplotlib` | **3.10.7** | plotting |
| `PyYAML` | **6.0.3** | YAML configuration parsing |

## Evaluation Integrity Protocol (EIP)

The **EIP** defines reproducibility and comparability standards across both teams (Heuristics & ML). It enforces:

| Category | Fixed Standard |
|-----------|----------------|
| **Dataset identity** | `data/processed/master.csv` pinned via SHA-256 in `configs/dataset.yaml` |
| **Schema** | 10 required columns `[Time, Source, Destination, Length, Ports, FunctionCode, Label, AttackFamily, FunctionCodeNum]` |
| **Labels** | `Attack=1`, `Benign=0` |
| **Families order** | `[External, Compromised-IED, Compromised-SCADA]` |
| **Seed** | `random_state=42` |
| **Metrics** | Precision, Recall, F1; ROC-AUC and PR-AUC (for ML) |
| **Rounding** | 3-decimal precision across all CSV outputs |
| **Figures** | DPI = 300, unified font sizes per `matplotlibrc` |
| **Results format** | All metrics written via `src/utils/metrics.write_metrics_csv()` |
| **Audit gate** | `python -m scripts.eip_audit` must output **“ALL GREEN”** before commits/merges |

## Workflow Summary

### 1. Compute dataset checksum
```bash
python -m scripts.compute_checksum
````

Writes the SHA-256 of `master.csv` into `configs/dataset.yaml`.

### 2. Run EIP audit

```bash
python -m scripts.eip_audit
```

Verifies schema, checksum, label encoding, and matplotlib configuration.

### 3. Run baseline ML models

```bash
python -m scripts.run_baselines
```

Trains Logistic Regression and Random Forest (optional: Isolation Forest) and appends standardized metric rows to `results/metrics.csv`.

### 4. Run Leave-One-Attack-Out (LOAO)

```bash
python -m scripts.run_loao
```

Performs LOAO evaluation per attack family, computing **average recall** per model.

### 5. Verify continuous-integration audit (GitHub Actions)

Every push or pull request automatically runs the same audit via
`.github/workflows/eip-audit.yml`.
View the result under the **Actions** tab on GitHub.

## CI: EIP Audit (Light in CI, Full Locally)

This repository uses a **GitHub Actions workflow** (`.github/workflows/eip-audit.yml`) to enforce EIP integrity.

**What CI checks (light mode):**

* `configs/dataset.yaml` exists and includes:

  * `dataset_path`, **non-empty** `sha256`, `columns`, `label_encoding`, `families_order`
  * `label_encoding` must include both `Attack` and `Benign`
* `matplotlibrc` exists in the repo root

**What CI skips:**
Loading `data/processed/master.csv` and schema validation (because the dataset is not stored in the repo).

**Why “light” mode?**
The dataset is too large for the repo. CI still validates all **static** EIP guarantees; **full** checks run locally.

**Run the full audit locally:**

```bash
python -m scripts.eip_audit
# expects data/processed/master.csv to exist locally
```

**Force light mode locally (optional):**

```bash
EIP_LIGHT_AUDIT=1 python -m scripts.eip_audit
```

**Where to see results:**
GitHub → **Actions** → **EIP Audit** (each push/PR shows pass/fail).

## Output Convention

* **Metrics file:** `results/metrics.csv`
  Columns → `model, precision, recall, f1, roc_auc, pr_auc, avg_loao_recall, notes`
* **Figures:** saved under `figures/<team>/<model_metric>.png`
* **Footer metadata (auto-appended):** dataset name, commit hash, seed, UTC date

## How We Merge Changes

1. Create a feature branch (e.g., `feat/h1-heuristic`, `exp/rf-tuning`).
2. Push and open a Pull Request (PR) into `main`.
3. Ensure **EIP Audit** (GitHub Actions) passes.
4. Merge the PR (branch protection enforces the check).
5. Direct pushes to `main` are disabled for consistency.

---

## Dataset Reference

The project uses the **[CIC Modbus 2023 dataset](https://www.unb.ca/cic/datasets/modbus-2023.html)**.
Because of their large size, raw PCAPs and the `master.csv` are excluded from the repo.
To reproduce results:

1. Place PCAPs under `data/raw/Modbus Dataset/`
2. Process them into CSVs with Modbus filter (`tcp.port == 502`)
3. Merge and label them into `data/processed/master.csv`
4. Compute checksum and rerun EIP audit

## Future Work

* Implement the heuristic detectors (H1 & H2) directly in `src/heuristics.py`
* Add visualization scripts for heuristic vs. ML comparison
* Integrate MITRE ATT&CK-for-ICS mapping to figures/report (`docs/mitre_mapping.md`)
* Extend EIP CI to include metric-threshold enforcement

## Acknowledgements

This work is part of a project for **INSE 6640 Smart Grids and Control System Security**, focusing on integrity and anomaly detection in ICS Modbus environments.
All processing, baselines and evaluations are performed offline following the EIP reproducibility protocol.

