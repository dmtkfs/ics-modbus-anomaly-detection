# ICS Modbus Anomaly Detection
![EIP Audit](https://github.com/dmtkfs/ics-modbus-anomaly-detection/actions/workflows/eip-audit.yml/badge.svg)


**Baseline intrusion-detection framework for Industrial Control Systems (ICS) using Modbus/TCP traffic.**
Implements two complementary detection layers — **rule-based heuristics** and **machine learning baselines** — unified by a strict **Evaluation Integrity Protocol (EIP)** that guarantees reproducibility, dataset consistency and comparable metrics.

## Overview

This project analyzes the **CIC Modbus 2023 dataset** to detect anomalous behavior in industrial network traffic.

* **Heuristic detectors** provide interpretable, lightweight rule checks
* **Machine learning models** (Logistic Regression, Random Forest, Isolation Forest) provide adaptive statistical detection
* Both layers share the same dataset, schema, metrics and seed under the **EIP** standard
* A **PowerShell script** automates end-to-end evaluation for reproducibility

## Repository Structure

```
ics-modbus-anomaly-detection/
│
├── .github/
│   └── workflows/
│       └── eip-audit.yml           # GitHub Actions CI audit enforcing EIP
│
├── configs/
│   ├── dataset.yaml                # Dataset path, SHA-256, schema, label map
│   └── ml.yaml                     # ML configuration (features, labels, seed)
│
├── docs/
│   ├── appendix_ml_final_run.md    # Final Phase III ML notes (artifacts & metrics)
│   ├── EIP_Checklist.md            # Tick-before-merge reproducibility checklist
│   └── Evaluation_Integrity_Protocol.md  # Full EIP specification
│
├── figures/
│   └── ml/
│       └── .gitkeep                # Placeholder (figures generated locally)
│
├── results/
│   └── ml/
│       └── .gitkeep                # Placeholder (CSV results generated locally)
│
├── scripts/
│   ├── __init__.py
│   ├── aggregate_phase3_metrics.py # Aggregates calibration + LOAO outputs
│   ├── compute_checksum.py         # Computes and pins dataset SHA-256
│   ├── eip_audit.py                # Validates schema, checksum, matplotlibrc
│   ├── proc_dataset_audit.py       # Optional preprocessing audit
│   ├── run_baselines.py            # Trains LR/RF/IF baselines (80/20 split)
│   ├── run_calibration.py          # Legacy calibrator (unbalanced)
│   ├── run_calibration_balanced.py # Final constrained calibration (balanced)
│   ├── run_final_ml.ps1            # Full PowerShell pipeline (audit→train→LOAO→aggregate)
│   ├── run_loao.py                 # Simple LOAO prototype
│   ├── run_loao_ml.py              # ML-only LOAO (legacy)
│   ├── run_loao_ml_balanced.py     # Balanced LOAO for LR/RF/IF (Phase III)
│   ├── smoke_dataset.py            # Dataset presence & schema sanity check
│   └── smoke_heuristics.py         # Quick heuristics dry-run on subset
│
├── src/
│   ├── ml/
│   │   ├── balanced.py             # Class balancing and tree growth logic
│   │   └── calibration.py          # Calibration sweep & constraint selection
│   ├── utils/
│   │   ├── data_prep.py            # Dataset/config loaders, checksum utilities
│   │   ├── metrics.py              # Metric computation & CSV writer
│   │   ├── ml_data_prep.py         # ML-specific data preparation helpers
│   │   └── plot_utils.py           # Standardized figure styling
│   ├── heuristics.py               # Implements H1/H2F detectors
│   └── __init__.py
│
├── .gitignore                      # Excludes data/, cache, and local artifacts
├── LICENSE                         # Open license declaration
├── matplotlibrc                     # Unified plotting style (DPI, fonts)
├── requirements.txt                 # Stable dependencies (NumPy, Pandas, etc.)
└── README.md
```

## Evaluation Integrity Protocol (EIP)

EIP enforces **reproducibility and comparability** across all runs.

| Standard             | Description                                                                                                                             |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset identity** | `data/processed/master.csv` pinned via SHA-256 in `configs/dataset.yaml`                                                                |
| **Schema**           | 10 columns – `[Time, Source, Destination, Length, Source Port, Destination Port, Function Code, Label, Attack Family, FunctionCodeNum]` |
| **Labels**           | `Attack = 1`, `Benign = 0`                                                                                                              |
| **Families order**   | `[External, Compromised-IED, Compromised-SCADA]`                                                                                        |
| **Random seed**      | 42                                                                                                                                      |
| **Metrics**          | Precision, Recall, F1 (+ ROC-AUC / PR-AUC for ML)                                                                                       |
| **Figures**          | DPI 300, standard fonts per `matplotlibrc`                                                                                              |
| **Audit**            | `python -m scripts.eip_audit` → **“ALL GREEN”** before merge                                                                            |

A lightweight version of this audit runs automatically in **GitHub Actions** for every push or pull request.

## How to Run

### 1. Dataset Checksum & Audit

```bash
python -m scripts.compute_checksum     # write SHA-256 into configs/dataset.yaml
python -m scripts.eip_audit            # full integrity check
```

### 2. Heuristic Detection

```bash
python -m src.heuristics
```

Generates:

* `results/heuristics_metrics.csv`
* `figures/heuristics/confusion_combined.png`
* `figures/heuristics/performance_comparison.png`
* `figures/heuristics/recall_by_attack_family.png`

Executes H1 (Write Rate Spike) and H2 (Function Code + Role Anomaly) in ~5 minutes on standard CPU.

### 3. Machine-Learning Baselines

Train baseline models (80/20 split):

```bash
python -m scripts.run_baselines
```

Calibrate thresholds and LOAO (Leave-One-Attack-Out) evaluation:

```bash
python -m scripts.run_calibration_balanced
python -m scripts.run_loao_ml_balanced
python -m scripts.aggregate_phase3_metrics
```

### 4. Fully Automated ML Pipeline (PowerShell)

Run every step under EIP control:

```powershell
.\run_final_ml.ps1
```

Performs:
Audit → Baselines → Balanced calibration → LOAO (simple + balanced) → Aggregate → Light audit
Outputs stored in `results/ml/final_<timestamp>/` and `figures/ml/final_<timestamp>/`.

## Key Findings (Shortened)

| Detector                               | Precision | Recall | F1    | Notes                               |
| -------------------------------------- | --------- | ------ | ----- | ----------------------------------- |
| **H1: Write-Rate Spike**              | 0.948     | 0.866  | 0.905 | Detects write surges                |
| **H2: Function-Code & Role Anomaly** | 1.000     | 0.306  | 0.469 | Flags mixed-role clients            |
| **Combined (H1 ∨ H2)**                | 0.948     | 0.866  | 0.905 | Balanced precision-recall           |
| **Logistic Regression (80/20)**        | 0.955     | 0.462  | 0.623 | Supervised baseline                |
| **Random Forest (80/20)**              | 0.962     | 0.305  | 0.463 | Tree-based baseline                 |
| **Isolation Forest (unsupervised)**    | 0.948     | 0.786  | 0.860 | Generalizes best to unseen families |

**Interpretation:** Heuristics excel in precision and clarity, ML extends coverage to novel patterns. Both combined offer a reproducible baseline for ICS intrusion detection.

## Continuous Integration (CI)

GitHub Actions workflow `.github/workflows/eip-audit.yml` performs a **light EIP audit** on each push/PR:

* verifies config files, schema fields, and matplotlib setup
* ensures dataset checksum present
* blocks merge if audit fails

Full audits can be run locally with:

```bash
python -m scripts.eip_audit --full
```

## Dataset Reference

Canadian Institute for Cybersecurity (CIC).
*Modbus 2023 Dataset.*
[https://www.unb.ca/cic/datasets/modbus-2023.html](https://www.unb.ca/cic/datasets/modbus-2023.html)

Raw PCAPs and the merged `master.csv` are excluded from the repo for size and license reasons.

## Acknowledgements

Developed as part of **INSE 6640 - Smart Grids and Control System Security**, Concordia University (2025).

All processing and evaluations follow the Evaluation Integrity Protocol (EIP) to ensure reproducibility and cross-phase consistency.

The complete final report and executive summary are available upon request.

## How to Cite

If you use this repository or its evaluation framework in academic or research work, please cite it as:

> **Baseline Anomaly Detection for ICS Modbus Traffic: Heuristics vs Machine Learning under Leave-One-Attack-Out Evaluation**, 
> *Concordia University - INSE 6640: Smart Grids and Control System Security*, 2025. 
> Available at: [https://github.com/dmtkfs/ics-modbus-anomaly-detection](https://github.com/dmtkfs/ics-modbus-anomaly-detection)
