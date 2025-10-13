# Evaluation Integrity Protocol (EIP) — Repo Implementation

Core principle: one dataset, one metric language, one evaluation logic.

Fixed across both teams:
- Dataset pinning via SHA-256 in `configs/dataset.yaml`
- Column schema + label encoding (Attack=1, Benign=0)
- Metrics: Precision, Recall, F1; (ROC-AUC/PR-AUC for ML)
- Seed = 42; results rounded to 3 decimals
- Matplotlib rc for identical figures

Run `python scripts/eip_audit.py` before merging results.
