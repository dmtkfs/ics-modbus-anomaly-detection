# Appendix — ML Final Run (EIP Reproduction)

This appendix documents the exact CLI used to reproduce the final EIP-compliant ML pipeline (audit → baselines → balanced calibration → LOAO → aggregation → light audit). Outputs are timestamped dynamically; there are no hardcoded run folders.

## How to run
```powershell
pwsh -File scripts/run_final_ml.ps1
# or to skip the simple LOAO step:
pwsh -File scripts/run_final_ml.ps1 -IncludeSimpleLOAO:$false
````

## Outputs

* Results: `results/ml/final_<YYYYMMDD_HHMMSS>/`
* Figures: `figures/ml/final_<YYYYMMDD_HHMMSS>/`

These include:

* `metrics.csv` (baselines)
* `calib_metrics.csv` (balanced calibrator)
* `loao_metrics.csv` (balanced LOAO)
* `aggregate_phase3_summary.csv`
* PR/ROC/Threshold-F1 plots, LOAO bars, histograms, TP/FN overlays

All runs must pass the EIP audit both before and after execution.