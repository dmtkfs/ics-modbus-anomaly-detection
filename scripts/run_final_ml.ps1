# Reproducible end-to-end ML pipeline under EIP
# Runs: audit → baselines → balanced calibration → LOAO (simple + balanced) → aggregate → light audit

param(
  [switch]$IncludeSimpleLOAO = $true  # set to $false to skip step 4
)

$ErrorActionPreference = "Stop"

# Dynamic, non-hardcoded run stamp
$run = Get-Date -Format "yyyyMMdd_HHmmss"
$OUT = "results/ml/final_$run"
$FIG = "figures/ml/final_$run"
New-Item -ItemType Directory -Force -Path $OUT, $FIG | Out-Null

Write-Host "== EIP full audit + smoke test =="
python -m scripts.eip_audit --full
python -m scripts.smoke_dataset

Write-Host "== Baselines (LR, RF, IF) =="
python -m scripts.run_baselines

# Scoop fresh baseline outputs into this run's folders
Get-ChildItem results\* -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-30) } | `
  Copy-Item -Destination $OUT -Force
Get-ChildItem figures\ml\* -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-30) } | `
  Copy-Item -Destination $FIG -Force

Write-Host "== Balanced calibrator (PRIMARY) =="
python -m scripts.run_calibration_balanced `
  --out-prefix "$OUT/calib" `
  --fig-prefix "$FIG/calib" `
  --rf-trees 200 --rf-trees-per-pass 50 --rf-max-samples 1000000 `
  --rf-n-jobs 1 `
  --target-recall 0.90 `
  --min-precision 0.90 `
  --max-fpr 0.02 `
  --include-best-f1

if ($IncludeSimpleLOAO) {
  Write-Host "== LOAO (simple) =="
  python -m scripts.run_loao

  # Scoop simple LOAO outputs into this run's folders
  Get-ChildItem results\* -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) } | `
    Copy-Item -Destination $OUT -Force
  Get-ChildItem figures\ml\* -File | Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-10) } | `
    Copy-Item -Destination $FIG -Force
} else {
  Write-Host "== LOAO (simple) skipped =="
}

Write-Host "== LOAO (balanced) =="
python -m scripts.run_loao_ml_balanced `
  --out-prefix "$OUT/loao" `
  --fig-prefix "$FIG/loao" `
  --rf-trees 200 --rf-trees-per-pass 50 --rf-max-samples 1000000 `
  --rf-n-jobs 1 `
  --if-trees 200 --if-trees-per-pass 50 --if-benign-cap 1000000 `
  --target-recall 0.90 `
  --min-precision 0.90 `
  --max-fpr 0.02 `
  --write-hist --write-tpfn

Write-Host "== Aggregate JUST this run =="
python -m scripts.aggregate_phase3_metrics `
  --in "$OUT" `
  --out "$OUT/aggregate_phase3_summary.csv"

Write-Host "== Light audit (CI-style) =="
$env:EIP_LIGHT_AUDIT = "1"
python -m scripts.eip_audit
$env:EIP_LIGHT_AUDIT = $null

Write-Host "`nDone. Artifacts under:`n  $OUT`n  $FIG"
