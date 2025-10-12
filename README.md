# ICS Modbus Anomaly Detection

Baseline anomaly detection for ICS Modbus traffic, using both heuristics and machine learning approaches.  
This project is based on the CIC Modbus 2023 dataset and compares rule-based baselines with supervised ML models.  

## Project Tasks

- **Task A** – Data preprocessing (PCAP → CSVs, cleaning, labeling)  
- **Task B** – Heuristic detection (H1: write-rate spike, H2: rare/invalid function codes)  
- **Task C** – Supervised baselines (Logistic Regression, Random Forest, optional Isolation Forest) and LOAO evaluation across attack families  

## Repository Structure

```

src/         # core Python modules (processing, heuristics, ML, LOAO)
scripts/     # runnable entrypoints (e.g., run\_baselines.py, run\_loao.py)
configs/     # dataset config (paths, labels, mappings)
data/        # raw and processed data (gitignored, except small samples)
figures/     # auto-saved plots and results

````

## Quickstart

1. Clone the repo:
   ```bash
   git clone https://github.com/dmtkfs/ics-modbus-anomaly-detection.git
   cd ics-modbus-anomaly-detection
    ````

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows PowerShell
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

This project uses the **[CIC Modbus 2023 dataset](https://www.unb.ca/cic/datasets/modbus-2023.html)**.
Due to size, raw PCAPs and processed CSVs are not included in the repo.

* Place raw PCAPs in `data/raw/`
* Processed CSVs are generated into `data/processed/`

A small sample may later be included under `data/sample/` for quick testing.
