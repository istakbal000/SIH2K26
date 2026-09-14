# Forest and Industrial Fire ML Models

This repository contains the Machine Learning backend for detecting and classifying Forest and Industrial fires using satellite data (NASA FIRMS), OpenStreetMap, and environmental/weather data.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in any required credentials.
4. Place real FIRMS CSV data or other raw data sources into `data/raw/`.

## Workflow

To run the full pipeline:

1. **Prepare Data & Feature Engineering**:
   ```bash
   python scripts/prepare_data.py
   ```
   *Note: If real data is missing from `data/raw/`, this script will output "REAL DATA REQUIRED" but will still run the pipeline on any available data to generate the required outputs.*

2. **Train Forest Fire Model**:
   ```bash
   python scripts/train_forest.py
   ```

3. **Train Industrial Fire Model**:
   ```bash
   python scripts/train_industrial.py
   ```

4. **Evaluate Models**:
   ```bash
   python scripts/evaluate_models.py
   ```

5. **Run Inference**:
   ```bash
   python -m src.inference
   ```

## Final Report Details
(To be generated after pipeline execution)
1. Data sources detected:
2. Number of records:
3. Features generated:
4. Forest training samples:
5. Industrial training samples:
6. Train/validation/test methodology:
7. Best forest model:
8. Best industrial model:
9. Precision / Recall / F1 / ROC-AUC / PR-AUC:
10. Major limitations:
11. Commands to reproduce training: (See above)
