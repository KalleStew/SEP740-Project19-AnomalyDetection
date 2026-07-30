# SEP740 Project 19: Anomaly Detection Using Autoencoders

## 1. Project Overview

This project detects anomalous network traffic (cyberattacks) using an autoencoder trained on the KDD Cup 1999 dataset. The autoencoder is trained exclusively on normal traffic; anomalies are then identified at evaluation time using a reconstruction-error threshold, i.e., traffic the model reconstructs poorly is flagged as an attack. Shared preprocessing, model, training, and evaluation utilities live in `src/`, and all notebooks resolve file paths dynamically from the repository root, so the project runs unmodified on any machine.

See also [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for a full reference of the 41 KDD Cup 1999 features.

## 2. Dataset Download Instructions

The project expects the KDD Cup 1999 files in `data/raw/` (this folder is git-ignored; you must populate it yourself).

Required files:
- `data/raw/kddcup.data_10_percent_corrected`
- `data/raw/kddcup.names`
- `data/raw/training_attack_types`

Steps:
1. Download the KDD Cup 1999 dataset from the UCI Machine Learning Repository or Keagle: <http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html> | <https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data>
2. Download `kddcup.data_10_percent.gz`, decompress it, and save it as `data/raw/kddcup.data_10_percent_corrected`.
3. Download `kddcup.names` and place it at `data/raw/kddcup.names`.
4. Download `training_attack_types` and place it at `data/raw/training_attack_types`.

## 3. Environment Setup

Recommended Python version: 3.11

Steps:
1. Create and activate a virtual environment.
   - macOS / Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
2. Install dependencies.
   ```
   pip install -r requirements.txt
   ```
3. If you are running the notebooks in VS Code or JupyterLab, select the `venv` kernel before executing cells.

## 4. Repository Pipeline

The notebooks in `notebooks/` are numbered to reflect the required execution order:

1. `00_exploratory_data_analysis.ipynb`: Loads the raw KDD metadata and inspects dataset structure, class imbalance, and feature quality.
2. `01_data_preprocessing.ipynb`: Uses `src/data_preprocessing.py` to clean the raw dataset, encode categorical features, scale numeric features, and create the processed train/test splits in `data/processed/`.
3. `02_baseline_autoencoder_model.ipynb`: Loads the processed arrays and trains the baseline autoencoder.
4. `03_hyperparameter_optimization.ipynb`: Runs three hyperparameter-search strategies (grid search, Bayesian optimization with Optuna, and an extended Optuna search that also tunes the encoder/decoder activation functions) and saves the best model found.
5. `04_evaluation_and_visualization.ipynb`: Loads all trained models and generates the final comparison metrics and plots.

The `src/` module (`data_preprocessing.py`, `train.py`, `evaluate.py`, `model.py`) provides the shared, reusable implementations behind these notebooks. `src/data_preprocessing.py` also exposes a standalone entry point that can be run directly from the command line: `python src/data_preprocessing.py`. `train.py` and `evaluate.py` are library modules consumed by the notebooks and do not expose a separate command-line interface.

## 5. How to Reproduce the Results

Execute the notebooks in this order:
1. Run `notebooks/00_exploratory_data_analysis.ipynb` to confirm the raw files are available and review the EDA.
2. Run `notebooks/01_data_preprocessing.ipynb` to clean the data and export the processed arrays.
3. Run `notebooks/02_baseline_autoencoder_model.ipynb` to train the baseline model.
4. Run `notebooks/03_hyperparameter_optimization.ipynb` to reproduce the hyperparameter search (set `RUN_MODE = False` near the top of the notebook to instead reload previously saved results).
5. Run `notebooks/04_evaluation_and_visualization.ipynb` to compute the final anomaly-detection metrics and view the comparison plots.

## 6. Output Locations

Generated artifacts are written to:
- `data/processed/`: train/test arrays (`.npy`) and tabular exports (`.csv`)
- `models/`: saved Keras model files (`.keras`)

## 7. Reproducibility Notes

1. The notebooks and scripts set Python, NumPy, and TensorFlow seeds for reproducibility (`src/train.py::set_global_seed`).
2. `src/data_preprocessing.py` normalizes column names, handles missing values, and saves cleaned data deterministically.
3. All path resolution goes through the repository root (`src/data_preprocessing.py::resolve_project_root`) rather than hardcoded, machine-specific paths.
4. If your raw data file lives elsewhere, edit the `DATA_PATH` variable in the relevant notebook accordingly.

## 8. Recommended Validation

After running the notebooks, verify that:
1. `data/processed/X_train.npy` and `data/processed/X_test.npy` exist.
2. `models/` contains the saved `.keras` autoencoder file(s).
3. The evaluation notebook reports stable accuracy, precision, recall, and F1 values when rerun with the same seed.
