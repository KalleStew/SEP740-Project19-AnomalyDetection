SEP740 Project 19: Anomaly Detection Using Autoencoders

1. Project Overview

This project detects anomalous network traffic using an autoencoder trained on the KDD Cup 1999 dataset.
The repository now uses shared preprocessing utilities in src/data_preprocessing.py and notebook entry points
that all resolve paths from the repository root.

2. Dataset Download Instructions

The project expects the KDD Cup 1999 files in data/raw/.

Required files:
- data/raw/kddcup.data_10_percent_corrected
- data/raw/kddcup.names
- data/raw/training_attack_types

Optional fallback file:
- data/raw/kddcup.data_10_percent

Steps:
1. Download the KDD Cup 1999 dataset from the UCI Machine Learning Repository.
2. Place kddcup.data_10_percent_corrected in data/raw/.
3. Place kddcup.names in data/raw/.
4. Place training_attack_types in data/raw/.
5. If the corrected file is unavailable, place kddcup.data_10_percent in data/raw/ and update the notebook path if needed.

3. Environment Setup

Recommended Python version: 3.11

Steps:
1. Create and activate a virtual environment.
   - macOS / Linux:
     python -m venv venv
     source venv/bin/activate
   - Windows:
     python -m venv venv
     venv\Scripts\activate
2. Install dependencies.
   pip install -r requirements.txt
3. If you are running the notebooks in VS Code, select the venv kernel before executing cells.

4. Repository Data Flow

The intended execution flow is:
1. notebooks/00_base_eda.ipynb
   - Loads the raw KDD metadata and inspects the dataset structure.
2. notebooks/01_data_processing_eda.ipynb
   - Uses src/data_preprocessing.py to clean the raw dataset.
   - Creates the processed train/test splits in data/processed/.
3. notebooks/02_BaselineModelDevelopment.ipynb
   - Loads the processed arrays and trains the baseline autoencoder.
4. notebooks/03_HyperparameterOptimization_v1.ipynb
   - Runs the first hyperparameter search workflow.
5. notebooks/03_HyperparameterOptimization_v2.ipynb
   - Runs the second hyperparameter search workflow.
6. notebooks/03_HyperparameterOptimization_v3.ipynb
   - Runs the final hyperparameter search workflow used for the report.
7. notebooks/04_Evaluation_Visualization.ipynb
   - Loads the trained model and generates the evaluation metrics and plots.

5. How to Reproduce the Results

Execute the notebooks in this order:
1. Run notebooks/00_base_eda.ipynb to confirm the raw files are available.
2. Run notebooks/01_data_processing_eda.ipynb to clean the data and export the processed arrays.
3. Run notebooks/02_BaselineModelDevelopment.ipynb to train the baseline model.
4. Run notebooks/03_HyperparameterOptimization_v1.ipynb if you want to review the first tuning pass.
5. Run notebooks/03_HyperparameterOptimization_v2.ipynb if you want to review the second tuning pass.
6. Run notebooks/03_HyperparameterOptimization_v3.ipynb to reproduce the final tuning workflow.
7. Run notebooks/04_Evaluation_Visualization.ipynb to compute the final anomaly metrics and save plots.

6. Output Locations

Generated artifacts are written to:
- data/processed/ for train/test arrays and tabular exports
- data/clean/ for cleaned CSV output from preprocessing
- models/ for saved Keras model files
- results/ for evaluation plots and metrics

7. Reproducibility Notes

1. The notebooks set NumPy and TensorFlow seeds for reproducibility.
2. The shared preprocessing module normalizes column names, handles missing values, and saves cleaned data deterministically.
3. Path resolution should go through the repository root rather than hardcoded working-directory assumptions.
4. If you change the data file location, update the DATA_PATH values in the EDA notebooks accordingly.

8. Recommended Validation

After running the notebooks, verify that:
1. data/processed/X_train.npy and data/processed/X_test.npy exist.
2. models/ contains the saved .keras autoencoder file.
3. results/ contains the evaluation plots and metrics text file.
4. The evaluation notebook reports stable accuracy, precision, recall, and F1 values when rerun with the same seed.
