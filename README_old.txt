# SEP740 Project 19: Anomaly Detection Using Autoencoders

## Project Overview
This repository contains the codebase for SEP740 Project 19. The objective is to develop an anomaly detection system using autoencoder neural networks trained on the KDD Cup 1999 dataset to identify anomalous network traffic (cyberattacks).

## Documentation
The following documents support Phase 1 collaboration and reproducibility:

- [CONTRIBUTING](CONTRIBUTING.md) - team workflow, branching, commit, and review guidance.
- [Data Dictionary](data/DATA_DICTIONARY.md) - feature reference for the KDD Cup 1999 dataset.
- [Phase 1 Notes](docs/PHASE1_NOTES.md) - scope, assumptions, and open items for the first project phase.
- [EDA Notebook](notebooks/01_eda.ipynb) - exploratory analysis for the raw KDD99 data.

## Prerequisites & Setup
1. Clone this repository to your local machine.
2. Ensure you have Python 3.9+ installed.
3. Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
4. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## Dataset Instructions
1. The repository now keeps the metadata files used by the EDA workflow in `data/raw/`: `kddcup.names` for column initialization and `training_attack_types` for attack-family grouping.
2. The repository also keeps `kddcup.data_10_percent_corrected`, which is the default raw file used by [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).
3. Other raw exports can still be stored locally under `data/raw/`, but they remain ignored by default.
4. If you are using a different raw export, update the notebook path in [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) accordingly.

## How to Run the Code
To replicate the results outlined in our final report, execute the scripts in the following order:

For exploratory analysis, start with [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb). The notebook now:

- loads column names from `data/raw/kddcup.names`
- groups attack labels with `data/raw/training_attack_types`
- analyzes class imbalance, attack families, zero-variance features, skewness, outliers, and categorical cardinality

1. **Preprocess the data:**
   `python src/data_preprocessing.py`
   *(Cleans, normalizes, and splits the data into `data/processed/`)*

2. **Train the Autoencoder:**
   `python src/train.py`
   *(Trains the baseline model and saves weights to `models/`)*

3. **Evaluate the Model:**
   `python src/evaluate.py`
   *(Generates Precision, Recall, and F1 scores, and saves reconstruction error plots to `results/`)*

---

## Group Git Workflow Guide

To prevent code conflicts and lost work, our team follows a **Feature Branch Workflow**. Please adhere to these steps when contributing to the project:

### 1. Never work directly on the `main` branch.
The `main` branch should only contain working, tested code. 

### 2. Create a branch for your task.
Before starting new work, pull the latest changes and create a branch:
    ```
    git checkout main
    git pull origin main
    git checkout -b feature/your-feature-name
    ```
*(Use prefixes like `feature/`, `bugfix/`, or `docs/` for clarity).*

### 3. Commit your changes regularly.
Write clear, descriptive commit messages:
    ```
    git add .
    git commit -m "Add normalization logic to data_preprocessing.py"
    ```

### 4. Push and create a Pull Request (PR).
When your code is ready, push your branch to GitHub:
    ```
    git push origin feature/your-feature-name
    ```
Go to GitHub, open a Pull Request against the `main` branch, and tag at least one team member to review your code. Once approved, it will be merged into `main`.