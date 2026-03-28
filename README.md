# Breast Cancer Prediction

A machine learning model that predicts whether a breast tumor is **benign** or **malignant** based on cell nucleus measurements from fine needle aspirate (FNA) biopsies. Built with XGBoost and served through an interactive Streamlit dashboard.

## Results

- **Accuracy:** 97.4%
- **ROC-AUC:** 0.992
- **Recall (Malignant):** 95.2%
- **Precision (Malignant):** 97.6%

Trained on the [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) (569 samples, 21 features after redundancy reduction).

## Project Structure

```
├── preprocessing.ipynb    # Exploratory data analysis notebook
├── preprocess.py          # Preprocessing pipeline (cleaning, encoding, scaling)
├── train.py               # XGBoost training, evaluation, and artifact saving
├── app.py                 # Streamlit dashboard for interactive predictions
├── model/
│   ├── model.json         # Trained XGBoost model
│   ├── scaler.pkl         # Fitted StandardScaler
│   └── feature_names.pkl  # Feature name ordering
└── TrainingPortion.csv    # Source dataset
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install dependencies
uv sync

# Train the model
uv run python train.py

# Launch the dashboard
uv run streamlit run app.py
```

## Feature Engineering

The original 30 features were reduced to 21 by dropping highly correlated pairs (r > 0.9):

- **Perimeter & Area** (mean, SE, worst) — redundant with Radius
- **Concave Points** (mean, worst) — redundant with Concavity
- **Texture Mean** — redundant with Texture Worst

This simplifies the input form without sacrificing model performance.
