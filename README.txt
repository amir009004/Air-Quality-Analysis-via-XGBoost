# Air Quality Analysis & Time Series Forecasting via XGBoost

A machine learning time-series forecasting pipeline built to predict hourly Carbon Monoxide CO(GT)concentrations using the **UCI Air Quality Dataset**. This project covers end-to-end data preprocessing, cyclical temporal feature engineering, lag variable construction, and gradient-boosted decision tree modeling.

---

##  Project Overview

Urban air quality monitoring generates high-frequency multivariate sensor data prone to missing entries and nonlinear daily patterns (such as rush-hour traffic spikes and weekend shifts). This pipeline transforms raw environmental sensor readings into a structured time-series dataset and evaluates predictive performance using **XGBoost** alongside a baseline **Ridge Regression** model.

### Key Highlights
* **Target Variable:** Hourly Carbon Monoxide concentration (`CO(GT)` in $\text{mg/m}^3$).
* **Dataset:** [UCI Machine Learning Repository – Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality) (9,357 hourly instances recorded between March 2004 and April 2005).
* **Validation Strategy:** Strict **80/20 chronological train-test split** to prevent temporal data leakage.

---

##  Data Preprocessing & Feature Engineering

The raw dataset (`AirQualityUCI.csv`) was cleaned and transformed into `AirQualityUCI_timeseries_clean.csv` through the following pipeline:

1. **Parsing & Formatting:** Resolved European semicolon delimiters (`;`) and decimal commas (`,`), and removed trailing empty artifact columns.
2. **Missing Value Handling:** Replaced sensor error sentinel values (`-200`) with `NaN` and generated binary missingness indicator flags (`*_was_missing`) to preserve sensor outage context.
3. **Chronological Indexing:** Unified `Date` and `Time` strings into a sorted `timestamp` index.
4. **Temporal Features:** Extracted cyclical and calendar context:
   * `hour` (0–23)
   * `day_of_week` (0–6)
   * `month` (1–12)
   * `is_weekend` (binary flag)
5. **Autoregressive Lag Features:** Engineered short-term historical predictors (`CO(GT)_lag_1`, `CO(GT)_lag_2`, `CO(GT)_lag_3`) capturing concentrations from the previous 1 to 3 hours.

---

## Modeling & Results

Models were trained on the first 80% of the timeline and evaluated on the unseen final 20% (January 2005 – April 2005).

| Model | Features Used | RMSE | $R^2$ Score |
| :--- | :--- | :---: | :---: |
| **Ridge Regression (Baseline)** | Temporal + 3-Hour Lags | ~0.63 | ~0.79 |
| **XGBoost Regressor** | Temporal + 3-Hour Lags | **~0.60** | **~0.81** |

*Gradient boosting effectively captures nonlinear interactions between rush-hour timing and short-term pollution momentum, achieving an $R^2$ above 0.81 on unseen test data.*

---
## Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib
---

## Repository Structure

```text
Air_Quality_Analysis_via_XGBoost/
├── AirQualityAnalysis.py                 # Main preprocessing, training, and visualization script
├── AirQualityUCI_timeseries_clean.csv    # Cleaned & feature-engineered dataset
├── .gitignore                            # Excludes virtual environments and IDE files
└── README.md                             # Project documentation