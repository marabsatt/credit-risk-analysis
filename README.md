# Credit Default Risk Classification

This repository contains a Jupyter notebook (`main.ipynb`) implementing a complete workflow to predict credit default risk of clients using the UCI Credit Card Default dataset. The notebook covers data loading, exploratory analysis, preprocessing, feature engineering, model training and hyperparameter tuning, model evaluation and interpretation.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Features](#features)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Notebook Structure](#notebook-structure)
8. [Methodology](#methodology)
9. [Modeling & Evaluation](#modeling--evaluation)
10. [Feature Importance & Interpretation](#feature-importance--interpretation)
11. [Next Steps](#next-steps)
12. [License](#license)

---

## Project Overview

This project aims to build a robust classification model that predicts whether a credit card client will default on their payment next month. It demonstrates best practices in:

* Handling imbalanced binary classification
* Advanced feature engineering on payment and bill history
* Building and tuning multiple machine learning models
* Evaluating with precision, recall, F₁-score, and ROC-AUC
* Interpreting model predictions using SHAP values

## Dataset

The data is sourced from the UCI Machine Learning Repository:

* **Title**: Default of Credit Card Clients Dataset
* **Instances**: 30,000
* **Attributes**: 24 (including `LIMIT_BAL`, demographic features, past payment history, bill and payment amounts, and the target `default.payment.next.month`)
* **Link**: [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

## Features

Key predictors include:

* **Demographics**: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
* **Payment History**: `PAY_0`–`PAY_6` (repayment status for the past six months)
* **Bill Amounts**: `BILL_AMT1`–`BILL_AMT6`
* **Payment Amounts**: `PAY_AMT1`–`PAY_AMT6`

Additional engineered features:

* Credit utilization ratios (`bill_amt / limit_bal`)
* Total of delinquent months (`PAY_n` > 0)
* Trend and ratio features across bill/payment series

## Requirements

* Python 3.8+
* Jupyter Notebook
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* shap

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/marabsatt/credit-risk-analysis.git
   cd credit-risk-analysis
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter**:

   ```bash
   jupyter notebook
   ```

## Usage

1. Open `main.ipynb` in Jupyter Notebook.

2. Run cells sequentially to:

   * Load and inspect the dataset
   * Perform EDA and visualize class imbalance
   * Preprocess and engineer features
   * Train baseline and ensemble classifiers
   * Tune hyperparameters with cross-validation
   * Evaluate on a hold-out test set
   * Interpret results with SHAP plots

3. Modify parameters (e.g., hyperparameter grid, resampling strategies) as needed for experimentation.

## Notebook Structure

1. **Data Loading & Overview**: Read CSV, display summary stats, check for missing values.
2. **Exploratory Data Analysis (EDA)**: Class distributions, feature distributions, correlation heatmaps.
3. **Preprocessing Pipeline**:

   * handle outliers
   * Scale numeric features containing outliers (`RobustScaler`)
   * Assemble with `ColumnTransformer` and `Pipeline`.

4. **Feature Engineering**: Utilization ratios, delinquency counts.
5. **Modeling**:

   * Baseline models: Support Vector Classifier, Random Forest Classifier, Logistic Regression, K-Nearest Neighbors, and Gradient Boosting Classifier
   * Random Forest, XGBoost/LightGBM with `RandomizedSearchCV` for model optimization
   * Stratified K‑Fold cross-validation

6. **Evaluation**:

   * Classification report (precision, recall, F₁)
   * ROC and Precision-Recall curves
   * Calibration and threshold selection

7. **Interpretation**:

   * SHAP global and local explanations
   * Feature importance ranking.

8. **Conclusion & Next Steps**: Summary of findings and deployment considerations.

## Methodology

* **Data Split**: 80% training, 20% hold-out test, stratified on the target.
* **Imbalance Handling**: Class weighting on training data.
* **Hyperparameter Tuning**: Randomized search over model grids optimizing ROC‑AUC.
* **Final Validation**: Evaluate chosen model on test set and report metrics.

## Modeling & Evaluation

| Metric    | Definition                          | Why It Matters                           |
| --------- | ----------------------------------- | ---------------------------------------- |
| Precision | TP / (TP + FP)                      | Avoid false alarms on non-defaulters     |
| Recall    | TP / (TP + FN)                      | Catch as many defaulters as possible     |
| F₁-Score  | Harmonic mean of precision & recall | Balances FP vs. FN costs                 |
| ROC-AUC   | Area under ROC curve                | Discrimination ability across thresholds |

## Feature Importance & Interpretation

* **SHAP Summary Plot**: Ranks features by mean absolute SHAP value.
* **SHAP Dependence Plots**: Visualize how changes in a feature impact default probability.

## Next Steps

* **Deploy** model as a REST API (e.g., FastAPI) for real-time scoring.
* **Monitor** performance drift and recalibrate on new data quarterly.
* **Extend** with additional data sources (demographics, behavioral records).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
