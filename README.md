# ML Model Training — Asthma Classification

Train, evaluate, and tune classifiers to predict asthma diagnosis from patient features.

## What this lab covers

- Problem framing: binary classification (`diagnosis`)
- Baselines: **Logistic Regression**, **Random Forest**, **Gradient Boosting**
- Metrics focused on medicine: Accuracy, Precision, Recall, **F1**, ROC-AUC
- Stratified train/test split and threshold tuning
- Feature importance analysis
- Hyperparameter search with **RandomizedSearchCV**
- Class imbalance handling (`class_weight` / SMOTE discussion)
- Saving the best model to `models/` and logging metrics to PostgreSQL

## Stack

`Python` · `pandas` · `scikit-learn` · `imbalanced-learn` (optional) · `joblib` · `SQLAlchemy` · `PostgreSQL`
