# ⚡ DataPulse AI — Automated EDA & Machine Learning Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An interactive, automated Exploratory Data Analysis (EDA) & AutoML web application built with **Python, Streamlit, Pandas, Plotly, Scikit-Learn, and XGBoost**.

---

## 🌟 Key Features

- **📊 Module 1: Dataset Overview & Quick Inspection**
  - Instant metric cards for rows, columns, missing values, duplicates, and memory usage.
  - Interactive DataFrame viewer and automatic schema/summary statistics.

- **📈 Module 2: Automated EDA & Interactive Plots**
  - **Histograms & Boxplots:** Distribution plots with categorical group coloring.
  - **Correlation Heatmap:** Interactive correlation matrix using Plotly.
  - **Categorical Analysis:** Interactive bar and pie charts for categorical features.
  - **Missing Data Audit:** Column-wise missing value percentage visualizations.

- **🧹 Module 3: Preprocessing & Data Cleaning**
  - Select columns to drop and choose pipeline imputation or row removal.
  - Categorical features are one-hot encoded inside each training pipeline.
  - Column selector to drop unnecessary ID or text columns.

- **🤖 Module 4: AutoML Model Training & Competition**
  - **Auto-Task Detection:** Treats categorical, boolean, or low-cardinality targets as classification; numeric continuous targets as regression. The task can also be selected manually.
  - **Model Competition:** Trains and compares **Logistic Regression, Decision Trees, Random Forest, and XGBoost** simultaneously.
  - **Leaderboard:** Compares classification with Accuracy, Precision, Recall, and F1-Score, or regression with R², RMSE, and MAE.
  - Numeric and categorical feature preprocessing is fitted with each model and included in the exported pipeline. Unknown categories at inference are safely ignored.

- **🔮 Module 5: Real-Time Predictions & Inference**
  - Interactive form fields for entering feature values, transformed by the same fitted preprocessing pipeline used in training.
  - Real-time probability prediction output with confidence charts when supported by the selected model.

Training requires at least five rows, at least one feature, and a target with two or more observed values. Target values cannot be missing; feature values are imputed inside the model pipeline. An unseen categorical feature value is handled by the encoder without crashing. CSV upload is limited to 50 MB, 500,000 rows, and 300 columns.

---




## 🛠️ Tech Stack & Libraries

- **Frontend & UI:** Streamlit, HTML/CSS Glassmorphism
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express, Plotly Graph Objects, Seaborn
- **Machine Learning & AutoML:** Scikit-Learn (Random Forest, Decision Tree, Logistic/Linear Regression), XGBoost

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.
