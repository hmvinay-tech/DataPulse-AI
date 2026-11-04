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

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/hmvinay-tech/DataPulse-AI.git
cd DataPulse-AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository. Include `app.py`, `modeling.py`, `requirements.txt`, and `.streamlit/config.toml`.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select your repository, branch, and `app.py` as the main file, then deploy.
4. Wait for dependency installation and the app health check. The public app URL will be shown in the Streamlit dashboard.

The repository does not need secrets for the current features. Uploaded CSVs and trained models live only in the running app session; the app does not provide permanent storage, user accounts, or a database. The 50 MB upload cap and row/column caps are also enforced by the application. For larger workloads or private datasets, deploy behind an authenticated service and add durable storage before sharing publicly.

## 🧪 Development Checks

```bash
python -m pytest -q
python -m streamlit run app.py
```

---

## 🛠️ Tech Stack & Libraries

- **Frontend & UI:** Streamlit, HTML/CSS Glassmorphism
- **Data Manipulation:** Pandas, NumPy
- **Visualizations:** Plotly Express, Plotly Graph Objects, Seaborn
- **Machine Learning & AutoML:** Scikit-Learn (Random Forest, Decision Tree, Logistic/Linear Regression), XGBoost

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details.
