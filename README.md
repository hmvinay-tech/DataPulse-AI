# ⚡ DataPulse AI

**Explore CSV datasets, build interactive analyses, and compare machine-learning models in one Streamlit dashboard.**

[![Open the live app](https://img.shields.io/badge/Launch-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://datapulse-ai-eppme63zjrertzykkes57f.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)

## Overview

DataPulse AI is an interactive dashboard for exploring tabular data and trying common machine-learning workflows. Upload a CSV or choose a generated sample dataset, inspect and visualize the data, prepare features, train several models, and make predictions with the winning model.

**[Open DataPulse AI](https://datapulse-ai-eppme63zjrertzykkes57f.streamlit.app/)**

## Features

- **Dataset overview:** Review dataset dimensions, column types, missing values, duplicates, summary statistics, and a data preview.
- **Exploratory analysis:** Create distribution charts, boxplots, correlation heatmaps, categorical summaries, scatter plots, and missing-value reports.
- **Data preparation:** Select features and target, handle missing values, and encode categorical data as part of model pipelines.
- **Classification and regression:** Automatically infer the task from the target, or choose the task manually.
- **Model comparison:** Train and compare Logistic Regression, Decision Trees, Random Forest, and XGBoost models.
- **Predictions and exports:** Use the winning model for interactive predictions and download the trained model and leaderboard.
- **Sample data:** Try customer churn classification or housing-price regression without uploading a dataset.

## Run locally

Use Python 3.12 to match the deployed app.

~~~bash
git clone https://github.com/hmvinay-tech/DataPulse-AI.git
cd DataPulse-AI
python -m venv .venv
~~~

Activate the environment, then install dependencies and run the app.

**Windows PowerShell**
~~~powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
~~~

**macOS / Linux**
~~~bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
~~~

Open the local address printed by Streamlit, usually [http://localhost:8501](http://localhost:8501).

## Deploy

The live app runs on [Streamlit Community Cloud](https://share.streamlit.io/) from the **main** branch, using the root-level **app.py** entrypoint and Python 3.12. Commits to the connected GitHub branch are picked up by Streamlit and redeployed automatically.

To create another deployment, connect GitHub in Community Cloud, choose **Create app**, and select:

- **Repository:** hmvinay-tech/DataPulse-AI
- **Branch:** main
- **Main file path:** app.py
- **Python version:** 3.12

Dependencies are listed in **requirements.txt**. The app does not currently require secrets or external services.

## Data and limitations

The app is designed for interactive exploration, not persistent storage. Uploaded CSVs and trained models are used during the running app session; this project does not provide user accounts, a database, or durable file storage. The app enforces limits of 50 MB per CSV, 500,000 rows, and 300 columns. Avoid uploading confidential or sensitive datasets to the publicly accessible demo.

Model training requires at least five rows, at least one usable feature, and a target with at least two observed values. Target values cannot be missing.

## Project files

~~~text
.
├── app.py             # Streamlit interface and workflow
├── modeling.py        # Preprocessing, model training, and metrics
├── requirements.txt   # Python dependencies
└── data AI/           # Original uploaded project copy
~~~

The deployed app uses the root-level files.

## Built with

Python · Streamlit · Pandas · NumPy · Plotly · scikit-learn · XGBoost
