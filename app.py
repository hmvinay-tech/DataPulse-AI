import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import pickle
import io
import hashlib
from modeling import encode_target, fit_competition, infer_task_type

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="DataPulse AI — Automated EDA & AutoML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism Dark Theme Styling
st.markdown("""
<style>
    /* Dark Theme Core Settings */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Typography */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }
    
    /* Custom Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.4rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
        color: #FFFFFF;
    }

    /* Container Box */
    .glass-box {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SAMPLE DATASET GENERATORS
# ==========================================
@st.cache_data
def load_churn_dataset():
    """Generates synthetic Telecom Customer Churn dataset (Classification)."""
    np.random.seed(42)
    n = 400
    age = np.random.randint(21, 70, n)
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(25.0, 120.0, n), 2)
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 40, n), 2)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['Fiber optic', 'DSL', 'No'], n, p=[0.45, 0.40, 0.15])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n)
    paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4])
    
    # Calculate realistic probability of churn
    logit = (
        (age / 100) * 0.3 +
        (monthly_charges / 120) * 0.5 -
        (tenure / 72) * 0.8 +
        (contract == 'Month-to-month') * 0.6 +
        (internet_service == 'Fiber optic') * 0.4 - 0.5
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = np.where(prob > 0.5, 'Yes', 'No')
    
    return pd.DataFrame({
        'CustomerID': [f'CUST-{1000+i}' for i in range(n)],
        'Age': age,
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Total_Charges': np.maximum(total_charges, 0),
        'Contract_Type': contract,
        'Internet_Service': internet_service,
        'Paperless_Billing': paperless_billing,
        'Payment_Method': payment_method,
        'Churn': churn
    })

@st.cache_data
def load_housing_dataset():
    """Generates synthetic Real Estate dataset (Regression)."""
    np.random.seed(42)
    n = 400
    square_feet = np.random.randint(750, 4500, n)
    bedrooms = np.random.randint(1, 6, n)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], n)
    age_years = np.random.randint(0, 50, n)
    garage_cars = np.random.randint(0, 4, n)
    location_score = np.random.randint(1, 10, n)
    has_pool = np.random.choice(['Yes', 'No'], n, p=[0.25, 0.75])
    
    price = (
        square_feet * 180 +
        bedrooms * 15000 +
        bathrooms * 22000 -
        age_years * 1200 +
        garage_cars * 12000 +
        location_score * 25000 +
        (has_pool == 'Yes') * 35000 +
        np.random.normal(0, 25000, n)
    )
    
    return pd.DataFrame({
        'Property_ID': [f'PROP-{5000+i}' for i in range(n)],
        'Square_Feet': square_feet,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Age_Years': age_years,
        'Garage_Cars': garage_cars,
        'Location_Score': location_score,
        'Has_Pool': has_pool,
        'Price_USD': np.round(np.maximum(price, 50000), 2)
    })


# ==========================================
# HEADER & SIDEBAR NAVIGATION
# ==========================================
st.markdown('<div class="main-header">⚡ DataPulse AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Exploratory Data Analysis & AutoML Machine Learning Dashboard</div>', unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size:2.4rem;line-height:1'>⚡</div>", unsafe_allow_html=True)
st.sidebar.title("DataPulse Controls")

data_source = st.sidebar.radio(
    "Select Data Source",
    ["Upload CSV File", "Sample: Customer Churn (Classification)", "Sample: Housing Prices (Regression)"]
)

df = None
if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.size > 50 * 1024 * 1024:
                raise ValueError("CSV files are limited to 50 MB.")
            df = pd.read_csv(uploaded_file)
            if df.shape[0] > 500_000 or df.shape[1] > 300:
                raise ValueError("CSV files are limited to 500,000 rows and 300 columns.")
            if df.columns.duplicated().any():
                raise ValueError("The CSV contains duplicate column names. Rename them and upload again.")
            st.sidebar.success(f"Loaded: `{uploaded_file.name}` ({df.shape[0]} rows, {df.shape[1]} cols)")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")
    else:
        st.sidebar.info("Please upload a CSV file to get started.")
elif data_source == "Sample: Customer Churn (Classification)":
    df = load_churn_dataset()
    st.sidebar.info("Loaded Telecom Churn Dataset (400 rows, 10 columns)")
else:
    df = load_housing_dataset()
    st.sidebar.info("Loaded Real Estate Housing Dataset (400 rows, 9 columns)")

if df is None:
    st.title("Welcome to DataPulse AI")
    st.write("Explore a sample dataset to try the dashboard, or upload a CSV of your own.")
    welcome_left, welcome_right = st.columns(2)
    with welcome_left:
        st.subheader("Try a sample")
        st.write("Choose **Sample: Customer Churn** or **Sample: Housing Prices** in the sidebar to explore the app without uploading a file.")
    with welcome_right:
        st.subheader("Use your own data")
        st.write("Choose **Upload CSV File** in the sidebar, then upload a CSV. After it loads, use the sidebar navigation to inspect, clean, train, and predict.")

if df is not None:
    if df.empty or len(df.columns) == 0:
        st.error("The selected dataset is empty or has no columns. Upload a non-empty CSV to continue.")
        st.stop()
    dataset_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    if st.session_state.get("dataset_fingerprint") != dataset_fingerprint:
        for state_key in ("clean_df", "best_model", "best_model_name", "features", "task_type", "target_col", "target_le", "results_df", "model_fingerprint"):
            st.session_state.pop(state_key, None)
        st.session_state["dataset_fingerprint"] = dataset_fingerprint
    if st.session_state.get("model_fingerprint") != dataset_fingerprint:
        for state_key in ("best_model", "best_model_name", "features", "task_type", "target_col", "target_le", "results_df"):
            st.session_state.pop(state_key, None)

    app_mode = st.sidebar.selectbox(
        "Navigation Module",
        [
            "📊 1. Overview & Data Inspection",
            "📈 2. Automated EDA & Visuals",
            "🧹 3. Preprocessing & Encoding",
            "🤖 4. AutoML Model Training",
            "🔮 5. Predict & Inference Engine"
        ]
    )

    # ==========================================
    # MODULE 1: OVERVIEW & INSPECTION
    # ==========================================
    if "1." in app_mode:
        st.header("📊 Dataset Overview & Health Check")
        
        num_cols_cnt = len(df.select_dtypes(include=[np.number]).columns)
        cat_cols_cnt = len(df.select_dtypes(include=['object', 'category', 'bool']).columns)
        total_missing = df.isnull().sum().sum()
        total_dups = df.duplicated().sum()
        mem_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[1]}</div><div class="metric-label">Total Columns</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{num_cols_cnt}</div><div class="metric-label">Numeric Cols</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{cat_cols_cnt}</div><div class="metric-label">Categorical</div></div>', unsafe_allow_html=True)
        with c5:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_missing}</div><div class="metric-label">Missing Cells</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{mem_mb} MB</div><div class="metric-label">Memory Size</div></div>', unsafe_allow_html=True)
            
        st.write("")
        st.write("---")
        
        # Interactive Preview
        st.subheader("📋 Raw Data Explorer")
        preview_rows = st.slider("Select row sample size:", 5, min(100, len(df)), 10)
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("🔍 Column Schema & Types")
            schema_df = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
                "Unique Values": [df[col].nunique() for col in df.columns]
            })
            st.dataframe(schema_df, use_container_width=True)
            
        with col_right:
            st.subheader("📐 Summary Statistics (Numerical)")
            num_df = df.select_dtypes(include=[np.number])
            if not num_df.empty:
                st.dataframe(num_df.describe().T[['mean', 'std', 'min', '50%', 'max']].rename(columns={'50%': 'median'}), use_container_width=True)
            else:
                st.info("No numerical columns found in dataset.")

    # ==========================================
    # MODULE 2: AUTOMATED EDA & VISUALIZATIONS
    # ==========================================
    elif "2." in app_mode:
        st.header("📈 Automated Exploratory Data Analysis (EDA)")
        
        t_dist, t_corr, t_cat, t_scatter, t_miss = st.tabs([
            "📊 Feature Distributions",
            "🔥 Correlation Matrix",
            "🏷️ Categorical Analysis",
            "🔵 Interactive Scatter Plot",
            "⚠️ Missing Data Audit"
        ])
        
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        with t_dist:
            st.subheader("Numerical Feature Distribution Analysis")
            if num_cols:
                sel_num = st.selectbox("Select Numerical Feature:", num_cols, key="dist_num")
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(
                        df, x=sel_num, marginal="box", nbins=30,
                        title=f"Histogram & Boxplot of {sel_num}",
                        color_discrete_sequence=['#818CF8']
                    )
                    fig_hist.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    sel_group = st.selectbox("Group / Color By (Optional Categorical):", [None] + cat_cols, key="dist_cat")
                    fig_box = px.box(
                        df, y=sel_num, x=sel_group, color=sel_group,
                        title=f"{sel_num} Distribution by {sel_group if sel_group else 'Overall'}"
                    )
                    fig_box.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No numerical features found for distribution analysis.")

        with t_corr:
            st.subheader("Feature Correlation Matrix")
            if len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Correlation Heatmap (Pearson Coefficient)"
                )
                fig_corr.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("At least 2 numerical columns are required to generate a correlation matrix.")

        with t_cat:
            st.subheader("Categorical Feature Breakdown")
            if cat_cols:
                sel_cat = st.selectbox("Select Categorical Feature:", cat_cols, key="cat_sel")
                cat_counts = df[sel_cat].value_counts().reset_index()
                cat_counts.columns = [sel_cat, "Count"]
                
                col1, col2 = st.columns(2)
                with col1:
                    fig_bar = px.bar(
                        cat_counts, x=sel_cat, y="Count", color=sel_cat,
                        title=f"Frequency Count of {sel_cat}", text_auto=True
                    )
                    fig_bar.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_bar, use_container_width=True)
                with col2:
                    fig_pie = px.pie(
                        cat_counts, names=sel_cat, values="Count",
                        title=f"Proportion Share of {sel_cat}", hole=0.4
                    )
                    fig_pie.update_layout(template="plotly_dark", height=420)
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No categorical features present in dataset.")

        with t_scatter:
            st.subheader("Pairwise Feature Relationship Scatter Plot")
            if len(num_cols) >= 2:
                c1, c2, c3 = st.columns(3)
                with c1:
                    x_axis = st.selectbox("X-Axis Feature:", num_cols, index=0)
                with c2:
                    y_axis = st.selectbox("Y-Axis Feature:", num_cols, index=min(1, len(num_cols)-1))
                with c3:
                    color_feature = st.selectbox("Color Marker By:", [None] + cat_cols + num_cols)
                    
                fig_scat = px.scatter(
                    df, x=x_axis, y=y_axis, color=color_feature,
                    trendline="ols" if color_feature is None else None,
                    title=f"{y_axis} vs {x_axis}"
                )
                fig_scat.update_layout(template="plotly_dark", height=480)
                st.plotly_chart(fig_scat, use_container_width=True)
            else:
                st.info("At least 2 numerical columns required for scatter plots.")

        with t_miss:
            st.subheader("Missing Value Audit & Profiling")
            null_counts = df.isnull().sum()
            null_df = pd.DataFrame({"Column": null_counts.index, "Missing Count": null_counts.values})
            null_df["Missing %"] = np.round((null_df["Missing Count"] / len(df)) * 100, 2)
            null_df = null_df[null_df["Missing Count"] > 0]
            
            if not null_df.empty:
                fig_miss = px.bar(
                    null_df, x="Column", y="Missing Count", color="Missing %",
                    title="Missing Value Count & Percentage per Column", text_auto=True,
                    color_continuous_scale="Reds"
                )
                fig_miss.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("🎉 Complete Data Integrity! Zero missing values detected.")

    # ==========================================
    # MODULE 3: PREPROCESSING & DATA CLEANING
    # ==========================================
    elif "3." in app_mode:
        st.header("🧹 Data Preprocessing & Feature Cleaning")
        
        st.subheader("1. Feature and Missing Value Handling")
        st.info("Categorical values are kept intact here; the training pipeline encodes them consistently and handles unseen categories during prediction.")
        prior_drop = st.session_state.get("preprocess_drop_columns", [])
        cols_to_drop = st.multiselect(
            "Select irrelevant or ID columns to drop:",
            options=df.columns,
            default=[column for column in prior_drop if column in df.columns],
            help="Drop high-cardinality unique IDs or non-informative text columns before model training.",
            key="preprocess_drop_columns",
        )
        clean_df = df.drop(columns=cols_to_drop).copy()
        if clean_df.isnull().any().any():
            missing_strategy = st.radio(
                "Missing feature value handling:",
                ["Impute inside training pipeline", "Drop rows with missing values"],
                key="preprocess_missing_strategy",
            )
            if missing_strategy == "Drop rows with missing values":
                clean_df = clean_df.dropna()
        st.session_state['clean_df'] = clean_df
        st.session_state['preprocess_config'] = {
            "drop_columns": cols_to_drop,
            "missing_strategy": st.session_state.get("preprocess_missing_strategy", "Impute inside training pipeline"),
        }

        st.write("---")
        st.subheader("Preprocessed Dataset Preview")
        st.dataframe(clean_df.head(10), use_container_width=True)

        # Download Cleaned CSV
        csv_buffer = io.StringIO()
        clean_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Download Preprocessed CSV Dataset",
            data=csv_buffer.getvalue(),
            file_name="preprocessed_dataset.csv",
            mime="text/csv"
        )

    # ==========================================
    # MODULE 4: AUTOML MODEL TRAINING & COMPETITION
    # ==========================================
    elif "4." in app_mode:
        st.header("🤖 AutoML Model Competition & Benchmark")
        
        working_df = st.session_state.get('clean_df', df.copy()).copy()
        
        all_null_cols = working_df.columns[working_df.isna().all()].tolist()
        if all_null_cols:
            st.warning(f"Ignoring columns that contain only missing values: {all_null_cols}")
            working_df = working_df.drop(columns=all_null_cols)
        if working_df.empty or working_df.shape[1] < 2:
            st.error("At least one feature and one target column with data are required for training.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            target_col = st.selectbox(
                "Select Target Variable (Y to predict):",
                options=working_df.columns,
                index=len(working_df.columns) - 1
            )
        
        feature_options = [c for c in working_df.columns if c != target_col]
        with col2:
            selected_features = st.multiselect(
                "Select Feature Variables (X):",
                options=feature_options,
                default=feature_options
            )

        if not selected_features:
            st.error("Please select at least one feature column (X).")
        else:
            X = working_df[selected_features]
            y_raw = working_df[target_col]

            # Determine task type
            auto_task = infer_task_type(y_raw)
            
            task_type = st.radio(
                "Task Type Selection:",
                ["Auto-Detect", "Classification", "Regression"],
                index=0,
                horizontal=True
            )
            if task_type == "Auto-Detect":
                task_type = auto_task

            st.info(f"Active Task Mode: **{task_type}** | Target Unique Values: `{y_raw.nunique()}`")

            if y_raw.isna().any():
                st.error("The selected target contains missing values. Clean or remove those rows before training.")
                st.stop()
            if y_raw.nunique() < 2:
                st.error("The selected target must contain at least two distinct values.")
                st.stop()
            if task_type == "Regression" and not pd.api.types.is_numeric_dtype(y_raw):
                st.error("Regression requires a numeric target. Choose Classification or select a numeric target.")
                st.stop()
            if len(working_df) < 5:
                st.error("At least five rows are required for a train/test split.")
                st.stop()

            # Encode target labels while keeping feature preprocessing inside each model pipeline.
            target_le = None
            y, target_le = encode_target(y_raw, task_type)

            test_ratio = st.slider("Train / Test Split Ratio (Test %):", 0.1, 0.4, 0.2, 0.05)
            
            stratify = y if task_type == "Classification" and pd.Series(y).value_counts().min() >= 2 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_ratio, random_state=42, stratify=stratify
                )
            except ValueError as exc:
                st.error(f"Unable to create a valid train/test split: {exc}")
                st.stop()

            if st.button("🚀 Run AutoML Model Competition"):
                with st.spinner("Training models and calculating benchmark metrics..."):
                    ranking_metric = st.selectbox(
                        "Rank models by:",
                        ["F1-Score", "Accuracy", "Precision", "Recall"] if task_type == "Classification" else ["R2 Score", "RMSE", "MAE"],
                        key=f"ranking_metric_{task_type}",
                    )
                    try:
                        results_df, models = fit_competition(
                            X_train, y_train, X_test, y_test, task_type, ranking_metric
                        )
                    except Exception as exc:
                        st.error(f"Model training failed: {exc}")
                        st.stop()
                    results_df = results_df.round(4)

                    best_name = results_df.iloc[0]['Model']
                    best_model = models[best_name]

                    st.session_state['best_model_name'] = best_name
                    st.session_state['best_model'] = best_model
                    st.session_state['target_le'] = target_le if task_type == "Classification" else None
                    st.session_state['features'] = selected_features
                    st.session_state['task_type'] = task_type
                    st.session_state['target_col'] = target_col
                    st.session_state['results_df'] = results_df
                    st.session_state['model_fingerprint'] = dataset_fingerprint

                    st.subheader("🏆 AutoML Leaderboard & Benchmark")
                    st.dataframe(results_df, use_container_width=True)
                    
                    st.success(f"🥇 Best Model Winner: **{best_name}**")

                    if task_type == "Classification":
                        y_best_pred = best_model.predict(X_test)
                        cm = confusion_matrix(y_test, y_best_pred)
                        fig_cm = px.imshow(cm, text_auto=True, x=target_le.classes_, y=target_le.classes_,
                                           labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                                           title="Confusion Matrix", color_continuous_scale="Blues")
                        fig_cm.update_layout(template="plotly_dark")
                        st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # Leaderboard Comparison Chart
                    sort_metric = "Accuracy" if task_type == "Classification" else "R2 Score"
                    fig_lead = px.bar(
                        results_df, x="Model", y=sort_metric, color="Model",
                        title=f"Model Performance Ranking ({sort_metric})", text_auto=True
                    )
                    fig_lead.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_lead, use_container_width=True)

                    # Feature Importance Chart
                    st.subheader("💡 Top Feature Importances & Driver Analysis")
                    fitted_estimator = best_model.named_steps["model"]
                    transformed_names = best_model.named_steps["preprocess"].get_feature_names_out()
                    if hasattr(fitted_estimator, "feature_importances_"):
                        imp_df = pd.DataFrame({
                            "Feature": transformed_names,
                            "Importance": fitted_estimator.feature_importances_
                        }).sort_values(by="Importance", ascending=True)
                        
                        fig_imp = px.bar(
                            imp_df, y="Feature", x="Importance", orientation="h",
                            title=f"Feature Importances — {best_name}",
                            color="Importance", color_continuous_scale="Purples"
                        )
                        fig_imp.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)
                    elif hasattr(fitted_estimator, "coef_"):
                        coefs = np.abs(fitted_estimator.coef_)
                        if coefs.ndim > 1:
                            coefs = coefs.mean(axis=0)
                        imp_df = pd.DataFrame({
                            "Feature": transformed_names,
                            "Coefficient Magnitude": coefs
                        }).sort_values(by="Coefficient Magnitude", ascending=True)
                        
                        fig_imp = px.bar(
                            imp_df, y="Feature", x="Coefficient Magnitude", orientation="h",
                            title=f"Model Feature Coefficients — {best_name}",
                            color="Coefficient Magnitude", color_continuous_scale="Blues"
                        )
                        fig_imp.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig_imp, use_container_width=True)

                    # Download Trained Model (.pkl)
                    model_pkl = pickle.dumps(best_model)
                    st.download_button(
                        label=f"📦 Download Trained Model (`{best_name}.pkl`)",
                        data=model_pkl,
                        file_name=f"{best_name.replace(' ', '_').lower()}_model.pkl",
                        mime="application/octet-stream"
                    )
                    leaderboard_csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📄 Download leaderboard CSV", leaderboard_csv,
                                       file_name="model_leaderboard.csv", mime="text/csv")

    # ==========================================
    # MODULE 5: PREDICT & INFERENCE ENGINE
    # ==========================================
    elif "5." in app_mode:
        st.header("🔮 Model Real-Time Inference & Predictions")
        
        if 'best_model' not in st.session_state:
            st.warning("⚠️ No trained model found! Please complete training in **4. AutoML Model Training**.")
        else:
            best_model = st.session_state['best_model']
            model_name = st.session_state['best_model_name']
            features = st.session_state['features']
            task_type = st.session_state['task_type']
            target_col = st.session_state['target_col']
            target_le = st.session_state.get('target_le', None)
            encoders = {}

            st.info(f"Loaded Winning Model: **{model_name}** | Task: **{task_type}** | Target: `{target_col}`")
            
            st.subheader("📥 Enter Feature Input Values for Real-Time Prediction")
            
            input_dict = {}
            cols = st.columns(2)
            
            for i, feat in enumerate(features):
                col = cols[i % 2]
                feat_series = df[feat] if feat in df.columns else None
                
                if feat_series is not None and feat_series.dtype == 'object':
                    opts = list(feat_series.dropna().astype(str).unique())
                    input_dict[feat] = col.selectbox(f"Select {feat}:", opts) if opts else col.text_input(f"Enter {feat}:", "Missing")
                elif feat_series is not None and feat_series.dtype.name in ('category', 'bool'):
                    opts = list(feat_series.dropna().unique())
                    input_dict[feat] = col.selectbox(f"Select {feat}:", opts) if opts else col.text_input(f"Enter {feat}:", "Missing")
                elif feat_series is not None and np.issubdtype(feat_series.dtype, np.integer):
                    min_val = int(feat_series.min())
                    max_val = int(feat_series.max())
                    med_val = int(feat_series.median())
                    if min_val == max_val:
                        max_val += 1
                    input_dict[feat] = col.number_input(f"{feat}:", value=med_val)
                elif feat_series is not None and np.issubdtype(feat_series.dtype, np.floating):
                    min_val = float(feat_series.min())
                    max_val = float(feat_series.max())
                    med_val = float(feat_series.median())
                    if min_val == max_val:
                        max_val += 1.0
                    input_dict[feat] = col.number_input(f"{feat}:", value=med_val)
                else:
                    if feat_series is not None and feat_series.dtype in ['object', 'category', 'bool']:
                        opts = list(feat_series.astype(str).unique())
                        input_dict[feat] = col.selectbox(f"Select {feat}:", opts)
                    else:
                        input_dict[feat] = col.number_input(f"{feat}:", value=0.0)

            st.write("")
            if st.button("✨ Make Real-Time Prediction"):
                input_df = pd.DataFrame([input_dict])
                
                pred_val = best_model.predict(input_df)[0]

                st.write("---")
                if task_type == "Classification":
                    final_label = target_le.inverse_transform([pred_val])[0] if target_le is not None else pred_val
                    
                    st.success(f"🎯 **Predicted Target Outcome:** `{final_label}`")
                    
                    if hasattr(best_model, "predict_proba"):
                        probs = best_model.predict_proba(input_df)[0]
                        classes = target_le.classes_ if target_le is not None else best_model.classes_
                        
                        proba_df = pd.DataFrame({
                            "Class Label": classes,
                            "Probability": probs
                        })
                        
                        st.subheader("Confidence Probabilities Breakdown")
                        fig_prob = px.bar(
                            proba_df, x="Class Label", y="Probability", color="Class Label",
                            text_auto=".1%", title="Model Prediction Confidence"
                        )
                        fig_prob.update_layout(template="plotly_dark", height=380, yaxis_range=[0, 1])
                        st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.success(f"📈 **Predicted Numerical Value:** `{round(float(pred_val), 4)}`")
