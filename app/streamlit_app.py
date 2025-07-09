import streamlit as st
import pandas as pd
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.ingestion import load_dataset, analyze_dataset
from app.cleaning import clean_data
from app.eda import perform_eda
from app.features import robust_feature_engineering_pipeline
from app.feature_selection import robust_feature_selection_pipeline
from app.model_selection import model_selection_pipeline
from app.hyperparameter_tuning import hyperparameter_tuning
from app.model_evaluation import evaluate_model_on_test
from app.model_explainability import explain_with_shap
import joblib

# Move auto_configure_pipeline here
import pandas as pd
def auto_configure_pipeline(df, target_columns):
    # Task type
    target = df[target_columns[0]]
    if pd.api.types.is_numeric_dtype(target):
        task_type = "regression" if target.nunique() > 20 else "classification"
    else:
        task_type = "classification"
    # Missing value strategy
    missing_ratio = df.isna().mean().mean()
    missing_strategy = "drop" if missing_ratio < 0.05 else "mean"
    # Feature selection
    feature_selection_method = "auto"
    num_features = min(10, df.shape[1] - len(target_columns))
    # Encoding
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    max_cardinality = 20
    encoding = "onehot" if all(df[col].nunique() <= max_cardinality for col in cat_cols) else "label"
    # Scaling
    scale = "standard"
    # Correlation threshold
    correlation_threshold = 0.95
    return {
        "task_type": task_type,
        "missing_strategy": missing_strategy,
        "feature_selection_method": feature_selection_method,
        "num_features": num_features,
        "encoding": encoding,
        "scale": scale,
        "correlation_threshold": correlation_threshold,
        "max_cardinality": max_cardinality
    }

st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("🤖 AutoML Pipeline Dashboard")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    df = None
    target_columns = []
    config = None
    show_advanced = False
    train_clicked = False
    # Removed Prediction section (header, file uploader, and predict button)
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = [col.lower() for col in df.columns]
        columns = df.columns.tolist()
        target_columns = st.multiselect("Select target column(s)", columns)
        target_columns = [col.lower() for col in target_columns]
        if target_columns:
            config = auto_configure_pipeline(df, target_columns)
            show_advanced = st.checkbox("Show Advanced Options", value=False)
            if show_advanced:
                task_type = st.selectbox("Task Type", ["auto", "classification", "regression"], index=["auto", "classification", "regression"].index(config["task_type"]))
                missing_strategy = st.selectbox("Missing Value Strategy", ["drop", "mean", "mode"], index=["drop", "mean", "mode"].index(config["missing_strategy"]))
                feature_selection_method = st.selectbox("Feature Selection Method", ["auto", "variance", "rfe", "mutual_info", "univariate"], index=["auto", "variance", "rfe", "mutual_info", "univariate"].index(config["feature_selection_method"]))
                num_features = st.slider("Number of Features to Select", 1, min(50, len(df.columns)), config["num_features"])
                encoding = st.selectbox("Categorical Encoding", ["onehot", "label"], index=["onehot", "label"].index(config["encoding"]))
                scale = st.selectbox("Scaling Method", ["standard", "minmax", "none"], index=["standard", "minmax", "none"].index(config["scale"]))
                correlation_threshold = st.slider("Correlation Threshold", 0.7, 1.0, config["correlation_threshold"])
                max_cardinality = st.slider("Max Categorical Cardinality", 2, 100, config["max_cardinality"])
            else:
                task_type = config["task_type"]
                missing_strategy = config["missing_strategy"]
                feature_selection_method = config["feature_selection_method"]
                num_features = config["num_features"]
                encoding = config["encoding"]
                scale = config["scale"]
                correlation_threshold = config["correlation_threshold"]
                max_cardinality = config["max_cardinality"]
            train_clicked = st.button("🚀 Train Model")
            st.markdown("---")
            # Removed Prediction section (header, file uploader, and predict button)

# --- MAIN AREA ---
if uploaded_file and target_columns and train_clicked:
    st.success(f"Loaded dataset with shape: {df.shape}")

    with st.expander("1️⃣ Data Preview", expanded=True):
        st.dataframe(df.head())

    with st.expander("2️⃣ Exploratory Data Analysis (EDA)"):
        eda_report = perform_eda(df, target_columns)
        st.json(eda_report)
        import json
        st.download_button("Download EDA Report (JSON)", json.dumps(eda_report, default=str), "eda_report.json")

    with st.expander("3️⃣ Data Cleaning"):
        cleaned_df = clean_data(df, missing_strategy=missing_strategy)
        st.dataframe(cleaned_df.head())

    with st.expander("4️⃣ Feature Engineering"):
        engineered_df = robust_feature_engineering_pipeline(
            cleaned_df,
            target_column=target_columns,
            encoding=encoding,
            scale=scale,
            drop_corr=True,
            correlation_threshold=correlation_threshold,
            max_cardinality=max_cardinality
        )
        st.dataframe(engineered_df.head())

    with st.expander("5️⃣ Feature Selection"):
        selected_df = robust_feature_selection_pipeline(
            engineered_df,
            target_column=target_columns,
            method=feature_selection_method,
            k=num_features,
            task_type=task_type if task_type != "auto" else "classification",
            variance_threshold=0.01
        )
        st.dataframe(selected_df.head())
        st.write("Shape of selected_df:", selected_df.shape)
        st.write("NaN count per column:", selected_df.isna().sum())
        st.write("Unique values in target:", selected_df[target_columns].nunique())
        st.write("Describe:", selected_df.describe())

    with st.expander("6️⃣ Model Training & Evaluation"):
        valid = True
        msg = ""
        def validate_dataset_for_ml(df, target_columns, min_rows=10, min_features=1):
            df = df.dropna(axis=1, how='all')
            df = df.dropna(axis=0, how='all')
            nunique = df.nunique()
            constant_cols = nunique[nunique <= 1].index.tolist()
            df = df.drop(columns=constant_cols)
            for target in target_columns:
                if target not in df.columns:
                    return False, f"Target column '{target}' not found after cleaning.", df
                if df[target].isna().all():
                    return False, f"Target column '{target}' is all NaN.", df
                if df[target].nunique() <= 1:
                    return False, f"Target column '{target}' has only one unique value.", df
            feature_cols = [col for col in df.columns if col not in target_columns]
            if df.shape[0] < min_rows:
                return False, f"Not enough rows after cleaning (found {df.shape[0]}, need at least {min_rows}).", df
            if len(feature_cols) < min_features:
                return False, f"Not enough features after cleaning (found {len(feature_cols)}, need at least {min_features}).", df
            return True, "", df
        valid, msg, selected_df = validate_dataset_for_ml(selected_df, target_columns, min_rows=10, min_features=1)
        if not valid:
            st.error(f"Dataset is not suitable for ML: {msg}")
        else:
            try:
                model_result = model_selection_pipeline(
                    selected_df,
                    target_column=target_columns,
                    task_type=task_type if task_type != "auto" else "classification"
                )
                st.subheader("Best Model")
                st.write(model_result["model"])
                st.subheader("Model Metrics")
                st.json(model_result["metrics"])

                tuning_result = hyperparameter_tuning(
                    selected_df,
                    target_column=target_columns,
                    model_name=model_result["model"],
                    task_type=task_type if task_type != "auto" else "classification",
                    method="random"
                )
                st.subheader("Tuned Model Parameters")
                st.json(tuning_result["best_params"])
                st.write(f"Cross-validated Score: {tuning_result['score']}")

                test_metrics = evaluate_model_on_test(
                    selected_df,
                    target_column=target_columns,
                    model_path="models/tuned_model.pkl",
                    task_type=task_type if task_type != "auto" else "classification",
                    output_dir="outputs"
                )
                st.subheader("Test Metrics")
                st.json(test_metrics)

                model = joblib.load("models/tuned_model.pkl")
                import numpy as np
                X_test = selected_df.drop(columns=target_columns).astype(float)
                explain_with_shap(model, X_test, output_dir="outputs")
                shap_summary_path = os.path.join("outputs", "shap_summary.png")
                if os.path.exists(shap_summary_path):
                    st.subheader("SHAP Summary Plot")
                    st.image(shap_summary_path)
            except Exception as e:
                st.error(f"Model training or evaluation failed: {e}")

    # --- DOWNLOAD BUTTONS ---
    st.header("Download Results")
    # Save outputs to temp files for download
    cleaned_csv = cleaned_df.to_csv(index=False).encode('utf-8') if 'cleaned_df' in locals() else None
    engineered_csv = engineered_df.to_csv(index=False).encode('utf-8') if 'engineered_df' in locals() else None
    selected_csv = selected_df.to_csv(index=False).encode('utf-8') if 'selected_df' in locals() else None
    if cleaned_csv:
        st.download_button("Download Cleaned Data", cleaned_csv, "cleaned_data.csv")
    if engineered_csv:
        st.download_button("Download Engineered Data", engineered_csv, "engineered_data.csv")
    if selected_csv:
        st.download_button("Download Selected Features", selected_csv, "selected_features.csv")
    if os.path.exists("models/tuned_model.pkl"):
        with open("models/tuned_model.pkl", "rb") as f:
            st.download_button("Download Tuned Model", f, "tuned_model.pkl")
    if os.path.exists("outputs/shap_summary.png"):
        with open("outputs/shap_summary.png", "rb") as f:
            st.download_button("Download SHAP Summary Plot", f, "shap_summary.png")

# --- PREDICTION AREA ---
# Removed prediction area: all code that runs when prediction_file and predict_clicked 