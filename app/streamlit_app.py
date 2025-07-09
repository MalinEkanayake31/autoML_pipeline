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

# --- 1. Data Upload ---
st.header("1. Upload Your Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded dataset with shape: {df.shape}")
    st.dataframe(df.head())

    # --- 2. Target Selection ---
    st.header("2. Select Target Column(s)")
    columns = df.columns.tolist()
    target_columns = st.multiselect("Select target column(s)", columns)

    if target_columns:
        # --- 3. Pipeline Configuration ---
        st.header("3. Configure Pipeline")
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

        def validate_dataset_for_ml(df, target_columns, min_rows=10, min_features=1):
            # Drop columns with all NaN
            df = df.dropna(axis=1, how='all')
            # Drop rows with all NaN
            df = df.dropna(axis=0, how='all')
            # Remove constant columns
            nunique = df.nunique()
            constant_cols = nunique[nunique <= 1].index.tolist()
            df = df.drop(columns=constant_cols)
            # Check target
            for target in target_columns:
                if target not in df.columns:
                    return False, f"Target column '{target}' not found after cleaning.", df
                if df[target].isna().all():
                    return False, f"Target column '{target}' is all NaN.", df
                if df[target].nunique() <= 1:
                    return False, f"Target column '{target}' has only one unique value.", df
            # Check shape
            feature_cols = [col for col in df.columns if col not in target_columns]
            if df.shape[0] < min_rows:
                return False, f"Not enough rows after cleaning (found {df.shape[0]}, need at least {min_rows}).", df
            if len(feature_cols) < min_features:
                return False, f"Not enough features after cleaning (found {len(feature_cols)}, need at least {min_features}).", df
            return True, "", df

        def run_automl_pipeline(df, target_columns, task_type, missing_strategy, feature_selection_method, num_features, encoding, scale, correlation_threshold, max_cardinality):
            # Step 1: Analyze dataset
            analysis = analyze_dataset(df)
            with st.expander("Show Dataset Analysis (details)", expanded=False):
                st.json(analysis)
                import json
                st.download_button("Download Dataset Analysis (JSON)", json.dumps(analysis, default=str), "dataset_analysis.json")

            # Step 2: EDA
            eda_report = perform_eda(df, target_columns)
            with st.expander("Show EDA Report (details)", expanded=False):
                st.json(eda_report)
                st.download_button("Download EDA Report (JSON)", json.dumps(eda_report, default=str), "eda_report.json")

            # Step 3: Clean data
            cleaned_df = clean_data(df, missing_strategy=missing_strategy)
            st.subheader("Cleaned Data Preview")
            st.dataframe(cleaned_df.head())

            # Step 4: Feature Engineering
            engineered_df = robust_feature_engineering_pipeline(
                cleaned_df,
                target_column=target_columns,
                encoding=encoding,
                scale=scale,
                drop_corr=True,
                correlation_threshold=correlation_threshold,
                max_cardinality=max_cardinality
            )
            st.subheader("Engineered Data Preview")
            st.dataframe(engineered_df.head())

            # Step 5: Feature Selection
            selected_df = robust_feature_selection_pipeline(
                engineered_df,
                target_column=target_columns,
                method=feature_selection_method,
                k=num_features,
                task_type=task_type if task_type != "auto" else "classification",
                variance_threshold=0.01
            )
            st.subheader("Selected Features Preview")
            st.dataframe(selected_df.head())
            # Debug output
            st.write("Shape of selected_df:", selected_df.shape)
            st.write("NaN count per column:", selected_df.isna().sum())
            st.write("Unique values in target:", selected_df[target_columns].nunique())
            st.write("Describe:", selected_df.describe())

            # Validate dataset before model training
            valid, msg, selected_df = validate_dataset_for_ml(selected_df, target_columns, min_rows=10, min_features=1)
            if not valid:
                st.error(f"Dataset is not suitable for ML: {msg}")
                return

            # Check for all-zero or NaN columns
            feature_cols = [col for col in selected_df.columns if col not in target_columns]
            all_zero_cols = [col for col in feature_cols if (selected_df[col] == 0).all()]
            nan_cols = [col for col in feature_cols if selected_df[col].isna().all()]
            if all_zero_cols or nan_cols:
                st.warning(f"Warning: The following selected features are all zero or NaN: {all_zero_cols + nan_cols}. Consider increasing the number of features or changing the feature selection method.")

            # Step 6: Model Selection with fallback
            try:
                model_result = model_selection_pipeline(
                    selected_df,
                    target_column=target_columns,
                    task_type=task_type if task_type != "auto" else "classification"
                )
            except Exception as e:
                st.warning(f"Model training failed with selected features: {e}. Retrying with more features...")
                # Try with more features (up to 20 or all available)
                max_features = min(20, engineered_df.shape[1] - len(target_columns))
                if max_features > num_features:
                    selected_df = robust_feature_selection_pipeline(
                        engineered_df,
                        target_column=target_columns,
                        method=feature_selection_method,
                        k=max_features,
                        task_type=task_type if task_type != "auto" else "classification",
                        variance_threshold=0.01
                    )
                    st.info(f"Retrying model training with {max_features} features.")
                    st.dataframe(selected_df.head())
                    try:
                        model_result = model_selection_pipeline(
                            selected_df,
                            target_column=target_columns,
                            task_type=task_type if task_type != "auto" else "classification"
                        )
                    except Exception as e2:
                        st.warning(f"Model training still failed: {e2}. Using all engineered features as fallback.")
                        selected_df = engineered_df.copy()
                        st.info("Retrying model training with all engineered features.")
                        st.dataframe(selected_df.head())
                        model_result = model_selection_pipeline(
                            selected_df,
                            target_column=target_columns,
                            task_type=task_type if task_type != "auto" else "classification"
                        )
                else:
                    st.error("Model training failed even after fallback attempts. Please check your data and configuration.")
                    return

            st.subheader("Best Model")
            st.write(model_result["model"])
            st.subheader("Model Metrics")
            st.json(model_result["metrics"])

            # Step 7: Hyperparameter Tuning
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

            # Step 8: Model Evaluation
            test_metrics = evaluate_model_on_test(
                selected_df,
                target_column=target_columns,
                model_path="models/tuned_model.pkl",
                task_type=task_type if task_type != "auto" else "classification",
                output_dir="outputs"
            )
            st.subheader("Test Metrics")
            st.json(test_metrics)

            # Step 9: Model Explainability
            try:
                model = joblib.load("models/tuned_model.pkl")
                X_test = selected_df.drop(columns=target_columns).astype(float)
                explain_with_shap(model, X_test, output_dir="outputs")
                shap_summary_path = os.path.join("outputs", "shap_summary.png")
                if os.path.exists(shap_summary_path):
                    st.subheader("SHAP Summary Plot")
                    st.image(shap_summary_path)
            except Exception as e:
                st.warning(f"Model explainability failed: {e}")

            # --- 10. Download Buttons ---
            st.header("Download Results")
            # Save outputs to temp files for download
            cleaned_csv = cleaned_df.to_csv(index=False).encode('utf-8')
            engineered_csv = engineered_df.to_csv(index=False).encode('utf-8')
            selected_csv = selected_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned Data", cleaned_csv, "cleaned_data.csv")
            st.download_button("Download Engineered Data", engineered_csv, "engineered_data.csv")
            st.download_button("Download Selected Features", selected_csv, "selected_features.csv")
            if os.path.exists("models/tuned_model.pkl"):
                with open("models/tuned_model.pkl", "rb") as f:
                    st.download_button("Download Tuned Model", f, "tuned_model.pkl")
            if os.path.exists("outputs/shap_summary.png"):
                with open("outputs/shap_summary.png", "rb") as f:
                    st.download_button("Download SHAP Summary Plot", f, "shap_summary.png")

        # --- 4. Run Pipeline Button ---
        if st.button("🚀 Run AutoML Pipeline"):
            with st.spinner("Running pipeline..."):
                run_automl_pipeline(
                    df,
                    target_columns,
                    task_type,
                    missing_strategy,
                    feature_selection_method,
                    num_features,
                    encoding,
                    scale,
                    correlation_threshold,
                    max_cardinality
                ) 