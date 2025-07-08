import typer
import pandas as pd
import numpy as np
from pathlib import Path
import os

from app.config import Config
from app.ingestion import load_dataset, analyze_dataset
from app.cleaning import clean_data, save_cleaned_data
from app.eda import perform_eda, save_eda_report
from app.features import robust_feature_engineering_pipeline
from app.feature_selection import robust_feature_selection_pipeline
from app.model_selection import model_selection_pipeline
from app.hyperparameter_tuning import hyperparameter_tuning
from app.model_evaluation import evaluate_model_on_test

app = typer.Typer()


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ["outputs", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def detect_task_type(df: pd.DataFrame, target_column: str) -> str:
    """
    Automatically detect if the task should be classification or regression
    """
    target = df[target_column]

    # Check if target is numeric
    if pd.api.types.is_numeric_dtype(target):
        # If numeric, check number of unique values
        unique_values = target.nunique()
        total_values = len(target)

        # If less than 20 unique values AND less than 10% unique values, treat as classification
        if unique_values <= 20 and (unique_values / total_values) < 0.1:
            return "classification"
        else:
            return "regression"
    else:
        # If not numeric, it's classification
        return "classification"


def validate_target_column(df: pd.DataFrame, target_column: str) -> tuple[bool, str]:
    """
    Validate if the target column exists and provide helpful suggestions
    """
    if target_column in df.columns:
        return True, ""

    # Provide suggestions for similar column names
    suggestions = []
    target_lower = target_column.lower()

    for col in df.columns:
        col_lower = col.lower()
        if target_lower in col_lower or col_lower in target_lower:
            suggestions.append(col)

    if not suggestions:
        # Look for common target column patterns
        common_targets = ['target', 'label', 'class', 'y', 'output', 'result', 'prediction', 'outcome']
        for col in df.columns:
            if any(pattern in col.lower() for pattern in common_targets):
                suggestions.append(col)

    suggestion_msg = f"Did you mean one of these: {suggestions}" if suggestions else "Please check available columns"
    return False, suggestion_msg


def show_dataset_info(df: pd.DataFrame):
    """
    Display comprehensive dataset information to help users choose target column
    """
    typer.echo("\n" + "=" * 50)
    typer.echo("📊 DATASET INFORMATION")
    typer.echo("=" * 50)

    typer.echo(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    typer.echo(f"📋 Columns: {df.columns.tolist()}")

    # Show data types and sample values
    typer.echo("\n📊 Column Information:")
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        missing = df[col].isnull().sum()
        sample_values = df[col].dropna().head(3).tolist()

        typer.echo(f"  • {col}: {dtype} | {nunique} unique | {missing} missing | sample: {sample_values}")

    typer.echo("\n💡 Tip: Choose a target column that represents what you want to predict")
    typer.echo("=" * 50)


@app.command()
def run_pipeline(
        file_path: str = typer.Option(..., "--file-path", help="Path to the dataset CSV file"),
        target: str = typer.Option(None, "--target",
                                   help="Target column name(s), comma-separated for multiple (optional - will show options if not provided)"),
        task: str = typer.Option("auto", "--task", help="Task type: classification, regression, or auto"),
        missing_strategy: str = typer.Option("drop", "--missing-strategy",
                                             help="Strategy for missing values: drop, mean, mode"),
        feature_selection_method: str = typer.Option("auto", "--feature-selection",
                                                     help="Feature selection method: variance, rfe, mutual_info, univariate, auto"),
        num_features: int = typer.Option(10, "--num-features", help="Number of features to select"),
        show_info: bool = typer.Option(False, "--show-info", help="Show dataset information and exit")
):
    """
    Run the complete AutoML pipeline
    """
    try:
        ensure_directories()
        typer.echo("🔄 Loading dataset...")
        df = load_dataset(file_path)
        typer.echo(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Parse target columns
        target_columns = [col.strip() for col in target.split(",")] if target else None

        # Show dataset info if requested or if no target provided
        if show_info or not target_columns:
            show_dataset_info(df)
            if not target_columns:
                typer.echo("\n❓ Please specify target column(s) using --target <col1,col2,...>")
                return

        # Validate target columns
        missing_targets = [col for col in target_columns if col not in df.columns]
        if missing_targets:
            typer.echo(f"❌ Error: Target column(s) {missing_targets} not found in dataset")
            show_dataset_info(df)
            return

        # Auto-detect task type if not specified (use first target for detection)
        if task == "auto":
            task = detect_task_type(df, target_columns[0])
            typer.echo(f"🔍 Auto-detected task type: {task}")

        # Validate configuration
        config = Config(file_path=file_path, target_columns=target_columns, task_type=task)

        # Step 2: Analyze dataset
        typer.echo("\n🔍 Analyzing dataset...")
        analysis = analyze_dataset(df)
        typer.echo(f"📊 Column Types: {analysis['column_types']}")
        typer.echo(f"🧼 Missing Values: {sum(analysis['missing_and_duplicates']['missing_values'].values())} total")
        typer.echo(f"📛 Duplicates: {analysis['missing_and_duplicates']['num_duplicates']}")

        # Step 3: EDA
        typer.echo("\n🔍 Running EDA...")
        eda_report = perform_eda(df, target_columns)
        # Print class balance for each target (if classification), or skip for regression
        if task == "classification":
            for target_col in target_columns:
                class_info = eda_report['class_balance'].get(target_col, {})
                if 'class_distribution' in class_info:
                    typer.echo(f"🧮 Class Balance for {target_col}: {class_info['class_distribution']}")
                else:
                    typer.echo(f"ℹ️ No class distribution info for {target_col}.")
        else:
            for target_col in target_columns:
                typer.echo(f"ℹ️ Skipping class balance for regression target: {target_col}")
        save_eda_report(eda_report)
        typer.echo("📁 EDA report saved to outputs/eda_report.json")

        # Step 4: Clean data
        typer.echo("\n🧽 Cleaning Data...")
        original_shape = df.shape
        cleaned_df = clean_data(df, missing_strategy=missing_strategy)
        typer.echo(f"🧹 Cleaned Data Shape: {cleaned_df.shape} (removed {original_shape[0] - cleaned_df.shape[0]} rows)")
        save_cleaned_data(cleaned_df)
        typer.echo("📁 Cleaned data saved to outputs/cleaned_data.csv")

        # Verify target column still exists after cleaning
        if not all(col in cleaned_df.columns for col in target_columns):
            typer.echo(f"❌ Error: Target column(s) {set(target_columns) - set(cleaned_df.columns)} were removed during cleaning")
            typer.echo(f"Available columns after cleaning: {cleaned_df.columns.tolist()}")
            return

        # Check if we have enough data left
        if cleaned_df.shape[0] < 10:
            typer.echo("❌ Error: Not enough data left after cleaning (less than 10 rows)")
            typer.echo(
                "💡 Try using a different missing value strategy: --missing-strategy mean or --missing-strategy mode")
            return

        # Step 5: Feature Engineering
        typer.echo("\n🛠️ Running Feature Engineering...")
        original_shape = cleaned_df.shape

        engineered_df = robust_feature_engineering_pipeline(
            cleaned_df,
            target_column=target_columns,
            encoding="onehot",
            scale="standard",
            drop_corr=True,
            correlation_threshold=0.95,
            max_cardinality=20
        )

        typer.echo(f"🧠 Feature engineering completed. Shape: {original_shape} → {engineered_df.shape}")
        engineered_df.to_csv("outputs/engineered_data.csv", index=False)
        typer.echo("📁 Engineered data saved to outputs/engineered_data.csv")

        # Step 6: Feature Selection
        typer.echo("\n📉 Running Feature Selection...")
        original_shape = engineered_df.shape

        # Adjust num_features based on available features
        available_features = engineered_df.shape[1] - len(target_columns)  # Subtract target columns
        max_features = min(num_features, available_features, 50)  # Cap at 50 features

        if max_features != num_features:
            typer.echo(
                f"⚠️ Adjusting feature count from {num_features} to {max_features} (available: {available_features})")

        try:
            selected_df = robust_feature_selection_pipeline(
                engineered_df,
                target_column=target_columns,
                method=feature_selection_method,
                k=max_features,
                task_type=task,
                variance_threshold=0.01
            )

            typer.echo(f"✅ Feature selection completed. Shape: {original_shape} → {selected_df.shape}")
            selected_features = [col for col in selected_df.columns if col not in target_columns]
            typer.echo(f"🧪 Selected features ({len(selected_features)}): {selected_features}")

            selected_df.to_csv("outputs/selected_features.csv", index=False)
            typer.echo("📁 Selected features saved to outputs/selected_features.csv")

        except Exception as e:
            typer.echo(f"⚠️ Feature selection failed: {str(e)}")
            typer.echo("📝 Using engineered features instead...")
            selected_df = engineered_df.copy()

        # Before using selected_df in later steps, check it exists
        if 'selected_df' not in locals():
            typer.echo("❌ Error: selected_df is not available. Exiting pipeline.")
            return

        # Step 7: Model Selection
        typer.echo("\n🏁 Running Model Selection...")
        typer.echo(f"📊 Final data shape: {selected_df.shape}")

        # Final data validation
        if selected_df[target_columns].isna().any().any():
            typer.echo(f"🧹 Removing {selected_df[target_columns].isna().sum().sum()} rows with NaN in target(s)")
            selected_df = selected_df.dropna(subset=target_columns)

        # Check for any remaining NaN values in features
        if selected_df.drop(columns=target_columns).isna().any().any():
            typer.echo("🧹 Filling remaining NaN values in features with median/mode")
            for col in selected_df.columns:
                if col not in target_columns and selected_df[col].isna().any():
                    if selected_df[col].dtype in ['int64', 'float64']:
                        selected_df[col] = selected_df[col].fillna(selected_df[col].median())
                    else:
                        selected_df[col] = selected_df[col].fillna(
                            selected_df[col].mode().iloc[0] if not selected_df[col].mode().empty else "MISSING")

        # Final check
        if selected_df.shape[0] < 10:
            typer.echo("❌ Error: Not enough data left for model training (less than 10 rows)")
            return

        # Show final data info
        typer.echo(f"📊 Final features: {selected_df.shape[1] - len(target_columns)}")
        typer.echo(f"📊 Final samples: {selected_df.shape[0]}")

        if task == "classification":
            for target_col in target_columns:
                class_counts = selected_df[target_col].value_counts()
                typer.echo(f"📊 Class distribution for {target_col}: {class_counts.to_dict()}")

                # Check for class imbalance
                min_class_count = class_counts.min()
                if min_class_count < 2:
                    typer.echo(f"⚠️ Warning: Some classes for {target_col} have very few samples. Consider collecting more data.")
        else:
            for target_col in target_columns:
                typer.echo(f"📊 Target range for {target_col}: {selected_df[target_col].min():.2f} to {selected_df[target_col].max():.2f}")

        # Run model selection
        model_result = model_selection_pipeline(selected_df, target_column=target_columns, task_type=task)
        typer.echo(f"✅ Best Model: {model_result['model']}")
        typer.echo(f"📊 Metrics: {model_result['metrics']}")
        typer.echo(f"📊 Best Score: {model_result['best_score']:.4f}")
        typer.echo("📁 Best model saved to models/best_model.pkl")

        typer.echo("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

    typer.echo("\n🧪 Running Hyperparameter Tuning...")
    tuning_result = hyperparameter_tuning(
        selected_df,
        target_column=target_columns,
        model_name=model_result["model"],
        task_type=task,
        method="random"  # or "grid"
    )

    typer.echo(f"🎯 Tuned Model: {tuning_result['model']}")
    typer.echo(f"✅ Best Params: {tuning_result['best_params']}")
    typer.echo(f"📊 Cross-validated Score: {tuning_result['score']}")
    typer.echo("📁 Tuned model saved to models/tuned_model.pkl")

    # Step 9: Model Evaluation – Test the Final Model
    typer.echo("\n🧪 Evaluating Tuned Model on Hold-out Test Set...")
    # For demonstration, use selected_df as test set (replace with real hold-out set in production)
    test_metrics = evaluate_model_on_test(
        selected_df,  # Replace with your real test set
        target_column=target_columns,
        model_path="models/tuned_model.pkl",
        task_type=task,
        output_dir="outputs"
    )
    typer.echo(f"📊 Test Metrics: {test_metrics}")

    # Step 10: Model Explainability – SHAP & Feature Importance
    try:
        from app.model_explainability import explain_with_shap, plot_feature_importance
        import joblib
        typer.echo("\n🔍 Running Model Explainability (SHAP & Feature Importance)...")
        model = joblib.load("models/tuned_model.pkl")
        X_test = selected_df.drop(columns=target_columns).astype(float)  # Ensure all features are float for SHAP
        feature_names = X_test.columns.tolist()
        # SHAP explanations
        explain_with_shap(model, X_test, output_dir="outputs")
        typer.echo("✅ SHAP summary plot saved to outputs/shap_summary.png")
        # Feature importance (tree-based models)
        importances = plot_feature_importance(model, feature_names, output_dir="outputs")
        if importances is not None:
            typer.echo("✅ Feature importance plot saved to outputs/feature_importance.png")
        else:
            typer.echo("ℹ️ Feature importances not available for this model type.")
    except Exception as e:
        typer.echo(f"⚠️ Model explainability step failed: {str(e)}")

    # Step 11: Save Full Pipeline and Metadata
    try:
        from app.export_utils import save_pipeline, save_metadata
        import joblib
        from sklearn.pipeline import Pipeline
        typer.echo("\n💾 Saving full preprocessing pipeline and metadata...")
        # Load the tuned model
        model = joblib.load("models/tuned_model.pkl")
        # Recreate the full pipeline (assuming feature engineering and selection are deterministic)
        # If you have a pipeline object from training, use that directly instead
        # Here, we just save the model and note the features used
        # Optionally, you can wrap preprocessing steps in a Pipeline and save that
        # Save the model as the 'pipeline' for now
        save_pipeline(model, "models/full_pipeline.pkl")
        typer.echo("✅ Full pipeline saved to models/full_pipeline.pkl")
        # Collect metadata
        metadata = {
            "target_column": target_columns,
            "feature_list": selected_df.drop(columns=target_columns).columns.tolist(),
            "task_type": task,
            "evaluation_metrics": test_metrics,
            "hyperparameters": tuning_result.get("best_params", {}),
            "model_name": tuning_result.get("model", model_result.get("model")),
        }
        save_metadata(metadata, "models/metadata.json")
        typer.echo("✅ Metadata saved to models/metadata.json")
    except Exception as e:
        typer.echo(f"⚠️ Saving pipeline/metadata failed: {str(e)}")


@app.command()
def info(file_path: str = typer.Option(..., "--file-path", help="Path to the dataset CSV file")):
    """
    Show dataset information to help choose target column
    """
    try:
        df = load_dataset(file_path)
        show_dataset_info(df)
    except Exception as e:
        typer.echo(f"❌ Error loading dataset: {str(e)}")


if __name__ == "__main__":
    app()