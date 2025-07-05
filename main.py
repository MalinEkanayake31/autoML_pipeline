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
                                   help="Target column name (optional - will show options if not provided)"),
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
        # Ensure directories exist
        ensure_directories()

        # Step 1: Load dataset
        typer.echo("🔄 Loading dataset...")
        df = load_dataset(file_path)
        typer.echo(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # Show dataset info if requested or if no target provided
        if show_info or target is None:
            show_dataset_info(df)
            if target is None:
                typer.echo("\n❓ Please specify a target column using --target <column_name>")
                return

        # Validate target column
        is_valid, suggestion = validate_target_column(df, target)
        if not is_valid:
            typer.echo(f"❌ Error: Target column '{target}' not found in dataset")
            typer.echo(f"💡 {suggestion}")
            show_dataset_info(df)
            return

        # Auto-detect task type if not specified
        if task == "auto":
            task = detect_task_type(df, target)
            typer.echo(f"🔍 Auto-detected task type: {task}")

        # Validate configuration
        config = Config(file_path=file_path, target_column=target, task_type=task)

        # Step 2: Analyze dataset
        typer.echo("\n🔍 Analyzing dataset...")
        analysis = analyze_dataset(df)
        typer.echo(f"📊 Column Types: {analysis['column_types']}")
        typer.echo(f"🧼 Missing Values: {sum(analysis['missing_and_duplicates']['missing_values'].values())} total")
        typer.echo(f"📛 Duplicates: {analysis['missing_and_duplicates']['num_duplicates']}")

        # Step 3: EDA
        typer.echo("\n🔍 Running EDA...")
        eda_report = perform_eda(df, target)
        if 'error' not in eda_report['class_balance']:
            typer.echo(f"🧮 Class Balance: {eda_report['class_balance']['class_distribution']}")
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
        if target not in cleaned_df.columns:
            typer.echo(f"❌ Error: Target column '{target}' was removed during cleaning")
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
            target_column=target,
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
        available_features = engineered_df.shape[1] - 1  # Subtract 1 for target column
        max_features = min(num_features, available_features, 50)  # Cap at 50 features

        if max_features != num_features:
            typer.echo(
                f"⚠️ Adjusting feature count from {num_features} to {max_features} (available: {available_features})")

        try:
            selected_df = robust_feature_selection_pipeline(
                engineered_df,
                target_column=target,
                method=feature_selection_method,
                k=max_features,
                task_type=task,
                variance_threshold=0.01
            )

            typer.echo(f"✅ Feature selection completed. Shape: {original_shape} → {selected_df.shape}")
            selected_features = [col for col in selected_df.columns if col != target]
            typer.echo(f"🧪 Selected features ({len(selected_features)}): {selected_features}")

            selected_df.to_csv("outputs/selected_features.csv", index=False)
            typer.echo("📁 Selected features saved to outputs/selected_features.csv")

        except Exception as e:
            typer.echo(f"⚠️ Feature selection failed: {str(e)}")
            typer.echo("📝 Using engineered features instead...")
            selected_df = engineered_df.copy()

        # Step 7: Model Selection
        typer.echo("\n🏁 Running Model Selection...")
        typer.echo(f"📊 Final data shape: {selected_df.shape}")

        # Final data validation
        if selected_df[target].isna().any():
            typer.echo(f"🧹 Removing {selected_df[target].isna().sum()} rows with NaN in target")
            selected_df = selected_df.dropna(subset=[target])

        # Check for any remaining NaN values in features
        if selected_df.drop(columns=[target]).isna().any().any():
            typer.echo("🧹 Filling remaining NaN values in features with median/mode")
            for col in selected_df.columns:
                if col != target and selected_df[col].isna().any():
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
        typer.echo(f"📊 Final features: {selected_df.shape[1] - 1}")
        typer.echo(f"📊 Final samples: {selected_df.shape[0]}")

        if task == "classification":
            class_counts = selected_df[target].value_counts()
            typer.echo(f"📊 Class distribution: {class_counts.to_dict()}")

            # Check for class imbalance
            min_class_count = class_counts.min()
            if min_class_count < 2:
                typer.echo("⚠️ Warning: Some classes have very few samples. Consider collecting more data.")
        else:
            typer.echo(f"📊 Target range: {selected_df[target].min():.2f} to {selected_df[target].max():.2f}")

        # Run model selection
        model_result = model_selection_pipeline(selected_df, target_column=target, task_type=task)
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
        target_column=target,
        model_name=model_result["model"],
        task_type=task,
        method="random"  # or "grid"
    )

    typer.echo(f"🎯 Tuned Model: {tuning_result['model']}")
    typer.echo(f"✅ Best Params: {tuning_result['best_params']}")
    typer.echo(f"📊 Cross-validated Score: {tuning_result['score']}")
    typer.echo("📁 Tuned model saved to models/tuned_model.pkl")


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