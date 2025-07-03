import typer
import pandas as pd
import numpy as np
from pathlib import Path
import os

from app.config import Config
from app.ingestion import load_dataset, analyze_dataset
from app.cleaning import clean_data, save_cleaned_data
from app.eda import perform_eda, save_eda_report
from app.features import feature_engineering_pipeline
from app.feature_selection import feature_selection_pipeline
from app.model_selection import model_selection_pipeline

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

        # If less than 10 unique values OR less than 5% unique values, treat as classification
        if unique_values <= 10 or (unique_values / total_values) < 0.05:
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
        common_targets = ['target', 'label', 'class', 'y', 'output', 'result', 'prediction']
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
        feature_selection_method: str = typer.Option("variance", "--feature-selection",
                                                     help="Feature selection method: variance, rfe, mutual_info"),
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
        typer.echo(f"📊 Column Types:\n{analysis['column_types']}")
        typer.echo(f"🧼 Missing Values:\n{analysis['missing_and_duplicates']['missing_values']}")
        typer.echo(f"📛 Duplicates: {analysis['missing_and_duplicates']['num_duplicates']}")

        # Step 3: EDA
        typer.echo("\n🔍 Running EDA...")
        eda_report = perform_eda(df, target)
        typer.echo(f"🧮 Class Balance: {eda_report['class_balance']}")
        save_eda_report(eda_report)

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

        # Separate features and target BEFORE feature engineering
        X_clean = cleaned_df.drop(columns=[target])
        y_clean = cleaned_df[target].copy()

        typer.echo(f"📊 Features shape before engineering: {X_clean.shape}")
        typer.echo(f"📊 Target shape: {y_clean.shape}")

        # Apply feature engineering only to features
        X_processed = feature_engineering_pipeline(X_clean)

        # Combine processed features with original target
        processed_df = X_processed.copy()
        processed_df[target] = y_clean

        processed_df.to_csv("outputs/processed_data.csv", index=False)
        typer.echo(f"🧠 Feature engineering completed. Shape: {processed_df.shape}")
        typer.echo("📁 Processed data saved to outputs/processed_data.csv")

        # Step 6: Feature Selection
        typer.echo("\n📉 Running Feature Selection...")

        # Split features and target for feature selection
        X_for_selection = processed_df.drop(columns=[target])
        y_for_selection = processed_df[target]

        typer.echo(f"📊 Features for selection: {X_for_selection.shape}")
        typer.echo(f"📊 Target for selection: {y_for_selection.shape}")

        # Apply feature selection
        try:
            # Ensure we don't select more features than available
            max_features = min(num_features, X_for_selection.shape[1])
            if max_features != num_features:
                typer.echo(f"⚠️ Adjusting feature count from {num_features} to {max_features} (max available)")

            selected_X = feature_selection_pipeline(
                X_for_selection,
                y_for_selection,
                method=feature_selection_method,
                k=max_features,
                task_type=task
            )

            # Combine selected features with target
            selected_df = selected_X.copy()
            selected_df[target] = y_for_selection

            selected_df.to_csv("outputs/selected_features.csv", index=False)
            typer.echo(f"✅ Feature selection completed. Shape: {selected_df.shape}")
            typer.echo(f"🧪 Selected features: {selected_X.columns.tolist()}")
            typer.echo("📁 Selected features saved to outputs/selected_features.csv")

        except Exception as e:
            typer.echo(f"⚠️ Feature selection failed: {str(e)}")
            typer.echo("📝 Using all processed features instead...")
            selected_df = processed_df.copy()

        # Step 7: Model Selection
        typer.echo("\n🏁 Running Model Selection...")
        typer.echo(f"📊 Final data shape: {selected_df.shape}")
        typer.echo(f"📊 Target column: {target}")

        # Final validation before model training
        final_X = selected_df.drop(columns=[target])
        final_y = selected_df[target]

        typer.echo(f"📊 Final X shape: {final_X.shape}")
        typer.echo(f"📊 Final y shape: {final_y.shape}")

        # Check for any remaining issues
        if final_X.isna().any().any():
            typer.echo("⚠️ Warning: NaN values found in features")
            nan_counts = final_X.isna().sum()
            typer.echo(f"NaN counts: {nan_counts[nan_counts > 0].to_dict()}")

        if final_y.isna().any():
            typer.echo("⚠️ Warning: NaN values found in target")
            typer.echo(f"NaN count in target: {final_y.isna().sum()}")

        # Show target distribution
        if task == "classification":
            typer.echo(f"📊 Target distribution: {final_y.value_counts().to_dict()}")
        else:
            typer.echo(f"📊 Target stats: min={final_y.min():.2f}, max={final_y.max():.2f}, mean={final_y.mean():.2f}")

        # Run model selection
        model_result = model_selection_pipeline(selected_df, target_column=target, task_type=task)
        typer.echo(f"✅ Best Model: {model_result['model']}")
        typer.echo(f"📊 Metrics: {model_result['metrics']}")
        typer.echo("📁 Best model saved to models/best_model.pkl")

        typer.echo("\n🎉 Pipeline completed successfully!")

    except Exception as e:
        typer.echo(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


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