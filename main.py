import typer
import pandas as pd
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


@app.command()
def run_pipeline(
        file_path: str = typer.Option(..., "--file-path", help="Path to the dataset CSV file"),
        target: str = typer.Option(..., "--target", help="Target column name"),
        task: str = typer.Option(..., "--task", help="Task type: classification or regression"),
        missing_strategy: str = typer.Option("drop", "--missing-strategy", help="Strategy for missing values"),
        feature_selection_method: str = typer.Option("variance", "--feature-selection",
                                                     help="Feature selection method"),
        num_features: int = typer.Option(10, "--num-features", help="Number of features to select")
):
    """
    Run the complete AutoML pipeline
    """
    try:
        # Ensure directories exist
        ensure_directories()

        # Validate configuration
        config = Config(file_path=file_path, target_column=target, task_type=task)

        # Step 1: Load dataset
        typer.echo("🔄 Loading dataset...")
        df = load_dataset(file_path)
        typer.echo(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

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
        cleaned_df = clean_data(df, missing_strategy=missing_strategy)
        typer.echo(f"🧹 Cleaned Data Shape: {cleaned_df.shape}")
        save_cleaned_data(cleaned_df)
        typer.echo("📁 Cleaned data saved to outputs/cleaned_data.csv")

        # FIXED: Step 5: Feature Engineering - Separate target before processing
        typer.echo("\n🛠️ Running Feature Engineering...")

        # Check if target column exists
        if target not in cleaned_df.columns:
            typer.echo(f"❌ Error: Target column '{target}' not found in cleaned data")
            typer.echo(f"Available columns: {cleaned_df.columns.tolist()}")
            return

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
            selected_X = feature_selection_pipeline(
                X_for_selection,
                y_for_selection,
                method=feature_selection_method,
                k=min(num_features, X_for_selection.shape[1]),  # Don't select more features than available
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
        typer.echo(f"📊 Available columns: {selected_df.columns.tolist()}")

        # Verify target column exists in final dataset
        if target not in selected_df.columns:
            typer.echo(f"❌ Error: Target column '{target}' missing from final dataset")
            return

        # Check for any remaining issues
        final_X = selected_df.drop(columns=[target])
        final_y = selected_df[target]

        typer.echo(f"📊 Final X shape: {final_X.shape}")
        typer.echo(f"📊 Final y shape: {final_y.shape}")
        typer.echo(f"📊 Final y sample values: {final_y.head().tolist()}")

        # Check for any NaN values that might cause issues
        if final_X.isna().any().any():
            typer.echo("⚠️ Warning: NaN values found in features")
            typer.echo(f"NaN counts: {final_X.isna().sum().sum()}")

        if final_y.isna().any():
            typer.echo("⚠️ Warning: NaN values found in target")
            typer.echo(f"NaN count in target: {final_y.isna().sum()}")

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


if __name__ == "__main__":
    app()