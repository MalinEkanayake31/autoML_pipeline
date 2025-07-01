import typer
from app.ingestion import load_dataset, analyze_dataset
from app.config import Config
from app.eda import perform_eda, save_eda_report
from app.cleaning import clean_data, save_cleaned_data
from app.features import feature_engineering_pipeline


def run_pipeline(
    file_path: str = typer.Option(..., "--file-path", "-f", help="Path to your dataset CSV"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="Task type: classification / regression / clustering"),
    missing_strategy: str = typer.Option("drop", help="Missing value handling: drop, mean, mode")
):
    """
    Run the AutoML pipeline from user input.
    """
    config = Config(file_path=file_path, target_column=target, task_type=task)
    df = load_dataset(config.file_path)

    typer.echo(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    summary = analyze_dataset(df)
    typer.echo("📊 Column Types:")
    typer.echo(summary["column_types"])
    typer.echo(f"🧼 Missing Values:\n{summary['missing_and_duplicates']['missing_values']}")
    typer.echo(f"📛 Duplicates: {summary['missing_and_duplicates']['num_duplicates']}")

    typer.echo("\n🔍 Running EDA...")
    eda_results = perform_eda(df, target)
    typer.echo(f"🧮 Class Balance: {eda_results['class_balance']}")
    save_eda_report(eda_results)

    typer.echo("\n🧽 Cleaning Data...")
    cleaned_df = clean_data(df, missing_strategy=missing_strategy)
    save_cleaned_data(cleaned_df)
    typer.echo(f"🧹 Cleaned Data Shape: {cleaned_df.shape}")
    typer.echo("📁 Cleaned data saved to outputs/cleaned_data.csv")

    typer.echo("\n🛠️ Running Feature Engineering...")
    final_df = feature_engineering_pipeline(cleaned_df, encoding="onehot", scale="standard", drop_corr=False)

    output_path = "outputs/processed_data.csv"
    final_df.to_csv(output_path, index=False)
    typer.echo(f"🧠 Final processed data saved to {output_path}")

if __name__ == "__main__":
    typer.run(run_pipeline)
