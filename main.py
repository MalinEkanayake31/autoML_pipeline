import typer
from app.ingestion import load_dataset, analyze_dataset
from app.config import Config
from app.eda import perform_eda, save_eda_report


def run_pipeline(
    file_path: str = typer.Option(..., "--file-path", "-f", help="Path to your dataset CSV"),
    target: str = typer.Option(..., "--target", "-t", help="Target column name"),
    task: str = typer.Option("classification", "--task", help="Task type: classification / regression / clustering")
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
    typer.echo("📁 EDA report saved to outputs/eda_report.json")

if __name__ == "__main__":
    typer.run(run_pipeline)
