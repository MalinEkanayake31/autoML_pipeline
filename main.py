import typer
from app.ingestion import load_dataset, analyze_dataset
from app.config import Config

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

if __name__ == "__main__":
    typer.run(run_pipeline)
