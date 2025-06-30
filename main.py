import typer
from app.ingestion import load_dataset
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
    typer.echo(f"✅ Dataset loaded with shape: {df.shape}")
    typer.echo(f"🎯 Target Column: {config.target_column}, Task Type: {config.task_type}")

if __name__ == "__main__":
    typer.run(run_pipeline)
