import pandas as pd
from .schema import summarize_dataset

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV and return it
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Return a profiling summary of the dataset
    """
    summary = summarize_dataset(df)
    return summary
