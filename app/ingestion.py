import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load dataset: {str(e)}")
