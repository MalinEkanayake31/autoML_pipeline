import pandas as pd
from typing import List, Dict

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

    return {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime
    }

def detect_missing_and_duplicates(df: pd.DataFrame) -> Dict:
    return {
        "missing_values": df.isnull().sum().to_dict(),
        "num_duplicates": df.duplicated().sum(),
    }

def summarize_dataset(df: pd.DataFrame) -> Dict:
    profile = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "column_types": detect_column_types(df),
        "missing_and_duplicates": detect_missing_and_duplicates(df),
    }
    return profile
