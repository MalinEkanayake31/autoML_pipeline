import pandas as pd
import json
from typing import Dict
from collections import Counter


def basic_stats(df: pd.DataFrame) -> Dict:
    numerical = df.select_dtypes(include=["number"])
    categorical = df.select_dtypes(include=["object", "category", "bool"])

    stats = {
        "numerical_summary": numerical.describe().to_dict(),
        "categorical_summary": {
            col: dict(df[col].value_counts().head(10))
            for col in categorical.columns
        }
    }
    return stats


def check_class_balance(df: pd.DataFrame, target_columns: list[str]) -> Dict:
    result = {}
    for col in target_columns:
        if col not in df.columns:
            result[col] = {"error": f"Target column {col} not found in dataset"}
        else:
            counter = dict(Counter(df[col]))
            result[col] = {"class_distribution": counter}
    return result


def perform_eda(df: pd.DataFrame, target_columns: list[str]) -> Dict:
    stats = basic_stats(df)
    stats["class_balance"] = check_class_balance(df, target_columns)
    return stats


def save_eda_report(report: Dict, path: str = "outputs/eda_report.json") -> None:
    with open(path, "w") as f:
        json.dump(report, f, indent=4, default=str)  # 👈 This avoids crashes on int64, float64

