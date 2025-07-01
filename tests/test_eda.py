from app.eda import perform_eda
import pandas as pd

def test_eda_summary():
    df = pd.read_csv("data/sample_dataset.csv")
    eda = perform_eda(df, target_column="label")
    assert "numerical_summary" in eda
    assert "categorical_summary" in eda
