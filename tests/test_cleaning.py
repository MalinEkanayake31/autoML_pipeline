from app.cleaning import clean_data
import pandas as pd

def test_cleaning_pipeline():
    df = pd.read_csv("data/sample_data.csv")
    cleaned = clean_data(df, missing_strategy="drop")
    assert cleaned.isnull().sum().sum() == 0
    assert cleaned.duplicated().sum() == 0
