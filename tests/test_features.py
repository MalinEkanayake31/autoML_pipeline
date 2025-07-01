from app.features import feature_engineering_pipeline
import pandas as pd

def test_feature_engineering():
    df = pd.read_csv("data/sample_data.csv")
    df = feature_engineering_pipeline(df, encoding="label", scale="minmax")
    assert not df.isnull().any().any()
