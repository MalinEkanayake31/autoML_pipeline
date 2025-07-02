from app.feature_selection import feature_selection_pipeline
import pandas as pd

def test_feature_selection():
    df = pd.read_csv("data/sample_data.csv")
    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    selected = feature_selection_pipeline(X, y, method="mutual_info", k=5)
    assert selected.shape[1] == 5
