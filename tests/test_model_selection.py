from app.model_selection import model_selection_pipeline
import pandas as pd

def test_model_selection():
    df = pd.read_csv("outputs/selected_features.csv")
    result = model_selection_pipeline(df, target_column="label", task_type="classification")
    assert "model" in result
