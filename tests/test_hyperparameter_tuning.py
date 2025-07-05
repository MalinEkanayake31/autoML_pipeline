from app.hyperparameter_tuning import hyperparameter_tuning
import pandas as pd

def test_tuning():
    df = pd.read_csv("outputs/selected_features.csv")
    result = hyperparameter_tuning(df, target_column="label", model_name="RandomForestClassifier", task_type="classification")
    assert "best_params" in result
