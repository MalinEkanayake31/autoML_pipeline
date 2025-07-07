import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.hyperparameter_tuning import hyperparameter_tuning
import pandas as pd

def test_tuning():
    df = pd.read_csv("outputs/selected_features.csv")
    result = hyperparameter_tuning(df, target_column="price", model_name="RandomForestRegressor", task_type="regression")
    assert "best_params" in result
