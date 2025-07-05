import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from typing import Literal, Dict, Union


def get_hyperparameter_space(model_name: str, task_type: str):
    if task_type == "classification":
        if model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=1000)
            param_grid = {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        elif model_name == "RandomForestClassifier":
            model = RandomForestClassifier()
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
    else:
        if model_name == "Ridge":
            model = Ridge()
            param_grid = {
                "alpha": [0.01, 0.1, 1, 10]
            }
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor()
            param_grid = {
                "n_estimators": [50, 100, 150],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
    return model, param_grid


def hyperparameter_tuning(
        df: pd.DataFrame,
        target_column: str,
        model_name: str,
        task_type: Literal["classification", "regression"],
        method: Literal["grid", "random"] = "random"
) -> Dict:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, param_grid = get_hyperparameter_space(model_name, task_type)

    search = GridSearchCV if method == "grid" else RandomizedSearchCV
    searcher = search(model, param_grid, cv=3, scoring="f1_weighted" if task_type == "classification" else "r2",
                      n_iter=10 if method == "random" else None)
    searcher.fit(X_train, y_train)

    best_model = searcher.best_estimator_
    joblib.dump(best_model, "models/tuned_model.pkl")

    return {
        "best_params": searcher.best_params_,
        "score": searcher.best_score_,
        "model": model_name
    }
