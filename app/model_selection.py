import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from typing import Literal, Dict


def evaluate_classification(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr') if hasattr(model,
                                                                                                    "predict_proba") else None
    }


def evaluate_regression(model, X_test, y_test) -> Dict:
    y_pred = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred)
    }


def model_selection_pipeline(
        df: pd.DataFrame,
        target_column: str,
        task_type: Literal["classification", "regression"]
) -> Dict:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier()
        }
    else:
        models = {
            "Ridge": Ridge(),
            "RandomForestRegressor": RandomForestRegressor()
        }

    best_score = float("-inf") if task_type == "classification" else float("inf")
    best_model = None
    best_metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        if task_type == "classification":
            metrics = evaluate_classification(model, X_test, y_test)
            score = metrics["f1_score"]
        else:
            metrics = evaluate_regression(model, X_test, y_test)
            score = -metrics["rmse"]

        if (task_type == "classification" and score > best_score) or (task_type == "regression" and score < best_score):
            best_score = score
            best_model = model
            best_metrics = metrics

    joblib.dump(best_model, "models/best_model.pkl")
    return {"model": best_model.__class__.__name__, "metrics": best_metrics}
