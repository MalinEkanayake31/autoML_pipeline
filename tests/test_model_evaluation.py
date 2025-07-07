import pandas as pd
from app.model_evaluation import evaluate_model_on_test

def test_evaluate_regression():
    # Create a small synthetic regression test set
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'price': [100, 200, 300, 400, 500]
    })
    # Save a dummy model
    from sklearn.linear_model import Ridge
    import joblib
    model = Ridge().fit(df[['feature1', 'feature2']], df['price'])
    joblib.dump(model, 'models/tuned_model.pkl')
    # Evaluate
    metrics = evaluate_model_on_test(df, 'price', 'models/tuned_model.pkl', 'regression', output_dir='outputs')
    assert 'mae' in metrics and 'rmse' in metrics and 'r2' in metrics
    print('Regression evaluation test passed.')

def test_evaluate_classification():
    # Create a small synthetic classification test set
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [10, 20, 30, 40, 50, 60],
        'label': [0, 1, 0, 1, 0, 1]
    })
    # Save a dummy model
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    model = RandomForestClassifier().fit(df[['feature1', 'feature2']], df['label'])
    joblib.dump(model, 'models/tuned_model.pkl')
    # Evaluate
    metrics = evaluate_model_on_test(df, 'label', 'models/tuned_model.pkl', 'classification', output_dir='outputs')
    assert 'accuracy' in metrics and 'f1' in metrics and 'roc_auc' in metrics and 'confusion_matrix' in metrics
    print('Classification evaluation test passed.')

if __name__ == "__main__":
    test_evaluate_regression()
    test_evaluate_classification()
