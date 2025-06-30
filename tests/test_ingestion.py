from app.ingestion import load_dataset, analyze_dataset

def test_analyze_dataset():
    df = load_dataset("data/sample_dataset.csv")
    profile = analyze_dataset(df)
    assert "column_types" in profile
    assert "missing_and_duplicates" in profile
