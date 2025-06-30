from app.ingestion import load_dataset

def test_load_dataset():
    df = load_dataset("data/sample_dataset.csv")
    assert not df.empty
