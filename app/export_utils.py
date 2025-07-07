import os
import json
import joblib

def save_pipeline(pipeline, path):
    """Save the full sklearn pipeline to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path):
    """Load a saved sklearn pipeline from disk."""
    return joblib.load(path)


def save_metadata(metadata: dict, path: str):
    """Save metadata as a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: str) -> dict:
    """Load metadata from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
