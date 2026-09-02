"""Shared paths and model metadata for training, API, and UI."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "data" / "Training Data.csv"
TEST_PATH = ROOT_DIR / "data" / "Test Data.csv"
MODEL_PATH = ROOT_DIR / "models" / "final_model_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "models" / "model_metadata.json"
TARGET_COL = "Risk_Flag"
ID_COL = "Id"

