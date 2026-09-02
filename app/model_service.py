"""Single model-loading and prediction service used by API and Gradio."""
import json
from functools import lru_cache

import joblib
import pandas as pd

from app.schemas import LoanApplication
from src import config
from src.features import engineer_features


class ModelNotAvailableError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_artifacts():
    if not config.MODEL_PATH.exists() or not config.METADATA_PATH.exists():
        raise ModelNotAvailableError("Model or metadata is missing; run python -m src.train.")
    pipeline = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    if "optimized_threshold" not in metadata:
        raise ModelNotAvailableError("Metadata has no optimized_threshold; retrain the model.")
    return pipeline, metadata


def get_metadata() -> dict:
    try:
        return _load_artifacts()[1]
    except ModelNotAvailableError:
        return {}


def is_model_loaded() -> bool:
    try:
        _load_artifacts()
        return True
    except Exception:
        return False


def _application_to_frame(app: LoanApplication) -> pd.DataFrame:
    return pd.DataFrame([{
        "Income": app.income, "Age": app.age, "Experience": app.experience,
        "Married/Single": app.married_single, "House_Ownership": app.house_ownership,
        "Car_Ownership": app.car_ownership, "Profession": app.profession,
        "CITY": app.city, "STATE": app.state,
        "CURRENT_JOB_YRS": app.current_job_yrs,
        "CURRENT_HOUSE_YRS": app.current_house_yrs,
    }])


def predict(app: LoanApplication) -> dict:
    pipeline, metadata = _load_artifacts()
    probability = float(pipeline.predict_proba(engineer_features(_application_to_frame(app)))[:, 1][0])
    threshold = float(metadata["optimized_threshold"])
    return {
        "risk_probability": probability,
        "risk_score": round(probability * 1000),
        "decision": "REVIEW / DECLINE" if probability >= threshold else "APPROVE",
        "threshold_used": threshold,
    }

