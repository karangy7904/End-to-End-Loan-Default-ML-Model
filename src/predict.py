"""Score a CSV with the saved model and optimized decision threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
try:
    from src.features import engineer_features
except ModuleNotFoundError:  # supports direct `python src/predict.py`
    from features import engineer_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV to score")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default="models/final_model_pipeline.joblib")
    parser.add_argument("--metadata", default="models/model_metadata.json")
    args = parser.parse_args()

    model = joblib.load(args.model)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    data = pd.read_csv(args.input)
    id_col = next((c for c in data.columns if c.lower() == "id"), None)
    ids = data[id_col].copy() if id_col else pd.Series(range(len(data)), name="Id")
    features = data.drop(columns=[id_col], errors="ignore") if id_col else data
    features = engineer_features(features)
    probabilities = model.predict_proba(features)[:, 1]
    threshold = float(metadata["optimized_threshold"])
    result = pd.DataFrame({"Id": ids, "risk_probability": probabilities,
                           "Risk_Flag": (probabilities >= threshold).astype(int)})
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
