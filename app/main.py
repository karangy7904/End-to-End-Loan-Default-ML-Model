"""
FastAPI service for the loan risk model.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import LoanApplication, RiskPrediction, HealthResponse
from app import model_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loan-risk-api")

app = FastAPI(
    title="Loan Risk Prediction API",
    description="Predicts loan Risk_Flag probability from applicant features "
                 "(income, age, experience, profession, location, etc.).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {"message": "Loan Risk Prediction API — see /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    metadata = model_service.get_metadata()
    loaded = model_service.is_model_loaded()
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_name=metadata.get("validation_metrics", {}).get("model"),
    )


@app.get("/ready", tags=["meta"])
def readiness():
    """Return 503 until both the fitted model and threshold metadata load."""
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model artifacts are not ready")
    return {"status": "ready"}


@app.post("/predict", response_model=RiskPrediction, tags=["prediction"])
def predict_risk(application: LoanApplication):
    """
    Score a single loan application and return a default-risk probability,
    a 0-1000 risk score, and a decision based on the tuned threshold.
    """
    try:
        result = model_service.predict(application)
    except model_service.ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return RiskPrediction(**result)
