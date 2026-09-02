"""
Pydantic request/response models for the Loan Risk API.

Field names match Training_Data.csv / Test_Data.csv exactly (Income, Age,
Experience, Married/Single, House_Ownership, Car_Ownership, Profession,
CITY, STATE, CURRENT_JOB_YRS, CURRENT_HOUSE_YRS).
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator

HOUSE_OWNERSHIP_VALUES = {"rented", "owned", "norent_noown"}
CAR_OWNERSHIP_VALUES = {"yes", "no"}
MARITAL_VALUES = {"single", "married"}


class LoanApplication(BaseModel):
    income: float = Field(..., ge=0, description="Annual income")
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    experience: int = Field(..., ge=0, le=60, description="Total years of work experience")
    married_single: str = Field(..., description="'single' or 'married'")
    house_ownership: str = Field(..., description="'rented', 'owned', or 'norent_noown'")
    car_ownership: str = Field(..., description="'yes' or 'no'")
    profession: str = Field(..., description="Applicant's profession, e.g. 'Software_Developer'")
    city: str = Field(..., description="City of residence")
    state: str = Field(..., description="State of residence")
    current_job_yrs: int = Field(..., ge=0, le=60, description="Years at current job")
    current_house_yrs: int = Field(..., ge=0, le=60, description="Years at current residence")

    @field_validator("married_single")
    @classmethod
    def validate_marital(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in MARITAL_VALUES:
            raise ValueError(f"married_single must be one of {sorted(MARITAL_VALUES)}")
        return v

    @field_validator("house_ownership")
    @classmethod
    def validate_house(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in HOUSE_OWNERSHIP_VALUES:
            raise ValueError(f"house_ownership must be one of {sorted(HOUSE_OWNERSHIP_VALUES)}")
        return v

    @field_validator("car_ownership")
    @classmethod
    def validate_car(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in CAR_OWNERSHIP_VALUES:
            raise ValueError(f"car_ownership must be one of {sorted(CAR_OWNERSHIP_VALUES)}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "income": 1303834,
                "age": 23,
                "experience": 3,
                "married_single": "single",
                "house_ownership": "rented",
                "car_ownership": "no",
                "profession": "Mechanical_engineer",
                "city": "Rewa",
                "state": "Madhya_Pradesh",
                "current_job_yrs": 3,
                "current_house_yrs": 13,
            }
        }
    }


class RiskPrediction(BaseModel):
    risk_probability: float = Field(..., description="Predicted probability of Risk_Flag=1 (0-1)")
    risk_score: int = Field(..., description="Risk probability scaled 0-1000 for readability")
    decision: str = Field(..., description="APPROVE or REVIEW/DECLINE, based on the tuned threshold")
    threshold_used: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
