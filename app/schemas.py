"""Validated request and response models for the prediction API."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HeartFeatures(BaseModel):

    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="Dataset encoding: 1=male, 0=female")
    cp: int = Field(ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: int = Field(ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl (1=yes)")
    restecg: int = Field(ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: int = Field(ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina (1=yes)")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(ge=0, le=2, description="Slope of peak exercise ST segment (0-2)")
    ca: int = Field(ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(ge=0, le=3, description="Thal dataset category (0-3)")

    def to_feature_dict(self) -> dict[str, int | float]:
        """Return fields in schema order as a plain dictionary."""
        return self.model_dump()


class PredictionResponse(BaseModel):
    heart_disease: bool = Field(description="Predicted positive class (target=1)")
    prediction: Literal["presence", "absence"]
    probability: float = Field(ge=0, le=1, description="Model probability for target=1")
    model: str
