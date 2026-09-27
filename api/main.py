from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predicts a student's exam score using a trained ML model.",
    version="1.0.0"
)

MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# Input features used by the model.
# student_id and exam_score are excluded.
FEATURE_COLUMNS = [
    "age",
    "gender",
    "academic_level",
    "study_hours",
    "self_study_hours",
    "online_classes_hours",
    "social_media_hours",
    "gaming_hours",
    "sleep_hours",
    "screen_time_hours",
    "exercise_minutes",
    "caffeine_intake_mg",
    "part_time_job",
    "upcoming_deadline",
    "internet_quality",
    "mental_health_score",
    "focus_index",
    "burnout_level",
    "productivity_score"
]

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class PredictionRequest(BaseModel):
    features: dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is running"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in request.features
    ]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing_features}"
        )

    # Keep only the expected columns in the correct order
    input_data = {
        column: request.features[column]
        for column in FEATURE_COLUMNS
    }

    input_df = pd.DataFrame([input_data])

    # Apply the same preprocessing used during training
    processed_data = preprocessor.transform(input_df)

    # Predict exam score
    prediction = model.predict(processed_data)[0]

    return {
        "predicted_exam_score": round(float(prediction), 2)
    }