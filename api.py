from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# =========================
# LOAD MODEL FILES
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "student_model.pkl"
)

scaler_path = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

encoder_path = os.path.join(
    BASE_DIR,
    "models",
    "label_encoder.pkl"
)

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

# =========================
# FASTAPI APP
# =========================

app = FastAPI(
    title="Student Performance Prediction API",
    description="Machine Learning API for Student Performance Prediction",
    version="1.0"
)

# =========================
# INPUT SCHEMA
# =========================

class StudentData(BaseModel):
    age: int
    studytime: int
    failures: int
    absences: int
    G1: int
    G2: int

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API is Running"
    }

# =========================
# PREDICTION ROUTE
# =========================

@app.post("/predict")
def predict(data: StudentData):

    input_data = np.array([
        [
            data.G1,
            data.G2,
            data.absences,
            data.age,
            data.studytime,
            data.failures
        ]
    ])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Probability
    prediction_proba = model.predict_proba(input_scaled)

    # Decode label
    result = label_encoder.inverse_transform(prediction)

    confidence = float(np.max(prediction_proba) * 100)

    return {
        "prediction": result[0],
        "confidence_score": round(confidence, 2)
    }