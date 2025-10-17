from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# -------------------------------
# Pydantic model for request body
# -------------------------------
class ETAPredictRequest(BaseModel):
    distance_km: float
    speed_kmh: float
    status: int
    bus_lat: float = 0.0
    bus_lng: float = 0.0
    student_lat: float = 0.0
    student_lng: float = 0.0
    speed: float = 0.0
    emergency: int = 0
    hour: int = 12
    weekday: int = 1

# -------------------------------
# Initialize FastAPI
# -------------------------------
app = FastAPI(title="ETA Prediction API")

# -------------------------------
# Load model
# -------------------------------
try:
    model = joblib.load("eta_model.joblib")
except Exception as e:
    print("Error loading model:", e)
    model = None

# -------------------------------
# Health check
# -------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "ETA Prediction API is live 🚀"}

# -------------------------------
# ETA prediction endpoint
# -------------------------------
@app.post("/predict_eta")
def predict_eta(request: ETAPredictRequest):
    if model is None:
        return {"error": "Model not loaded"}

    # Extract features in the order expected by your model
    features = np.array([[
        request.distance_km,
        request.speed_kmh,
        request.status,
        request.bus_lat,
        request.bus_lng,
        request.student_lat,
        request.student_lng,
        request.speed,
        request.emergency,
        request.hour,
        request.weekday
    ]])

    try:
        eta = model.predict(features)[0]
        return {
            "eta_minutes": float(eta),
            "inputs_used": request.dict()
        }
    except Exception as e:
        return {"error": str(e)}
