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
# Define features actually used by the model
# -------------------------------
MODEL_FEATURES = ["distance_km", "speed_kmh", "status"]

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

    # Extract only the features your model was trained on
    features = np.array([[getattr(request, f) for f in MODEL_FEATURES]])

    try:
        eta = model.predict(features)[0]
        return {
            "eta_minutes": float(eta),
            "inputs_used": {f: getattr(request, f) for f in MODEL_FEATURES},
            "all_inputs": request.dict()  # optional: keep all fields for logging/debugging
        }
    except Exception as e:
        return {"error": str(e)}
