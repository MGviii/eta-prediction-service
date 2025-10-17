from fastapi import FastAPI
from pydantic import BaseModel
from geopy.distance import geodesic
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="ETA Prediction Service")

# Load your trained model
model = joblib.load("eta_model.pkl")

# Define input schema
class ETARequest(BaseModel):
    bus_lat: float
    bus_lng: float
    student_lat: float
    student_lng: float
    speed: float          # current bus speed (m/s)
    emergency: bool
    hour: int
    weekday: int

@app.post("/predict_eta")
def predict_eta(data: ETARequest):
    """
    Predict Estimated Time of Arrival (ETA) in minutes.
    """
    # Compute the distance between bus and student
    distance_km = geodesic(
        (data.bus_lat, data.bus_lng),
        (data.student_lat, data.student_lng)
    ).km

    # Convert speed from m/s to km/h
    speed_kmh = data.speed * 3.6

    # For now, define status = 1 (e.g., "en route")
    status = 1

    # Build the input array exactly like the model was trained
    features = np.array([[distance_km, speed_kmh, status]])

    # Make prediction
    eta = model.predict(features)[0]

    return {
        "eta_minutes": float(eta),
        "inputs_used": {
            "distance_km": distance_km,
            "speed_kmh": speed_kmh,
            "status": status
        }
    }

@app.get("/")
def home():
    return {"message": "ETA Prediction API is running 🚍"}
