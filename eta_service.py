from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import math
from datetime import datetime
import uvicorn

# -----------------------------
# Load model and feature columns
# -----------------------------
model = joblib.load("eta_model.joblib")
feature_columns = joblib.load("eta_features.joblib")

app = FastAPI()

# -----------------------------
# Input data schema
# -----------------------------
class BusETARequest(BaseModel):
    bus_lat: float
    bus_lon: float
    speed: float  # m/s
    stops: list   # List of dicts: [{"student": str, "lat": float, "lon": float}]
    timestamp: str  # ISO datetime

# -----------------------------
# Haversine formula
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * 1000 * c  # meters

# -----------------------------
# Predict ETAs
# -----------------------------
@app.post("/predict_eta")
def predict_eta(request: BusETARequest):
    bus_lat, bus_lon = request.bus_lat, request.bus_lon
    hour = datetime.fromisoformat(request.timestamp).hour
    etas = {}

    for stop in request.stops:
        distance = haversine(bus_lat, bus_lon, stop['lat'], stop['lon'])
        features = pd.DataFrame([{
            'bus_lat': bus_lat,
            'bus_lon': bus_lon,
            'distance': distance,
            'speed': request.speed,
            'hour': hour
        }])
        eta = model.predict(features)[0]
        etas[stop['student']] = max(0, eta)
        # Move bus to this stop for next prediction
        bus_lat, bus_lon = stop['lat'], stop['lon']

    return {"etas": etas}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
