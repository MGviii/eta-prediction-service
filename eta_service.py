from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import pickle

# Load trained model
with open("eta_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Define Pydantic model for IoT input
class IoTETARequest(BaseModel):
    distance_km: float
    speed_kmh: float
    status: int
    bus_lat: Optional[float] = None
    bus_lng: Optional[float] = None
    student_lat: Optional[float] = None
    student_lng: Optional[float] = None
    speed: Optional[float] = None
    emergency: Optional[int] = None
    hour: Optional[int] = None
    weekday: Optional[int] = None

# List of features your model actually uses
MODEL_FEATURES = ["distance_km", "speed_kmh", "status"]

@app.post("/predict_eta")
def predict_eta(request: IoTETARequest):
    # Convert input to DataFrame
    input_data = request.dict()

    # Keep only features needed for prediction
    features = pd.DataFrame([{k: input_data[k] for k in MODEL_FEATURES}])

    # Make prediction
    eta = model.predict(features)[0]

    return {
        "eta_minutes": eta,
        "inputs_used": features.iloc[0].to_dict()
    }
