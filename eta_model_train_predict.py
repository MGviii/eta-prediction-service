import math
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import os
import joblib  # <--- for saving/loading model

# -----------------------------
# Haversine formula for distance
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # distance in meters

# -----------------------------
# Load JSON data
# -----------------------------
def load_data(json_file='data.json'):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"{json_file} not found")
    with open(json_file) as f:
        return json.load(f)

# -----------------------------
# Prepare training data
# -----------------------------
def prepare_training_data(data):
    records = []
    for bus_id, bus_info in data['busLocations'].items():
        history = bus_info.get('history', {})
        sorted_timestamps = sorted(history.keys())
        for i in range(1, len(sorted_timestamps)):
            prev = history[sorted_timestamps[i-1]]
            curr = history[sorted_timestamps[i]]
            time_diff = (curr['timestamp'] - prev['timestamp']) / 1000  # seconds
            distance = haversine(prev['Latitude'], prev['Longitude'], curr['Latitude'], curr['Longitude'])
            speed = distance / time_diff if time_diff > 0 else 0
            hour = datetime.strptime(curr['Date'] + ' ' + curr['Time (UTC)'], '%d/%m/%Y %H:%M:%S').hour
            records.append({
                'bus_lat': curr['Latitude'],
                'bus_lon': curr['Longitude'],
                'distance': distance,
                'speed': speed,
                'hour': hour,
                'eta_minutes': time_diff / 60  # actual travel time as ETA
            })
    return pd.DataFrame(records)

# -----------------------------
# Train Random Forest model
# -----------------------------
def train_model(df):
    X = df[['bus_lat', 'bus_lon', 'distance', 'speed', 'hour']]
    y = df['eta_minutes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"Model trained. MSE: {mse:.2f}")
    return model

# -----------------------------
# Save model to .joblib
# -----------------------------
def save_model(model, path='rf_eta_model.joblib'):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# -----------------------------
# Load model from .joblib
# -----------------------------
def load_model(path='rf_eta_model.joblib'):
    return joblib.load(path)

# -----------------------------
# Predict ETAs for multiple buses
# -----------------------------
def predict_etas_all_buses(data, model):
    all_etas = {}
    for bus_id, bus_info in data['busLocations'].items():
        current = bus_info['current']
        bus_lat = current['Latitude']
        bus_lon = current['Longitude']
        bus_logs = [v for v in data['logs'].values() if v['busId'] == bus_id]
        etas = {}

        for log in bus_logs:
            if 'studentName' not in log:
                continue
            student = log['studentName']
            stop_lat = log['location']['lat']
            stop_lon = log['location']['lng']
            distance = haversine(bus_lat, bus_lon, stop_lat, stop_lon)
            speed = current.get('Speed', 0.5) * 1000 / 3600  # m/s
            hour = datetime.strptime(current['Date'] + ' ' + current['Time (UTC)'], '%d/%m/%Y %H:%M:%S').hour
            features = pd.DataFrame([{
                'bus_lat': bus_lat,
                'bus_lon': bus_lon,
                'distance': distance,
                'speed': speed,
                'hour': hour
            }])
            eta = model.predict(features)[0]
            etas[student] = max(0, eta)
            bus_lat, bus_lon = stop_lat, stop_lon

        all_etas[bus_id] = etas
    return all_etas

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    data = load_data('data.json')
    df = prepare_training_data(data)
    model = train_model(df)
    
    # Save the model for Node.js backend usage
    save_model(model)

    all_etas = predict_etas_all_buses(data, model)
    for bus_id, etas in all_etas.items():
        print(f"Bus {bus_id} ETAs:")
        for student, eta in etas.items():
            print(f"  {student}: {eta:.2f} minutes")
