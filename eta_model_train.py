# eta_model_train.py
import json
import pandas as pd
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------------
# 1. Load the RTDB JSON data
# -----------------------------
with open("rtbd_data.json", "r") as f:
    data = json.load(f)

logs = data['logs']
bus_locations = data['busLocations']
students_data = data['students']

# -----------------------------
# 2. Prepare dataset
# -----------------------------
dataset = []

for log_id, log in logs.items():
    if 'status' not in log:  # Skip logs without student events
        continue

    student_name = log['studentName']
    bus_id = log['busId']
    timestamp = log['timestamp']
    status = log['status']  # check-in or check-out
    student_info = None

    # Find studentId in students_data by matching studentName
    for s_id, s in students_data.items():
        if s['name'] == student_name:
            student_info = s
            break

    if not student_info:
        continue

    # Bus current location
    bus_loc = bus_locations.get(bus_id, {}).get('current', None)
    if not bus_loc:
        continue

    # Feature: Distance between bus and student's location
    student_lat = log['location']['lat']
    student_lng = log['location']['lng']
    bus_lat = bus_loc['latitude']
    bus_lng = bus_loc['longitude']
    distance_km = geodesic((bus_lat, bus_lng), (student_lat, student_lng)).km

    # Feature: Speed of bus
    speed_kmh = bus_loc.get('speed', 0) * 3.6  # m/s to km/h

    # Target: ETA in minutes (simplified: distance / speed)
    if speed_kmh > 0:
        eta = (distance_km / speed_kmh) * 60
    else:
        eta = 0  # bus is stationary

    dataset.append({
        'bus_id': bus_id,
        'student_id': student_info['studentId'],
        'distance_km': distance_km,
        'speed_kmh': speed_kmh,
        'status': 1 if status == 'check-in' else 0,  # boarding=1, dropoff=0
        'eta_minutes': eta
    })

df = pd.DataFrame(dataset)

# -----------------------------
# 3. Train RandomForest model
# -----------------------------
features = ['distance_km', 'speed_kmh', 'status']
target = 'eta_minutes'

X = df[features]
y = df[target]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# -----------------------------
# 4. Save the trained model
# -----------------------------
joblib.dump(model, "eta_model.joblib")
print("Model trained and saved as eta_model.joblib")
