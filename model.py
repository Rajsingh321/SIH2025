import numpy as np
from skyfield.api import EarthSatellite, load, utc
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Parameters
# -----------------------------
NUM_SATELLITES = 5
FORCED_COLLISIONS = [(0, 1, 20), (2, 3, 40)]
DURATION_MINUTES = 60
COLLISION_THRESHOLD_KM = 9000
NOISE_LEVEL_KM = 0.2

# -----------------------------
# Load TLE data
# -----------------------------
with open('gp.txt') as file:
    lines = file.readlines()

ts = load.timescale()
start = datetime.now().replace(tzinfo=utc)
times = ts.utc(start.year, start.month, start.day, start.hour,
               start.minute + np.arange(0, DURATION_MINUTES, 1))

# -----------------------------
# Load satellites
# -----------------------------
satellites = []
for i in range(0, NUM_SATELLITES * 3, 3):
    if i + 2 >= len(lines):
        break  
    name = lines[i].strip()
    tle1 = lines[i+1].strip()
    tle2 = lines[i+2].strip()
    sat = EarthSatellite(tle1, tle2, name, ts)
    satellites.append(sat)

# -----------------------------
# Generate positions with noise
# -----------------------------
positions = []
for sat in satellites:
    geo = sat.at(times)
    pos = geo.position.km.T
    noise = np.random.normal(0.0, NOISE_LEVEL_KM, pos.shape)
    positions.append(pos + noise)

# Force collisions
for (i, j, t_idx) in FORCED_COLLISIONS:
    if i >= len(positions) or j >= len(positions) or t_idx >= len(times):
        continue
    offset = np.random.uniform(-0.4, 0.4, 3)
    positions[j][t_idx] = positions[i][t_idx] + offset

# -----------------------------
# Generate dataset
# -----------------------------
data = []
actual_satellites = len(positions)
time_steps = len(times)

for t in range(time_steps):
    for i in range(actual_satellites*10000):
        for j in range(i+1, actual_satellites):
            p1 = positions[i][t]
            p2 = positions[j][t]
            distance = np.linalg.norm(p1 - p2)
            label = 1 if distance <= COLLISION_THRESHOLD_KM else 0
            features = np.concatenate((p1, p2, [distance]))
            data.append(np.append(features, label))

df = pd.DataFrame(data, columns=['x1','y1','z1','x2','y2','z2','distance','label'])
df.to_csv('generated_collision_dataset.csv', index=False)

# -----------------------------
# Train model
# -----------------------------
X = df[['x1','y1','z1','x2','y2','z2','distance']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation (optional print)
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, 'collision_predictor_model.pkl')
print("✅ Model saved as 'collision_predictor_model.pkl'")
