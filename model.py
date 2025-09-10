from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# -------------------------
# Initialize FastAPI
# -------------------------
app = FastAPI(title="Satellite Collision Predictor API")

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = "model.pkl"   # change if needed
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------
# Define Input Schema
# -------------------------
class SatelliteInput(BaseModel):
    sat1_long: float
    sat1_lat: float
    sat1_speed: float
    sat2_long: float
    sat2_lat: float
    sat2_speed: float

# -------------------------
# Define Output Schema
# -------------------------
class PredictionOutput(BaseModel):
    prediction: str            # e.g., "Collision Likely" / "No Collision"
    conclusion: str            # short summary of the result
    risk_score: float          # probability/confidence score
    collision_point: tuple     # (longitude, latitude) if applicable
    time_to_collision: float   # estimated time in seconds (if collision predicted)

# -------------------------
# API Endpoint
# -------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict_collision(data: SatelliteInput):
    # Prepare features for model
    features = [
        data.sat1_long, data.sat1_lat, data.sat1_speed,
        data.sat2_long, data.sat2_lat, data.sat2_speed
    ]
    
    # Run model prediction (adjust as per your trained model)
    risk_score = model.predict_proba([features])[0][1]  # probability of collision
    prediction = "Collision Likely" if risk_score > 0.5 else "No Collision"
    
    # Example logic for conclusion
    conclusion = (
        "Satellites are on a potential collision course."
        if prediction == "Collision Likely"
        else "Satellites are at a safe distance."
    )
    
    # Dummy values for now (can be replaced with orbit calc)
    collision_point = (data.sat1_long, data.sat1_lat)
    time_to_collision = 120.5 if prediction == "Collision Likely" else -1
    
    return PredictionOutput(
        prediction=prediction,
        conclusion=conclusion,
        risk_score=round(risk_score, 3),
        collision_point=collision_point,
        time_to_collision=time_to_collision
    )