import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load('collision_predictor_model.pkl')

st.set_page_config(page_title="Satellite Collision Predictor", layout="centered")
st.title("🛰 Satellite Collision Predictor")

# -----------------------------
# Input Fields
# -----------------------------
st.subheader("Enter Satellite Positions (km)")
col1, col2 = st.columns(2)

with col1:
    x1 = st.number_input("Satellite 1 - X:", value=0.0, format="%.3f")
    y1 = st.number_input("Satellite 1 - Y:", value=0.0, format="%.3f")
    z1 = st.number_input("Satellite 1 - Z:", value=0.0, format="%.3f")

with col2:
    x2 = st.number_input("Satellite 2 - X:", value=0.0, format="%.3f")
    y2 = st.number_input("Satellite 2 - Y:", value=0.0, format="%.3f")
    z2 = st.number_input("Satellite 2 - Z:", value=0.0, format="%.3f")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Collision"):
    try:
        # Positions & distance
        p1 = np.array([x1, y1, z1])
        p2 = np.array([x2, y2, z2])
        distance = np.linalg.norm(p1 - p2)
        
        # Prediction
        features = np.concatenate((p1,p2,[distance])).reshape(1,-1)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]*100 if hasattr(model, "predict_proba") else None

        # -----------------------------
        # Time calculation (average speed assumption)
        # -----------------------------
        avg_speed_km_s = 7.8  # km/s typical LEO
        time_seconds = distance / avg_speed_km_s
        time_minutes = time_seconds / 60

        # -----------------------------
        # Message
        # -----------------------------
        if prediction==1:
            alert_msg = f"⚠ Collision Risk Detected!\nDistance: {distance:.3f} km\nProbability: {proba:.2f}%\nTime to collision: {time_minutes:.1f} min"
            conclusion = "Recommendation: Take preventive action. Adjust orbit or monitor closely."
        else:
            alert_msg = f"✅ No Collision Risk.\nDistance: {distance:.3f} km\nProbability: {proba:.2f}%\nTime: {time_minutes:.1f} min"
            conclusion = "Orbit is safe. No immediate action required."

        st.markdown(f"### Prediction Message:\n{alert_msg}")

        # -----------------------------
        # 3D Visualization
        # -----------------------------
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*p1, color='blue', s=50, label='Satellite 1')
        ax.scatter(*p2, color='red', s=50, label='Satellite 2')
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],[p1[2],p2[2]], color='green', linestyle='--', label='Distance Vector')
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
        ax.set_title('Satellite Positions')
        ax.legend()
        st.pyplot(fig)

        # -----------------------------
        # Risk Matrix
        # -----------------------------
        risk_data = {
            "Alert": ["Collision Risk" if prediction==1 else "Safe"],
            "Distance (km)": [distance],
            "Probability (%)": [proba],
            "Time (min)": [time_minutes]
        }
        risk_df = pd.DataFrame(risk_data)
        st.subheader("Risk Matrix")
        st.dataframe(risk_df)

        # -----------------------------
        # Conclusion
        # -----------------------------
        st.subheader("Conclusion")
        st.markdown(conclusion)

    except Exception as e:
        st.error(f"Error: {e}")
