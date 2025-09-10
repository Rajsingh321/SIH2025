import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


st.set_page_config(page_title="Satellite Collision Predictor", layout="centered")

# Title
st.title("🛰️ Satellite Collision Predictor")

# Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Satellite 1 Details")
    sat1_long = st.number_input("Longitude (x1)", -180.0, 180.0, 45.0, key="s1_long")
    sat1_lat = st.number_input("Latitude (y1)", -90.0, 90.0, 10.0, key="s1_lat")
    sat1_alt = st.number_input("Altitude (z1)", -90.0, 90.0, -12.0, key="s1_alt")

with col2:
    st.subheader("Satellite 2 Details")
    sat2_long = st.number_input("Longitude (x2)", -180.0, 180.0, -30.0, key="s2_long")
    sat2_lat = st.number_input("Latitude (y2)", -90.0, 90.0, -12.0, key="s2_lat")
    sat2_alt = st.number_input("Altitude (z3)", -90.0, 90.0, -12.0, key="s2_alt")


if st.button("🔍 Predict Collision (Demo)", type='primary'):

    payload = {
        "sat1_long": sat1_long,
        "sat1_lat": sat1_lat,
        "sat1_alt": sat1_alt,
        "sat2_long": sat2_long,
        "sat2_lat": sat2_lat,
        "sat2_alt": sat2_alt
    }

    # Call FastAPI model
    url = "http://127.0.0.1:8000/predict"  # replace with deployed URL if needed
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()

        # Extract model outputs
        prediction = result["prediction"]
        risk_score = result["risk_score"]           # 0-1
        conclusion = result["conclusion"]

        # Calculate metrics from inputs
        p1 = np.array([sat1_long, sat1_lat, sat1_alt])
        p2 = np.array([sat2_long, sat2_lat, sat2_alt])
        distance = np.linalg.norm(p1 - p2)

        # Collision probability %
        collision_prob = round(risk_score * 100, 2)

        # Alert Level
        if collision_prob < 30:
            alert = "Safe ✅"
        elif collision_prob <= 70:
            alert = "Caution ⚠"
        else:
            alert = "Danger 🚨"

    else:
        st.error("API request failed 🚨")
        prediction = "Error"
        collision_prob = 0
        distance = 0
        alert = "Unknown"
        conclusion = "Could not get prediction from model."  
    
        st.subheader("Satellites Visual")
        col1, col2 = st.columns(2)
        with col1:
            st.image("two.jpg")
        with col2:
            st.image("one.jpg")    
    
        st.subheader("Prediction Summary")
        st.write(f"The predicted collision probability is *{collision_prob}%*. "
             f"The satellites are approximately *{distance:.2f} km apart*, moving with relative velocity "
             f"of *{np.random.uniform(0, 10):.2f} km/s*. Alert Level: **{alert}**.")
        # 3D Orbit Visualization

        st.subheader("3D Orbit Visualization (Demo)")
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
    
        # Dummy orbit trajectories
        t = np.linspace(0, 2*np.pi, 100)
        ax.plot((sat1_alt+10)*np.cos(t), (sat1_alt+10)*np.sin(t), 10*t, label="Satellite 1")
        ax.plot((sat2_alt+20)*np.cos(t), (sat2_alt+20)*np.sin(t), 20*t, label="Satellite 2")
    
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title("Satellite Orbits (Demo)")
        ax.legend()
        st.pyplot(fig)

    
    # ---------------------------
    # Risk Matrix / Summary Table
    # ---------------------------
        st.subheader("Risk Summary Table")
        df = pd.DataFrame({
        "Metric": ["Collision Probability (%)", "Distance (km)", "Relative Velocity (km/s)", "Alert Level"],
        "Value": [collision_prob, f"{distance:.2f}", f"{np.random.uniform(0, 10):.2f}", alert]
        })
        st.table(df)
    
        st.subheader("Conclusion")
        st.write(f"Based on the demo data, the collision risk is *{alert}*. "
             "Please note this is a demo; actual predictions require the ML model API.")
        

