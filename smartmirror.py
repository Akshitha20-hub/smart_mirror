import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LinearRegression

# ----------------------------------
# 🌤 App Setup
# ----------------------------------
st.set_page_config(page_title="Smart Mirror - Fabric Type Predictor", page_icon="🪞", layout="centered")

st.title("🪞 Smart Mirror: Weather-Based Fabric Type Predictor 👕🧶")
st.write("Predicts comfort score using *Linear Regression* and recommends the best *fabric material* for the current weather.")

# ----------------------------------
# 🌍 Function: Get Live Weather Data
# ----------------------------------
def get_weather(city_name):
    try:
        res = requests.get(f"https://wttr.in/{city_name}?format=j1", timeout=8)
        if res.status_code == 200:
            data = res.json()
            temp = float(data["current_condition"][0]["temp_C"])
            humidity = float(data["current_condition"][0]["humidity"])
            condition = data["current_condition"][0]["weatherDesc"][0]["value"]
            return temp, humidity, condition
    except:
        return None
    return None

# ----------------------------------
# 🏙 User Input
# ----------------------------------
city = st.text_input("Enter your city name:", "Hyderabad")

if st.button("Predict Best Fabric Type"):
    weather = get_weather(city)

    if weather:
        temp, humidity, condition = weather
        st.success(f"🌦 Live Weather in {city}: {condition}")
        st.write(f"🌡 Temperature: {temp} °C  |  💧 Humidity: {humidity}%")

        # ----------------------------------
        # 📊 Training Dataset (Temperature → Comfort)
        # ----------------------------------
        df = pd.DataFrame({
            "Temperature": [0, 5, 10, 15, 20, 25, 30, 35, 40],
            "Comfort_Score": [1, 3, 5, 7, 8, 9, 8, 6, 4]
        })

        # Linear Regression Model
        X = df[["Temperature"]]
        y = df["Comfort_Score"]
        model = LinearRegression()
        model.fit(X, y)

        # Predict current comfort
        predicted = model.predict([[temp]])[0]
        comfort_score = round(predicted, 2)

        st.subheader(f"📊 Predicted Comfort Score: *{comfort_score}/10*")

        # ----------------------------------
        # 🧵 Fabric Recommendation Logic
        # ----------------------------------
        desc = condition.lower()
        if "rain" in desc or "drizzle" in desc or "shower" in desc:
            fabrics = [
                ("🧥 Nylon", "Water-resistant and quick-drying."),
                ("💧 Polyester", "Durable and moisture-repellent."),
                ("🧵 Blended Synthetic", "Comfortable under humidity.")
            ]
        elif temp >= 35:
            fabrics = [
                ("👕 Cotton", "Highly breathable and absorbs sweat."),
                ("👚 Linen", "Very light and airy for hot climates."),
                ("🩳 Rayon", "Soft, smooth, and keeps body cool.")
            ]
        elif 25 <= temp < 35:
            fabrics = [
                ("🧶 Cotton", "Soft and comfortable for warm conditions."),
                ("🩱 Linen", "Keeps airflow; perfect for moderate heat."),
                ("👕 Chambray", "Light cotton alternative for comfort.")
            ]
        elif 15 <= temp < 25:
            fabrics = [
                ("🧥 Denim / Twill", "Holds warmth, yet comfortable."),
                ("🧶 Cotton-blend", "Good insulation without heaviness."),
                ("🧣 Flannel", "Soft and mild for pleasant weather.")
            ]
        elif 5 <= temp < 15:
            fabrics = [
                ("🧥 Wool", "Excellent warmth for cool temperatures."),
                ("🧣 Fleece", "Soft, cozy and heat-retaining."),
                ("🧶 Acrylic Blend", "Lightweight yet warm fabric.")
            ]
        else:
            fabrics = [
                ("🧣 Wool", "Thick insulation for freezing weather."),
                ("🧥 Cashmere", "Luxurious and extremely warm."),
                ("🧤 Thermal Synthetic", "Best for extreme cold and snow.")
            ]

        # ----------------------------------
        # 👕 Display Recommended Fabric Types
        # ----------------------------------
        st.markdown("### 👗 Recommended Fabric Types:")
        for fabric, reason in fabrics:
            st.markdown(f"- *{fabric}* — {reason}")

        # ----------------------------------
        # 📈 Visualization
        # ----------------------------------
        st.markdown("### 📉 Comfort vs Temperature Visualization")
        fig, ax = plt.subplots()
        ax.plot(df["Temperature"], df["Comfort_Score"], "o-", color="skyblue", label="Training Data")
        ax.scatter(temp, comfort_score, color="red", s=120, label=f"Live ({city})")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Comfort Score (1-10)")
        ax.set_title(f"Comfort Prediction using Linear Regression ({city})")
        ax.legend()
        st.pyplot(fig)

        # ----------------------------------
        # 🧠 Interpretation
        # ----------------------------------
        if comfort_score >= 8:
            st.info("🌤 Ideal weather — natural breathable fabrics like cotton and linen are best.")
        elif 5 <= comfort_score < 8:
            st.warning("🌥 Moderate comfort — blended or light synthetic fabrics recommended.")
        else:
            st.error("❄ Extreme weather — choose insulating or water-resistant materials.")
    else:
        st.error("⚠ Could not fetch weather data. Check your city name or internet connection.")

st.caption("✨ Built with Python, Streamlit, and Linear Regression ✨")