import streamlit as st
from datetime import date

from src.llm_service import generate_guidance
from src.predict import DAYS_OF_WEEK, predict_demand, validate_inputs

st.set_page_config(page_title="Bike Demand AI", page_icon="🚲", layout="centered")
st.title("🚲 Seoul Bike Demand Assistant")
st.caption("Predict hourly bike rentals and receive concise AI guidance.")

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
days = list(DAYS_OF_WEEK)

with st.form("prediction_form"):
    current_day = date.today().weekday()
    month = st.selectbox("Month", range(1, 13), index=date.today().month - 1)
    day_of_week = st.selectbox("Day of week", days, index=current_day)
    hour = st.slider("Hour of day", 0, 23, 8)
    temperature = st.number_input("Temperature (°C)", -30.0, 45.0, 24.5, 0.1)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 58.0, 1.0)
    wind_speed = st.number_input("Wind speed (m/s)", 0.0, 50.0, 1.2, 0.1)
    visibility = st.number_input("Visibility (10m)", 0.0, 3000.0, 2000.0, 10.0)
    dew_point = st.number_input("Dew point temperature (°C)", -40.0, 40.0, 15.6, 0.1)
    solar_radiation = st.number_input("Solar radiation (MJ/m2)", 0.0, 10.0, 1.15, 0.01)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 1000.0, 0.0, 0.1)
    snowfall = st.number_input("Snowfall (cm)", 0.0, 100.0, 0.0, 0.1)
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])
    submitted = st.form_submit_button("Predict demand")

if submitted:
    inputs = {
        "Month": month,
        "DayOfWeek": day_of_week,
        "Hour": hour,
        "Temperature(°C)": temperature,
        "Humidity(%)": humidity,
        "Wind speed (m/s)": wind_speed,
        "Visibility (10m)": visibility,
        "Dew point temperature(°C)": dew_point,
        "Solar Radiation (MJ/m2)": solar_radiation,
        "Rainfall(mm)": rainfall,
        "Snowfall (cm)": snowfall,
        "Seasons": season,
        "Holiday": holiday,
    }

    errors = validate_inputs(inputs)
    if errors:
        st.error("Please correct the following input(s):\n\n- " + "\n- ".join(errors))
    else:
        try:
            prediction = predict_demand(inputs)
            st.success(f"Predicted bike demand: {prediction:,} rentals")

            with st.spinner("Generating concise AI guidance..."):
                guidance, llm_error = generate_guidance(inputs, prediction)

            if guidance:
                st.subheader("AI-generated guidance")
                st.write(guidance)
            else:
                st.warning(
                    "The ML prediction is available, but local LLM guidance "
                    f"could not be generated. {llm_error}"
                )
        except Exception as exc:
            st.error(f"Prediction could not be completed: {exc}")
