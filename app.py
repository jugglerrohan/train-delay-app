import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model_v2.pkl")

st.title("Train Delay Prediction")

# Inputs
distance = st.number_input("Distance Between Stations (km)", min_value=0.0)

train_type = st.selectbox("Train Type", ["Passenger", "Express", "Freight"])
weather = st.selectbox("Weather Conditions", ["Clear", "Rain", "Fog"])
route = st.selectbox("Route Congestion", ["Low", "Medium", "High"])
time = st.selectbox("Time of Day", ["Morning", "Evening", "Night"])
day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Saturday", "Sunday"])

# Encode categorical values
train_type_val = ["Passenger", "Express", "Freight"].index(train_type)
weather_val = ["Clear", "Rain", "Fog"].index(weather)

route_low = 1 if route == "Low" else 0
route_medium = 1 if route == "Medium" else 0

# Day encoding
day_dict = {
    'Day of the Week_Monday': 1 if day == "Monday" else 0,
    'Day of the Week_Saturday': 1 if day == "Saturday" else 0,
    'Day of the Week_Sunday': 1 if day == "Sunday" else 0,
    'Day of the Week_Thursday': 1 if day == "Thursday" else 0,
    'Day of the Week_Tuesday': 1 if day == "Tuesday" else 0,
    'Day of the Week_Wednesday': 1 if day == "Wednesday" else 0,
}

# Time encoding
time_dict = {
    'Time of Day_Evening': 1 if time == "Evening" else 0,
    'Time of Day_Morning': 1 if time == "Morning" else 0,
    'Time of Day_Night': 1 if time == "Night" else 0,
}

# Prediction
if st.button("Predict"):

    input_dict = {
        'Distance Between Stations (km)': distance,
        'Weather Conditions': weather_val,
        'Train Type': train_type_val,
        **day_dict,
        **time_dict,
        'Route Congestion_Low': route_low,
        'Route Congestion_Medium': route_medium
    }

    # Interaction features
    days = ["Monday","Saturday","Sunday","Thursday","Tuesday","Wednesday"]
    times = ["Evening","Morning","Night"]

    for d in days:
        for t in times:
            col = f'Day of the Week_{d}_Time of Day_{t}_interaction'
            input_dict[col] = day_dict.get(f'Day of the Week_{d}', 0) * time_dict.get(f'Time of Day_{t}', 0)

    # Additional features
    input_dict['Train_Type_squared'] = train_type_val ** 2
    input_dict['Previous_Delay'] = 0  # you can improve later

    input_df = pd.DataFrame([input_dict])

    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Delay: {prediction[0]:.2f} minutes")
    except Exception as e:
        st.error(f"Error: {e}")
        