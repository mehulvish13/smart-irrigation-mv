
# Import Streamlit for building the web app interface
import streamlit as st
# Import pandas for data manipulation
import pandas as pd
# Import joblib for loading the pre-trained ML model
import joblib


# Load the pre-trained irrigation model
model = joblib.load("Farm_Irrigation_System.pkl")

# Set the title and subheader for the Streamlit app
st.title("Smart Irrigation Prediction App")
st.subheader("Enter the sensor values below to predict the irrigation status:")


# Collect sensor input from the user using sliders
# There are 20 sensors, each value is normalized between 0 and 1
sensor_value = []
for i in range(20):
    # Slider for each sensor input
    val = st.slider(f"Sensor {i}", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    sensor_value.append(val)

# When the user clicks the prediction button
if st.button("Predict Irrigation Status"):
    # Convert the sensor values list to a DataFrame for model input
    input_data = pd.DataFrame([sensor_value])
    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the predicted irrigation status for each parcel
    st.markdown("### Predicted Irrigation Status:")
    for i, status in enumerate(prediction[0]):
        st.write(f"Sprinkler {i} (parcel_{i}): {'ON' if status == 1 else 'OFF'}")
    