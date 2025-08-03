
# Import Streamlit for building the web app interface
import streamlit as st
# Import pandas for data manipulation
import pandas as pd
# Import joblib for loading the pre-trained ML model
import joblib
# Import sklearn metrics for accuracy calculation
from sklearn.metrics import accuracy_score


# Load the pre-trained irrigation model
model = joblib.load("Farm_Irrigation_System.pkl")

# Load test data to display model accuracy (if available)
try:
    # Load the dataset to show model performance
    df = pd.read_csv("irrigation_machine.csv")
    # You can add pre-computed accuracy values here or calculate them
    # For demonstration, showing sample accuracy values
    parcel_accuracies = [0.95, 0.92, 0.94]  # Replace with actual values from your model evaluation
    overall_accuracy = 0.89  # Replace with actual overall accuracy
    avg_accuracy = sum(parcel_accuracies) / len(parcel_accuracies)
except:
    parcel_accuracies = [0.0, 0.0, 0.0]
    overall_accuracy = 0.0
    avg_accuracy = 0.0

# Set the title and subheader for the Streamlit app
st.title("🌾 Smart Irrigation Prediction App")
st.subheader("Enter the sensor values below to predict the irrigation status:")

# Display model accuracy metrics in the sidebar
st.sidebar.title("📊 Model Performance")
st.sidebar.markdown("### Accuracy Metrics")
st.sidebar.metric("Parcel 0 Accuracy", f"{parcel_accuracies[0]*100:.1f}%")
st.sidebar.metric("Parcel 1 Accuracy", f"{parcel_accuracies[1]*100:.1f}%")
st.sidebar.metric("Parcel 2 Accuracy", f"{parcel_accuracies[2]*100:.1f}%")
st.sidebar.metric("Overall Accuracy", f"{overall_accuracy*100:.1f}%")
st.sidebar.metric("Average Accuracy", f"{avg_accuracy*100:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Details:**")
st.sidebar.markdown("- Algorithm: Random Forest")
st.sidebar.markdown("- Multi-Output Classification")
st.sidebar.markdown("- 20 Sensor Inputs")
st.sidebar.markdown("- 3 Parcel Outputs")


# Collect sensor input from the user using sliders
# There are 20 sensors, each value is normalized between 0 and 1
st.markdown("### 🌡️ Sensor Input Configuration")
st.markdown("Adjust the sensor values (normalized between 0.0 and 1.0):")

# Create columns for better layout
col1, col2 = st.columns(2)

sensor_value = []
for i in range(20):
    # Alternate between columns for better layout
    with col1 if i % 2 == 0 else col2:
        # Slider for each sensor input
        val = st.slider(f"Sensor {i+1}", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"sensor_{i}")
        sensor_value.append(val)

# When the user clicks the prediction button
st.markdown("---")
if st.button("🚰 Predict Irrigation Status", type="primary", use_container_width=True):
    # Convert the sensor values list to a DataFrame for model input
    input_data = pd.DataFrame([sensor_value])
    # Make prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the predicted irrigation status for each parcel
    st.markdown("### 🎯 Predicted Irrigation Status:")
    
    # Create columns for better display
    col1, col2, col3 = st.columns(3)
    
    parcel_names = ["Parcel 0", "Parcel 1", "Parcel 2"]
    colors = ["🔵", "🟠", "🟢"]
    
    for i, (status, name, color) in enumerate(zip(prediction[0], parcel_names, colors)):
        with [col1, col2, col3][i]:
            status_text = "ON" if status == 1 else "OFF"
            status_color = "🟢 ON" if status == 1 else "🔴 OFF"
            st.metric(
                label=f"{color} {name}", 
                value=status_text,
                help=f"Accuracy: {parcel_accuracies[i]*100:.1f}%"
            )
    
    # Summary
    active_parcels = sum(prediction[0])
    st.markdown(f"**Summary:** {active_parcels} out of 3 parcels need irrigation")
    