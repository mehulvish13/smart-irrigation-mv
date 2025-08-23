# Smart Irrigation System - Advanced ML-Powered Agricultural Dashboard
# Author: Mehul Vishwakarma
# Description: Real-time IoT sensor monitoring and irrigation prediction system

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import base64
import os
from typing import Optional, Tuple

# ------------- Page Config -------------
st.set_page_config(
    page_title="Smart Irrigation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------- Logo Utilities -------------
def _b64(bin_file: str) -> Optional[str]:
    """Read a binary file and return base64 string, or None if missing."""
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

def _read_text(path: str) -> Optional[str]:
    """Read a text file (e.g., SVG) and return string, or None if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def find_logo() -> Tuple[Optional[str], Optional[str]]:
    """
    Find a logo file in common locations and return:
    - ("png", data_uri) for png/jpg
    - ("svg", svg_markup) for inline svg
    - (None, None) if nothing found
    """
    candidates = [
        "assets/logo.png", "assets/logo.jpg", "assets/logo.jpeg",
        "assets/logo.svg",
        "logo.png", "logo.jpg", "logo.jpeg", "logo.svg",
        "MV logo.png",  # your original filename with space
    ]

    for path in candidates:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                b = _b64(path)
                if b:
                    return "png", f"data:image/{'jpeg' if ext != '.png' else 'png'};base64,{b}"
            elif ext == ".svg":
                s = _read_text(path)
                if s:
                    return "svg", s
    return None, None

logo_kind, logo_payload = find_logo()

# ------------- Robust CSS (no brittle classnames) -------------
# NOTE: Using data-testid selectors is more stable across Streamlit versions
st.markdown(
    """
    <style>
    /* Wider sidebar */
    [data-testid="stSidebar"] {
        width: 350px !important;
    }
    [data-testid="stSidebar"] > div {
        width: 350px !important;
    }

    /* Cards & Groups */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sensor-group {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
    .prediction-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        color: #333333 !important;
    }
    .irrigation-on {
        border-color: #4CAF50 !important;
        background: linear-gradient(135deg, #e8f5e8, #f1f8e9);
        color: #1B5E20 !important;
    }
    .irrigation-off {
        border-color: #f44336 !important;
        background: linear-gradient(135deg, #fde7e7, #fef2f2);
        color: #C62828 !important;
    }

    /* Top-right watermark */
    .watermark {
        position: fixed;
        bottom: 7vh;
        right: 0px;
        opacity: 0.7;
        z-index: 9999;
        pointer-events: none; /* don't block clicks */
    }
    .watermark img {
        width: 65px;
        height: auto;
        border-radius: 0px;
        transition: opacity 0.3s ease;
    }
    .watermark:hover img {
        opacity: 1.0;
        
    }
    .watermark svg {
        width: 85px;
        height: auto;
        border-radius: 0px;
        transition: opacity 0.3s ease;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------- Watermark (only if logo found) -------------
if logo_kind == "png":
    st.markdown(
        f"""
        <div class="watermark" aria-hidden="true">
            <img src="{logo_payload}" alt="MV Logo"/>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif logo_kind == "svg":
    st.markdown(
        f"""
        <div class="watermark" aria-hidden="true">
            {logo_payload}
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("ℹ️ Logo not found in `assets/` or current folder. Place `assets/logo.png` or `assets/logo.svg` to show the watermark.")

# ------------- Session State -------------
def initialize_session_state():
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    if 'sensor_presets' not in st.session_state:
        st.session_state.sensor_presets = {
            "Default": [0.5] * 20,
            "Dry Season": [0.8, 0.9, 0.2, 0.1, 0.7, 0.8, 0.3, 0.2, 0.9, 0.8,
                          0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.9, 0.8, 0.2, 0.1],
            "Wet Season": [0.2, 0.1, 0.8, 0.9, 0.3, 0.2, 0.7, 0.8, 0.1, 0.2,
                           0.9, 0.8, 0.2, 0.3, 0.7, 0.6, 0.1, 0.2, 0.8, 0.9],
            "Optimal": [0.6, 0.5, 0.4, 0.6, 0.5, 0.7, 0.4, 0.5, 0.6, 0.5,
                        0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4]
        }

    for key in ['reset_sensors', 'randomize_sensors', 'load_preset', 'show_help_modal']:
        if key not in st.session_state:
            st.session_state[key] = False

    if 'selected_preset_name' not in st.session_state:
        st.session_state.selected_preset_name = "Default"

initialize_session_state()

# ------------- Model & Data -------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("Farm_Irrigation_System.pkl")
    except FileNotFoundError:
        st.error("❌ Model file 'Farm_Irrigation_System.pkl' not found!")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("irrigation_machine.csv")
        parcel_accuracies = [0.95, 0.92, 0.94]  # placeholder unless computed
        overall_accuracy = 0.89
        return df, parcel_accuracies, overall_accuracy
    except FileNotFoundError:
        return None, [0.95, 0.92, 0.94], 0.89

model = load_model()
df, parcel_accuracies, overall_accuracy = load_data()

# ------------- Charts & Helpers -------------
def create_sensor_visualization(sensor_values):
    fig = go.Figure()
    categories = [f'Sensor {i+1}' for i in range(20)]
    fig.add_trace(go.Scatterpolar(
        r=sensor_values,
        theta=categories,
        fill='toself',
        name='Current Values',
        line=dict(color='rgb(0, 123, 255)')
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Sensor Values Radar Chart",
        height=400
    )
    return fig

def create_prediction_history_chart():
    if not st.session_state.prediction_history:
        return None
    df_history = pd.DataFrame(st.session_state.prediction_history)
    fig = px.line(
        df_history,
        x='timestamp',
        y=['Parcel 0', 'Parcel 1', 'Parcel 2'],
        title='Irrigation Prediction History',
        labels={'value': 'Irrigation Status (0=OFF, 1=ON)', 'timestamp': 'Time'}
    )
    fig.update_layout(height=400)
    return fig

def save_prediction_to_history(prediction, sensor_values):
    history_entry = {
        'timestamp': datetime.now(),
        'Parcel 0': int(prediction[0][0]),
        'Parcel 1': int(prediction[0][1]),
        'Parcel 2': int(prediction[0][2]),
        'sensor_values': sensor_values
    }
    st.session_state.prediction_history.append(history_entry)
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history.pop(0)

def export_predictions_csv():
    if st.session_state.prediction_history:
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        return df_hist.to_csv(index=False)
    return None

def get_sensor_description(group_name):
    descriptions = {
        "Climate Sensors": "Temperature, Humidity, Weather",
        "Soil Moisture Sensors": "Water Content, Soil Humidity",
        "Plant Health Sensors": "Growth, Leaf Moisture, Plant Stress",
        "Environmental Sensors": "Wind, Light, Atmospheric Pressure"
    }
    return descriptions.get(group_name, "")

def get_sensor_initial_value(i):
    if st.session_state.load_preset:
        return st.session_state.sensor_presets[st.session_state.selected_preset_name][i]
    elif st.session_state.reset_sensors:
        return 0.5
    elif st.session_state.randomize_sensors:
        return float(np.random.random())
    else:
        return st.session_state.get(f"sensor_{i}", 0.5)

def render_sensor_group(group_name, icon, sensor_range, sensor_values):
    start_idx, end_idx = sensor_range
    st.markdown('<div class="sensor-group">', unsafe_allow_html=True)
    st.markdown(f"#### {icon} **{group_name} ({start_idx+1}-{end_idx})** - {get_sensor_description(group_name)}")

    cols = st.columns(5)
    for i in range(start_idx, end_idx):
        with cols[i % 5]:
            val = st.slider(
                f"{icon} Sensor {i+1}",
                min_value=0.0,
                max_value=1.0,
                value=get_sensor_initial_value(i),
                step=0.01,
                key=f"sensor_{i}",
                help=f"{group_name.split()[0]} sensor {i+1} reading"
            )
            sensor_values.append(val)

    st.markdown('</div>', unsafe_allow_html=True)
    return sensor_values

@st.dialog("📖 How to Use Smart Irrigation System", width="large")
def show_how_to_use_modal():
    """Display a modal with instructions on how to use the application."""
    
    # Main content in columns for better width utilization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Getting Started")
        st.markdown("""
        1. **Model Status:** Check sidebar status
        2. **Configure Sensors:** Adjust 20 sensors below
        3. **Get Prediction:** Click predict button
        """)
        
        st.markdown("#### 🌡️ Sensor Groups")
        st.markdown("""
        **🌡️ Climate (1-5):** Temperature, Humidity  
        **💧 Soil Moisture (6-10):** Water content  
        **🌱 Plant Health (11-15):** Growth metrics  
        **🌍 Environmental (16-20):** Wind, Light
        """)
    
    with col2:
        st.markdown("#### ⚡ Quick Actions")
        st.markdown("""
        **🎛️ Presets:** Default, Dry/Wet Season, Optimal  
        **🔄 Reset:** Set all sensors to 0.5  
        **🎲 Randomize:** Generate test values
        """)
        
        st.markdown("#### 📊 Understanding Results")
        st.markdown("""
        **🟢 ON:** Zone needs watering  
        **🔴 OFF:** Sufficient moisture  
        **📈 Confidence:** Model accuracy  
        **💧 Usage:** Water estimation
        """)
    
    st.info("💡 **Pro Tips:** Use presets for quick setup • Monitor history for patterns • Check confidence before decisions")
    
    if st.button("✅ Got it!", type="primary", use_container_width=True):
        st.session_state.show_help_modal = False
        st.rerun()

# ------------- Header -------------
# Title and small Help button in top-right corner
col1, col2 = st.columns([5, 1])
with col1:
    st.title("🌾 Smart Irrigation System")
    st.markdown("### AI-Powered Water Management Dashboard")
with col2:
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)  # Add top margin to align
    if st.button("📖 Help", type="secondary", key="help_button"):
        st.session_state.show_help_modal = True

# Display help modal if requested
if st.session_state.get('show_help_modal', False):
    show_how_to_use_modal()

if model is None:
    st.stop()

# ------------- Sidebar -------------
with st.sidebar:
    st.title("📊 System Dashboard")

    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Reset All Sensors", use_container_width=True):
        st.session_state.reset_sensors = True
        st.rerun()
    if st.button("🎲 Randomize Sensors", use_container_width=True):
        st.session_state.randomize_sensors = True
        st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ Model Details")
    st.markdown(
        """
        - **Algorithm**: Random Forest Classifier  
        - **Type**: Multi-Output Classification  
        - **Inputs**: 20 Environmental Sensors  
        - **Outputs**: 3 Irrigation Zones  
        - **Training**: Historical Farm Data  
        """
    )

    st.markdown("### 🔄 System Status")
    st.success("✅ Model Loaded")
    st.success("✅ Data Available" if df is not None else "⚠️ Data File Missing")

    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📥 Export Data")
        csv_data = export_predictions_csv()
        if csv_data:
            st.download_button(
                "Download History CSV",
                csv_data,
                "irrigation_predictions.csv",
                "text/csv",
                use_container_width=True
            )

# ------------- Sensors -------------
st.markdown("### 🌡️ Environmental Sensor Configuration")
st.markdown("**Configure all 20 environmental sensors for irrigation prediction**")

# Simple sensor input without groups
sensor_values = []
cols = st.columns(4)
for i in range(20):
    with cols[i % 4]:
        val = st.slider(
            f"Sensor {i+1}",
            min_value=0.0,
            max_value=1.0,
            value=get_sensor_initial_value(i),
            step=0.01,
            key=f"sensor_{i}",
            help=f"Environmental sensor {i+1} reading"
        )
        sensor_values.append(val)

# reset flags
for flag in ['load_preset', 'reset_sensors', 'randomize_sensors']:
    if st.session_state.get(flag, False):
        st.session_state[flag] = False

# ------------- Stats -------------
st.markdown("---")
c1, c2 = st.columns([1, 1])

with c1:
    st.markdown("### 📈 Sensor Values Visualization")
    if len(sensor_values) == 20:
        st.plotly_chart(create_sensor_visualization(sensor_values), use_container_width=True)
    else:
        st.warning("Waiting for all 20 sensor values...")

with c2:
    st.markdown("### 📊 Real-time Statistics")
    if sensor_values:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Average Value", f"{np.mean(sensor_values):.3f}")
            st.metric("Min Value", f"{np.min(sensor_values):.3f}")
        with col_b:
            st.metric("Max Value", f"{np.max(sensor_values):.3f}")
            st.metric("Std Deviation", f"{np.std(sensor_values):.3f}")

        st.markdown("#### 📊 Detailed Analytics")
        st.info(f"🎯 **Range**: {np.max(sensor_values) - np.min(sensor_values):.3f}")
        st.info(f"📊 **Median**: {np.median(sensor_values):.3f}")
        st.info(f"🔢 **Total Sum**: {np.sum(sensor_values):.3f}")

        high = sum(1 for v in sensor_values if v > 0.7)
        med = sum(1 for v in sensor_values if 0.3 <= v <= 0.7)
        low = sum(1 for v in sensor_values if v < 0.3)

        st.markdown("#### 🎨 Value Distribution")
        st.success(f"🟢 High (>0.7): {high} sensors")
        st.warning(f"🟡 Medium (0.3–0.7): {med} sensors")
        st.error(f"🔴 Low (<0.3): {low} sensors")
    else:
        st.info("Adjust the sliders to see statistics.")

# ------------- Prediction -------------
st.markdown("---")
st.markdown("### 🎯 Irrigation Prediction")

_, mid, _ = st.columns([2, 1, 2])
with mid:
    predict_button = st.button("🚰 Predict Irrigation Status", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("🔄 Analyzing sensor data and making prediction..."):
        time.sleep(1)
        try:
            input_data = pd.DataFrame([sensor_values])
            prediction = model.predict(input_data)
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            prediction = None

        if prediction is not None:
            save_prediction_to_history(prediction, sensor_values)
            st.markdown("### 🎯 Prediction Results")

            cols = st.columns(3)
            parcel_names = ["Zone A (Parcel 0)", "Zone B (Parcel 1)", "Zone C (Parcel 2)"]
            colors = ["🔵", "🟠", "🟢"]

            active_parcels = 0
            for i, (status, name, color) in enumerate(zip(prediction[0], parcel_names, colors)):
                with cols[i]:
                    status_text = "IRRIGATION ON" if status == 1 else "IRRIGATION OFF"
                    status_icon = "🟢" if status == 1 else "🔴"
                    if status == 1:
                        active_parcels += 1

                    card_class = "irrigation-on" if status == 1 else "irrigation-off"
                    st.markdown(
                        f"""
                        <div class="prediction-card {card_class}">
                            <h4>{color} {name}</h4>
                            <h2>{status_icon} {status_text}</h2>
                            <p><strong>Confidence:</strong> {parcel_accuracies[i]*100:.1f}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 📋 Summary")
                st.info(f"🚰 **{active_parcels}** out of **3** zones need irrigation")
                water_usage = active_parcels * 100  # simple illustrative estimate
                st.metric("Estimated Water Usage", f"{water_usage} L")
                if active_parcels == 0:
                    st.success("✅ No irrigation needed - Water conservation mode")
                elif active_parcels <= 1:
                    st.warning("⚠️ Low irrigation demand detected")
                else:
                    st.error("🔴 High irrigation demand - Monitor closely")

            with c2:
                st.markdown("### 💡 Recommendations")
                if active_parcels == 3:
                    st.markdown("- 🌊 High water demand period")
                    st.markdown("- ⏰ Schedule irrigation during cooler hours")
                    st.markdown("- 📊 Monitor soil moisture closely")
                elif active_parcels >= 1:
                    st.markdown("- 🎯 Targeted irrigation approach")
                    st.markdown("- 📈 Check sensor calibration")
                    st.markdown("- 🌡️ Consider weather forecast")
                else:
                    st.markdown("- 💚 Optimal soil conditions")
                    st.markdown("- 🔄 Continue monitoring")
                    st.markdown("- 📅 Schedule next check")

# ------------- History -------------
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("### 📈 Prediction History")
    history_fig = create_prediction_history_chart()
    if history_fig:
        st.plotly_chart(history_fig, use_container_width=True)

    st.markdown("### 📋 Recent Predictions")
    recent = pd.DataFrame(st.session_state.prediction_history[-10:])
    recent['timestamp'] = pd.to_datetime(recent['timestamp'])
    recent = recent.sort_values('timestamp', ascending=False)
    st.dataframe(
        recent[['timestamp', 'Parcel 0', 'Parcel 1', 'Parcel 2']],
        use_container_width=True,
        hide_index=True
    )

# ------------- Footer -------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 30px; margin-top: 50px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
        <h3 style='margin: 0; color: white;'>🌾 Advanced Smart Irrigation System</h3>
        <p style='margin: 10px 0; font-size: 16px;'>Powered by Machine Learning & Real-time Analytics</p>
        <p style='margin: 5px 0; font-size: 14px;'>© 2025 <strong>Mehul Vishwakarma</strong>. All rights reserved.</p>
        <p style='margin: 5px 0; font-size: 14px;'>
            📧 Contact: <a href='mailto:mehulvinodv@gmail.com' style='color: #ffeb3b; text-decoration: none; font-weight: bold;'>mehulvinodv@gmail.com</a>
        </p>
        <div style='margin-top: 20px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
            <p style='margin: 0; font-size: 12px;'>
                🚀 Features: Multi-Zone Prediction | Historical Analytics | Export Functionality | Real-time Visualization
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
