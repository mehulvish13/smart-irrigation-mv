
# 🌾 Smart Irrigation System using Machine Learning

## Overview

This project implements a **Smart Irrigation System** using **Machine Learning** to automate irrigation decisions based on real-time agricultural data. The system predicts the optimal irrigation mode by analyzing multiple environmental factors, helping conserve water and improve crop productivity.

---

## Key Features

- 🌱 **Data-Driven Irrigation Decisions**  
  Uses sensor data like soil moisture, temperature, and humidity to automate irrigation.

- 🤖 **Machine Learning Model**  
  Trains a **Random Forest Classifier** wrapped in a **MultiOutputClassifier** for multi-label prediction of irrigation modes.

- 📈 **Data Preprocessing & Normalization**  
  Uses **MinMaxScaler** to normalize features for better model performance.

- � **Interactive Web Application**  
  Streamlit-based web interface for real-time irrigation predictions with accuracy metrics display.

- �💾 **Model Saving and Reusability**  
  Exports trained models using `joblib` for later deployment or inference.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python     | Core Language |
| Scikit-learn | ML Modeling |
| Pandas     | Data Manipulation |
| Matplotlib/Seaborn | Data Visualization |
| Streamlit  | Web Application Interface |
| Joblib     | Model Persistence |

---

## Project Workflow

1. **Data Loading**  
   Dataset: `irrigation_machine.csv`

2. **Preprocessing**  
   - Feature Scaling using `MinMaxScaler`
   - Handling multi-output labels

3. **Model Training**  
   - `RandomForestClassifier` wrapped with `MultiOutputClassifier`

4. **Evaluation**  
   - Classification report generation
   - Visualization of feature importance

5. **Model Saving**  
   - Trained models saved using `joblib`

---

## Folder Structure

```
Irrigation_System/
│
├── irrigstion_SYS_mv.ipynb    # Main Jupyter Notebook
├── irrigation_machine.csv     # Input Dataset (Required)
├── Farm_Irrigation_System.pkl # Trained Model File
├── app.py                     # Streamlit Web App
├── README.md                  # Project Documentation
```

---

## How to Run

1. **Install dependencies**

```bash
pip install -r requirements.txt
```


2. **Run the notebook**

```bash
jupyter notebook irrigstion_SYS_mv.ipynb
```

3. **Train the model**  
   Follow the notebook steps to train and save the model as `Farm_Irrigation_System.pkl`.

4. **Run the Streamlit Web App**

```bash
streamlit run app.py
```

This will launch a web interface where you can:
- Input sensor values using interactive sliders
- Get irrigation status predictions for each parcel
- View model accuracy metrics in the sidebar
- See detailed performance information for each parcel

## App Features

The Streamlit web application includes:
- **Interactive Sensor Input**: 20 sensor value sliders for real-time input
- **Prediction Display**: Visual status indicators for each of the 3 parcels
- **Model Performance**: Sidebar showing accuracy metrics for each parcel
- **User-Friendly Interface**: Clean, organized layout with helpful indicators

---

## Future Improvements

- Integration with IoT devices for real-time sensor data.
- Deploying the model as a web API using Flask/FastAPI.
- Adding cloud storage support for large-scale farming.

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Mehul Vishwakarma**  
AI/ML Enthusiast | Hackathon Developer | Community Builder  
📧 mehulvinodv@gmail.com | 🌐 [LinkedIn](https://www.linkedin.com/in/mehulvinodv)
