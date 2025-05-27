from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__, template_folder="templates")

# ✅ Load trained LSTM model
try:
    model = load_model("fish_harvest_lstm_optimized.h5")
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"🚨 Model Loading Failed: {e}")
    exit()

# ✅ Load dataset to fit scalers
dataset_path = "/Users/priyansh18/Desktop/farmhelp/aquaponics/lstm/time1.csv"
try:
    df = pd.read_csv(dataset_path)  
except FileNotFoundError:
    print(f"🚨 Error: Dataset '{dataset_path}' not found!")
    exit()

# ✅ Define feature columns (EXCLUDING 'Estimated Final Weight (kg)')
feature_columns = [
    'Age (Weeks)', 'Current Weight (kg)', 'Feed Consumption (g/day)', 
    'Water Temperature (°C)', 'Dissolved Oxygen (mg/L)', 'Water pH', 
    'Stocking Density (fish/m³)', 'Market Price (₹/kg)'
]
target_column = 'Estimated Final Weight (kg)'

# ✅ Initialize separate scalers for input (X) and output (y)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# ✅ Fit scalers only on training data (not full dataset)
scaler_X.fit(df[feature_columns])  
scaler_y.fit(df[[target_column]])  

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("🔍 Received Data:", data)

        if not data:
            return jsonify({"error": "No data received"}), 400

        # ✅ Extract 8 input features (DO NOT include 'Estimated Final Weight')
        try:
            features = np.array([
                float(data.get('Age (Weeks)', 0)),
                float(data.get('Current Weight (kg)', 0)),
                float(data.get('Feed Consumption (g/day)', 0)),
                float(data.get('Water Temperature (°C)', 0)),
                float(data.get('Dissolved Oxygen (mg/L)', 0)),
                float(data.get('Water pH', 0)),
                float(data.get('Stocking Density (fish/m³)', 0)),
                float(data.get('Market Price (₹/kg)', 0))
            ]).reshape(1, -1)  # Reshape for scaler
        except ValueError:
            return jsonify({"error": "Invalid input values"}), 400

        print("✅ Processed Features:", features)

        # ✅ Normalize the input features
        features_scaled = scaler_X.transform(features)

        # ✅ Convert input into LSTM shape (sequence_length = 20, features = 8)
        sequence_length = 20
        X_input = np.tile(features_scaled, (sequence_length, 1))  # Repeat input 20 times
        X_input = np.expand_dims(X_input, axis=0)  # Shape (1, 20, 8)

        print("📊 Input Shape for LSTM:", X_input.shape)

        # ✅ Make Prediction
        predicted_weight_scaled = model.predict(X_input)

        # ✅ Convert predicted weight back to original scale
        predicted_weight = scaler_y.inverse_transform(predicted_weight_scaled)[0][0]

        print("🎯 Predicted Weight:", predicted_weight)

        # ✅ Best Harvest Week Calculation
        growth_rate = float(data.get("Feed Consumption (g/day)", 1)) / 1000  # Convert g to kg/day
        harvest_week = int(data['Age (Weeks)']) + max(1, int(predicted_weight / (growth_rate + 1e-5)))

        # ✅ Calculate Expected Profit
        profit = predicted_weight * float(data['Market Price (₹/kg)'])

        # ✅ Convert values to native Python types
        response_data = {
            "Predicted Weight (kg)": float(round(predicted_weight, 2)),
            "Best Harvest Week": int(harvest_week),
            "Expected Profit (₹)": float(round(profit, 2))
        }

        print("✅ API Response:", response_data)
        return jsonify(response_data)

    except Exception as e:
        print("🚨 Error in Prediction:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)