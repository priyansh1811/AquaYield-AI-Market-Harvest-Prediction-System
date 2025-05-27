import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ✅ Load the time-series dataset
df = pd.read_csv("/Users/priyansh18/Desktop/farmhelp/aquaponics/lstm/time.csv")

# ✅ Convert categorical variables to numerical values
df['Fish Species'] = df['Fish Species'].astype('category').cat.codes  # Convert species to numerical codes
df['Feed Type'] = df['Feed Type'].astype('category').cat.codes  # Convert feed type to numerical codes

# ✅ Define feature columns (EXCLUDING 'Estimated Final Weight (kg)')
feature_columns = [
    'Age (Weeks)', 'Current Weight (kg)', 'Feed Consumption (g/day)', 
    'Water Temperature (°C)', 'Dissolved Oxygen (mg/L)', 'Water pH', 
    'Stocking Density (fish/m³)', 'Market Price (₹/kg)'
]
target_column = 'Estimated Final Weight (kg)'

# ✅ Normalize numerical columns for better LSTM performance
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df[feature_columns] = scaler_X.fit_transform(df[feature_columns])  # Scale input features
df[target_column] = scaler_y.fit_transform(df[[target_column]])  # Scale target separately

# ✅ Sequence Length for Time-Series Data
sequence_length = 20  # Increased from 10 to 20 weeks

X, y = [], []
for fish_id in df['Fish ID'].unique():
    fish_data = df[df['Fish ID'] == fish_id].sort_values('Age (Weeks)')
    for i in range(len(fish_data) - sequence_length):
        X.append(fish_data.iloc[i:i+sequence_length][feature_columns].values)  # Only 8 features
        y.append(fish_data.iloc[i+sequence_length][target_column])  # Target is separate

# ✅ Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# ✅ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Define Optimized LSTM Model
model = Sequential([
    Bidirectional(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
    Dropout(0.4),
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.4),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(100, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dense(1)  # Output: Predicted Final Fish Weight
])

# ✅ Exponential Learning Rate Decay
initial_learning_rate = 0.001
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=5000,
    decay_rate=0.9,
    staircase=True
)

# ✅ Compile with AdamW Optimizer & Learning Rate Scheduling
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule),
    loss='mean_squared_error',
    metrics=['mae']
)

# ✅ Increase Training Time
epochs = 200  # Increased for better learning
batch_size = 8  # Reduced batch size for better weight updates

# ✅ Train the Model
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1
)

# ✅ Save the Trained Model
model.save("fish_harvest_lstm_model.h5")
print("🎉 Model training complete! Saved as 'fish_harvest_lstm_model.h5'.")

# =====================================================
# ✅ Load & Test the Model
# =====================================================

# ✅ Load trained model
model = load_model("fish_harvest_lstm_model.h5")

# ✅ Make Predictions
y_pred = model.predict(X_test)

# ✅ Convert Predictions and Actual Values Back to Original Scale
y_pred_original = scaler_y.inverse_transform(y_pred)  # Convert predicted values back
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # Convert actual values back

# ✅ Evaluate Model Performance
mae = mean_absolute_error(y_test_original, y_pred_original)
r2 = r2_score(y_test_original, y_pred_original)

# ✅ Create Results DataFrame
results_df = pd.DataFrame({
    'Actual Weight (kg)': y_test_original.flatten(),
    'Predicted Weight (kg)': y_pred_original.flatten()
})

# ✅ Save Predictions to CSV
results_df.to_csv("lstm_fish_growth_predictions.csv", index=False)

# ✅ Print Model Accuracy Metrics
print("\n✅ Model Evaluation Completed!")
print(f"📌 Mean Absolute Error (MAE): {mae:.3f}")
print(f"📌 R² Score: {r2:.3f}")
print(f"📂 Predictions saved to 'lstm_fish_growth_predictions.csv'.")
