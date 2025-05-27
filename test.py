from tensorflow.keras.utils import plot_model
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("/Users/priyansh18/Desktop/farmhelp/aquaponics/lstm/fish_harvest_lstm_optimized.h5")

# Save the model architecture as an image
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

print("✅ Model architecture image saved as 'model_architecture.png'")