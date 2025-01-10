from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.urandom(24)

# Path to the Keras model file
MODEL_KERAS_PATH = 'model2.h5'  # Path to the saved Keras model file

# Global variable to store the model after loading
model = None
class_names = None  # Replace with your class names if available, e.g., ["class1", "class2"]

# Define a confidence threshold for classification accuracy (e.g., 0.7)
CONFIDENCE_THRESHOLD = 0.7

# Function to load the model (only once when the app starts)
def load_model_once():
    global model
    if model is None:
        if os.path.exists(MODEL_KERAS_PATH):
            try:
                model = load_model(MODEL_KERAS_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
        else:
            print(f"Model file '{MODEL_KERAS_PATH}' not found.")
    return model

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for image upload and classification
@app.route('/classify', methods=['POST'])
def classify_image():
    model = load_model_once()

    if model is None:
        flash("Model could not be loaded. Please check the logs for more details.", 'error')
        return redirect(url_for('index'))

    if 'image' not in request.files or request.files['image'].filename == '':
        flash("No file uploaded or file not selected.", 'error')
        return redirect(url_for('index'))

    file = request.files['image']
    try:
        # Save the uploaded image
        image_filename = file.filename
        image_path = os.path.join('static', image_filename)
        file.save(image_path)

        # Process the image
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format

        # Resize the image to match model input (e.g., 150x150)
        image = image.resize((150, 150))

        img_array = np.array(image)

        # Normalize to [0, 1] and ensure the correct shape
        img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 150, 150, 3)

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Placeholder class names (Replace with actual class names if available)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(predictions.shape[1])]

        # Get the predicted class name
        predicted_class = class_names[predicted_class_index]

        # Check prediction confidence
        confidence = predictions[0][predicted_class_index]

        # If confidence is below threshold, show a message
        if confidence < CONFIDENCE_THRESHOLD:
            prediction_message = f"Prediction: {predicted_class} (Low confidence)"
        else:
            prediction_message = f"Prediction: {predicted_class} (Confidence: {confidence*100:.2f}%)"

        return render_template('index.html', prediction=prediction_message, image_filename=image_filename)

    except Exception as e:
        flash(f"Error in classifying the image: {e}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
