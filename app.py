from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Set a secret key for session management (for example, 16-byte hex)
app.secret_key = os.urandom(24)

# Paths to the model files
MODEL_JSON_PATH = 'my_model.json'  # Path to the model architecture (JSON file)
MODEL_H5_PATH = 'my_model.h5'      # Path to the model weights (H5 file)

# Global variable to store the model after loading
model = None

# Define a confidence threshold for classification accuracy (e.g., 0.7)
CONFIDENCE_THRESHOLD = 0.7

# Function to load the model (only once when the app starts)
def load_model_once():
    global model
    if model is None:
        if os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_H5_PATH):
            try:
                # Load the model architecture from the JSON file
                with open(MODEL_JSON_PATH, 'r') as json_file:
                    model_json = json_file.read()
                    model = model_from_json(model_json)

                # Load the weights into the model from the H5 file
                model.load_weights(MODEL_H5_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
        else:
            print(f"Model files '{MODEL_JSON_PATH}' or '{MODEL_H5_PATH}' not found.")
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

        # Resize the image to match model input (e.g., 180x180)
        image = image.resize((180, 180))  # Resize image to 180x180, as expected by the model

        img_array = np.array(image)

        # Normalize to [0, 1] and ensure the correct shape
        img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 180, 180, 3)

        # Make predictions
        predictions = model.predict(img_array)
        
        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # List of class names as expected
        classNames = ['Nazli', 'Buzgulu', 'Ak', 'Dimnit', 'Ala_Idris']

        # Check prediction confidence
        confidence = predictions[0][predicted_class_index]

        # If confidence is below threshold, return "Not available in dataset"
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_class = "Not available in dataset"
        else:
            predicted_class = classNames[predicted_class_index]

        return render_template('index.html', prediction=predicted_class, image_filename=image_filename)

    except Exception as e:
        flash(f"Error in classifying the image: {e}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
