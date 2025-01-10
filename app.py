from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Set a secret key for session management (for example, 16-byte hex)
app.secret_key = os.urandom(24)

# Local path for the model file
MODEL_PATH = 'my_model.keras'  # Path to the model .keras file

# Global variable to store the model after loading
model = None

# Function to load the model (only once when the app starts)
def load_model_once():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            try:
                # Load the model directly from the .keras file
                model = tf.keras.models.load_model(MODEL_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                model = None
        else:
            print(f"Model file '{MODEL_PATH}' not found.")
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

        # Resize the image to match model input
        image = image.resize((180, 180))  # Resize image to 224x224

        img_array = np.array(image)

        # Normalize to [0, 1] and ensure the correct shape
        img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

        # Make predictions
        predictions = model.predict(img_array)
        classNames = ['Nazli', 'Buzgulu', 'Ak', 'Dimnit', 'Ala_Idris']
        predicted_class = classNames[np.argmax(predictions)]

        return render_template('index.html', prediction=predicted_class, image=image_filename)

    except Exception as e:
        flash(f"Error in classifying the image: {e}", 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
