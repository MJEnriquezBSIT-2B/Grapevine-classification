from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Set a secret key for session management (for example, 16-byte hex)
app.secret_key = os.urandom(24)

# Local paths for the model files
MODEL_JSON_PATH = 'model.json'  # Path to the model JSON file
MODEL_H5_PATH = 'model.h5'      # Path to the model weights file

# Function to load the model
def load_model():
    if os.path.exists(MODEL_JSON_PATH) and os.path.exists(MODEL_H5_PATH):
        try:
            # Load the model architecture from the JSON file
            with open(MODEL_JSON_PATH, "r") as json_file:
                model_json = json_file.read()
                model = model_from_json(model_json)

            # Load the model weights from the .h5 file
            model.load_weights(MODEL_H5_PATH)
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("Model files not found.")
        return None


# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


# Route for image upload and classification
@app.route('/classify', methods=['POST'])
def classify_image():
    model = load_model()

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
        img_array = np.array(image)

        img_array = tf.image.resize(img_array, [224, 224])  # Resize image
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        classNames = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']
        predicted_class = classNames[np.argmax(predictions)]

        return render_template('result.html', prediction=predicted_class, image=image_filename)

    except Exception as e:
        flash(f"Error in classifying the image: {e}", 'error')
        return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
