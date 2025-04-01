from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Set maximum upload size to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

@app.errorhandler(413)
def request_entity_too_large(error):
    return "File is too large. Maximum upload size is 50MB.", 413

# Load pre-trained model
MODEL_PATH = "crop_disease_model.h5"
if not os.path.exists(MODEL_PATH):
    tf.keras.utils.get_file(
        "crop_disease_model.h5",
        "https://github.com/AshishSalaskar1/Plant-Leaf-Disease-Detection/raw/master/model.h5",
        cache_subdir=os.path.abspath("."), 
        cache_dir=".",
    )

# Load the model
model = load_model(MODEL_PATH, compile=False)

# Define class labels and descriptions (modify as per your dataset)
class_labels = ["Healthy", "Leaf Spot", "Blight", "Powdery Mildew", "Rust"]
disease_descriptions = {
    "Healthy": "Ang tanom maayo kag wala sang bisan anuman nga sakit.",
    "Leaf Spot": (
        "Ang tanom may leaf spot disease nga ginhalinan sang fungal ukon bacterial infections. "
        "Kuhaa ang mga apektado nga dahon kag gamita ang fungicide kung kinahanglan. "
        "Siguraduhon nga may maayo nga drainage kag indi mag-overwater ang tanom."
    ),
    "Blight": (
        "Ang tanom nagapakita sang blight, nga nagadulot sang madali nga pagkabrown kag pagkapatay sang mga dahon. "
        "Kuhaa ang mga apektado nga bahin kag gamita ang angay nga fungicides. "
        "Iwasan ang pagtanom sang tanom nga masyado nga maghuot kag siguraduhon nga maayo ang airflow."
    ),
    "Powdery Mildew": (
        "Ang tanom may powdery mildew, isa ka fungal disease nga nagapakita bilang puti nga pulbos sa mga dahon. "
        "Pasanyoga ang pag-agos sang hangin kag gamita ang fungicide. "
        "Iwasan ang pag-spray sang tubig sa mga dahon kag siguraduhon nga may husto nga distansya ang mga tanom."
    ),
    "Rust": (
        "Ang tanom may rust disease, nga nagapakita bilang mga orange ukon pula nga spots sa mga dahon. "
        "Kuhaa ang mga apektado nga dahon kag gamita ang fungicide. "
        "Siguraduhon nga indi mag-overwater ang tanom kag gamita ang resistant nga klase sang tanom kung mahimo."
    )
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        
        file_path = os.path.join("static", file.filename)
        file.save(file_path)
        img = image.load_img(file_path, target_size=(224, 224))
    elif 'image' in request.form:
        # Handle image from camera (base64 encoded)
        image_data = request.form['image']
        image_data = image_data.split(",")[1]  # Remove the data:image/...;base64, prefix
        image_bytes = BytesIO(base64.b64decode(image_data))
        img = Image.open(image_bytes).resize((224, 224))
    else:
        return "No image provided", 400

    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Debugging: Print input shape and model input shape
    print("Input image shape:", img_array.shape)
    print("Model input shape:", model.input_shape)

    # Make prediction
    prediction = model.predict(img_array)

    # Debugging: Print prediction output
    print("Prediction output:", prediction)
    print("Prediction shape:", prediction.shape)

    # Handle invalid predictions
    if prediction.size == 0:
        return "Model did not return a valid prediction", 500

    # Get the predicted label and description
    predicted_label = class_labels[np.argmax(prediction)]
    disease_description = disease_descriptions[predicted_label]

    return render_template(
        'index.html',
        prediction=predicted_label,
        description=disease_description,
        img_path=file_path if 'file' in request.files else None
    )

if __name__ == '__main__':
    # Ensure the static folder exists for saving uploaded files
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True)