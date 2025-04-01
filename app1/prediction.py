import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Print TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Load the model once
model = load_model(r'C:\Users\simpl\Downloads\location_finder\backend_locationfinder\location_classifier_v2.keras')

# Label mapping
label_mapping = {
    0: 'eiffel_tower',
    1: 'grand_canyon',
    2: 'burj_khalifa',
    3: 'pyramids_of_giza',
    4: 'stonehenge',
    5: 'chichen_itza',
    6: 'great_wall_of_china',
    7: 'roman_colosseum',
    8: 'taj_mahal',
    9: 'christ_the_redeemer',
    10: 'machu_picchu',
    11: 'statue_of_liberty',
    12: 'venezuela_angel_falls'
}

def predict_location(image_path):
    """Process image and predict location"""
    # Open the image file
    with open(image_path, 'rb') as image_file:
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Preprocess the image
        img = cv2.resize(img, (224, 224)) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img)
        location_index = np.argmax(prediction)

        # Get location label
        location = label_mapping.get(location_index, "Unknown Location")

        return location

# Call the function with the correct image path
image_path = r"C:\Users\simpl\Downloads\location_finder\backend_locationfinder\tajmahal.jpg"
print(predict_location(image_path))
