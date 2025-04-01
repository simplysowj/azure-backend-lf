from google.cloud import vision

def detect_labels(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.label_detection(image=image)

    labels = response.label_annotations
    print("Detected Labels:")
    for label in labels:
        print(f"{label.description}: {label.score:.2f}")

# Replace 'image.jpg' with the path of an image file
detect_labels(r"C:\Users\simpl\Downloads\location_finder\backend_locationfinder\taking-selfie-golden-gate-bridge-glden-lookout-san-francisco-california-117087125.webp")
from google.cloud import vision

def detect_landmarks(image_path):
    """Detects landmarks in an image and returns their latitude and longitude."""
    client = vision.ImageAnnotatorClient()

    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations

    if landmarks:
        for landmark in landmarks:
            print(f"Landmark: {landmark.description}")
            for location in landmark.locations:
                lat, lng = location.lat_lng.latitude, location.lat_lng.longitude
                print(f"Latitude: {lat}, Longitude: {lng}")
    else:
        print("No landmarks detected.")

# Replace 'bridge.jpg' with your image file
detect_landmarks(r"C:\Users\simpl\Downloads\location_finder\backend_locationfinder\taking-selfie-golden-gate-bridge-glden-lookout-san-francisco-california-117087125.webp")
