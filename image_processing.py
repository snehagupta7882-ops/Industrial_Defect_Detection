import cv2
import numpy as np


# Function 1: Convert image bytes into OpenCV image

def load_image(image_bytes):

    numpy_image = np.frombuffer(
        image_bytes,
        np.uint8
    )

    image = cv2.imdecode(
        numpy_image,
        cv2.IMREAD_COLOR
    )

    return image


# Function 2: Convert BGR to RGB

def convert_to_rgb(image):

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return rgb_image


# Function 3: Resize image

def resize_image(image, size=(224, 224)):

    resized_image = cv2.resize(
        image,
        size
    )

    return resized_image


# Function 4: Normalize image

def normalize_image(image):

    normalized_image = image.astype(
        np.float32
    ) / 255.0

    return normalized_image

# Function 5: Preprocess for MobileNetV2 (range -1 to 1)

def preprocess_for_model(image):
    image = image.astype(np.float32)
    return (image / 127.5) - 1.0