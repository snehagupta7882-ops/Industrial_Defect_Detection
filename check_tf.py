import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

model = keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
print("MobileNetV2 loaded. Layers:", len(model.layers))
