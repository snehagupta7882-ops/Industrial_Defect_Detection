import json
from pathlib import Path
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/material.keras")
classes = json.load(open("models/material_classes.json"))

for p in sorted(Path("my_photos").glob("*")):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    img = tf.io.read_file(str(p))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224))
    probs = model.predict(img[None], verbose=0)[0]
    top = probs.argmax()
    print(f"{p.name}: {classes[top]} ({probs[top]:.0%})  | all: "
          + ", ".join(f"{c}={v:.0%}" for c, v in zip(classes, probs)))