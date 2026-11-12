import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG, BATCH = 224, 32
model = tf.keras.models.load_model("models/material.keras")
classes = json.load(open("models/material_classes.json"))

df = pd.read_csv("dataset_split.csv")
df = df[df["split"] == "test"].reset_index(drop=True)
df["label"] = df["material"].map({c: i for i, c in enumerate(classes)})


def load(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG))


ds = tf.data.Dataset.from_tensor_slices(df["path"].values)
ds = ds.map(load).batch(BATCH)

probs = model.predict(ds)
df["pred"] = probs.argmax(axis=1)
df["conf"] = probs.max(axis=1)

print(classification_report(df["label"], df["pred"], target_names=classes))
print(confusion_matrix(df["label"], df["pred"]))

# Shortcut check: NEU vs MVTec metal alag-alag
metal = df[df["material"] == "metal"]
for src in ["neu", "mvtec"]:
    sub = metal[metal["source"] == src]
    acc = (sub["label"] == sub["pred"]).mean()
    print(f"Metal ({src}): {acc:.2%} sahi ({len(sub)} images)")
print("Average confidence:", df["conf"].mean())