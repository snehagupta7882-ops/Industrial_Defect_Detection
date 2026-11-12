import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

IMG, BATCH = 224, 32
model = keras.models.load_model("models/condition.keras")
classes = json.load(open("models/condition_classes.json"))   # ["defective", "normal"]

df = pd.read_csv("dataset_split.csv")
df = df[(df["split"] == "test") & (df["source"] == "mvtec")].reset_index(drop=True)
df["label"] = df["condition"].map({c: i for i, c in enumerate(classes)})


def load(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG))


ds = tf.data.Dataset.from_tensor_slices(df["path"].values).map(load).batch(BATCH)
probs = model.predict(ds)
df["pred"] = probs.argmax(axis=1)

print(classification_report(df["label"], df["pred"], target_names=classes))
print(confusion_matrix(df["label"], df["pred"]))

print("\nPer-material recall (defective):")
for m in sorted(df["material"].unique()):
    sub = df[(df["material"] == m) & (df["label"] == 0)]
    print(f"  {m}: {(sub['pred'] == 0).mean():.0%} ({len(sub)} defective images)")