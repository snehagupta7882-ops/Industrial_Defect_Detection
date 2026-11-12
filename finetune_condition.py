import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight

IMG, BATCH, SEED = 224, 32, 42

df = pd.read_csv("dataset_split.csv")
df = df[df["source"] == "mvtec"].copy()
classes = json.load(open("models/condition_classes.json"))   # ["defective", "normal"]
df["label"] = df["condition"].map({c: i for i, c in enumerate(classes)})


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG)), label


def make_ds(split, training):
    d = df[df["split"] == split]
    ds = tf.data.Dataset.from_tensor_slices((d["path"].values, d["label"].values))
    if training:
        ds = ds.shuffle(len(d), seed=SEED)
    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds, val_ds = make_ds("train", True), make_ds("val", False)

y = df[df["split"] == "train"]["label"].values
w = compute_class_weight("balanced", classes=np.arange(2), y=y)
class_weight = {i: float(x) for i, x in enumerate(w)}

model = keras.models.load_model("models/condition.keras")

# MobileNetV2 wala nested model dhoondho
base = [l for l in model.layers if isinstance(l, keras.Model)][0]
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
# BatchNorm frozen rakho (chhote data par stable rehta hai)
for layer in base.layers:
    if isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weight,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

model.save("models/condition_ft.keras")
print("Saved models/condition_ft.keras")