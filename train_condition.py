import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

keras = tf.keras

IMG, BATCH, SEED = 224, 32, 42

df = pd.read_csv("dataset_split.csv")
df = df[df["source"] == "mvtec"].copy()          # NEU hata diya
classes = ["defective", "normal"]
df["label"] = df["condition"].map({c: i for i, c in enumerate(classes)})
print(df.groupby(["split", "condition"]).size())


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
print("Class weights:", class_weight)

augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.04),
    keras.layers.RandomZoom(0.1),
])

base = keras.applications.MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG, IMG, 3))
base.trainable = False

inputs = keras.Input(shape=(IMG, IMG, 3))
x = augment(inputs)
x = keras.layers.Rescaling(1 / 127.5, offset=-1)(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=val_ds, epochs=10, class_weight=class_weight,
          callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

model.save("models/condition.keras")
json.dump(classes, open("models/condition_classes.json", "w"))
print("Saved models/condition.keras")