import json
import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras

IMG, BATCH = 224, 32
TARGET_RECALL = 0.90

model = keras.models.load_model("models/condition_ft.keras")
classes = json.load(open("models/condition_classes.json"))   # ["defective", "normal"]
DEF = classes.index("defective")

df_all = pd.read_csv("dataset_split.csv")
df_all = df_all[df_all["source"] == "mvtec"]


def load(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG))


def get_probs(split):
    d = df_all[df_all["split"] == split].reset_index(drop=True)
    ds = tf.data.Dataset.from_tensor_slices(d["path"].values).map(load).batch(BATCH)
    p = model.predict(ds, verbose=0)[:, DEF]          # defective ki probability
    y = (d["condition"] == "defective").values
    return p, y


def metrics(p, y, t):
    pred = p >= t
    tp = int((pred & y).sum())
    fn = int((~pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return recall, precision, tp, fn, fp, tn


p_val, y_val = get_probs("val")
p_test, y_test = get_probs("test")

print("VALIDATION: threshold sweep")
print("thr   recall  precision")
best = None
for t in np.arange(0.05, 0.96, 0.05):
    r, pr, *_ = metrics(p_val, y_val, t)
    print(f"{t:.2f}   {r:.0%}     {pr:.0%}")
    if r >= TARGET_RECALL:
        best = t          # sabse bada threshold jahan recall target se upar hai

if best is None:
    best = 0.05
    print("Target recall nahi mila, sabse kam threshold use ho raha hai.")

print(f"\nChosen threshold (validation se): {best:.2f}")

for name, t in [("Default 0.50", 0.50), (f"Tuned {best:.2f}", best)]:
    r, pr, tp, fn, fp, tn = metrics(p_test, y_test, t)
    print(f"\nTEST with {name}")
    print(f"  Defective recall: {r:.0%}  precision: {pr:.0%}")
    print(f"  Defective pakde: {tp}, chhute: {fn}")
    print(f"  Normal sahi: {tn}, galat reject: {fp}")

with open("models/condition_ft_threshold.json", "w") as f:
    json.dump({"defective_threshold": float(best)}, f)
print("\nSaved models/condition_ft_threshold.json")