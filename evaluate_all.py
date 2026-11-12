import json
import numpy as np
import pandas as pd

try:
    import tensorflow as tf  # type: ignore[import-not-found]
    from tensorflow import keras  # type: ignore[import-not-found]
except ImportError as exc:
    raise RuntimeError(
        "TensorFlow is required for this script. Please install it in the active Python environment."
    ) from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

IMG, BATCH = 224, 32

df = pd.read_csv("dataset_split.csv")
test = df[df["split"] == "test"].reset_index(drop=True)


def load(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    return tf.image.resize(img, (IMG, IMG))


def predict(model, paths):
    ds = tf.data.Dataset.from_tensor_slices(paths).map(load).batch(BATCH)
    return model.predict(ds, verbose=0)


def plot_cm(cm, labels, title, fname):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------- Model 1: Material ----------
m_model = keras.models.load_model("models/material.keras")
m_classes = json.load(open("models/material_classes.json"))
y1 = test["material"].map({c: i for i, c in enumerate(m_classes)}).values
p1 = predict(m_model, test["path"].values)
pred1 = p1.argmax(axis=1)

report1 = classification_report(y1, pred1, target_names=m_classes)
cm1 = confusion_matrix(y1, pred1)
plot_cm(cm1, m_classes, "Model 1: Material (test set)", "results/cm_material.png")

plt.figure(figsize=(5, 3.5))
plt.hist(p1.max(axis=1), bins=20, color="steelblue")
plt.xlabel("Prediction confidence")
plt.ylabel("Images")
plt.title("Model 1: confidence distribution")
plt.tight_layout()
plt.savefig("results/conf_material.png", dpi=150)
plt.close()

# ---------- Model 2: Condition (MVTec only, tuned threshold) ----------
c_model = keras.models.load_model("models/condition.keras")
c_classes = json.load(open("models/condition_classes.json"))
DEF = c_classes.index("defective")
THR = round(float(json.load(open("models/condition_threshold.json"))["defective_threshold"]), 2)

t2 = test[test["source"] == "mvtec"].reset_index(drop=True)
p_def = predict(c_model, t2["path"].values)[:, DEF]
is_def = (t2["condition"] == "defective").values
y2 = np.where(is_def, 0, 1)                 # 0 = defective, 1 = normal
pred2 = np.where(p_def >= THR, 0, 1)

labels2 = ["defective", "normal"]
report2 = classification_report(y2, pred2, target_names=labels2)
cm2 = confusion_matrix(y2, pred2)
plot_cm(cm2, labels2, f"Model 2: Condition (threshold {THR})", "results/cm_condition.png")

plt.figure(figsize=(5.5, 3.5))
plt.hist(p_def[is_def], bins=20, alpha=0.7, label="actual defective", color="crimson")
plt.hist(p_def[~is_def], bins=20, alpha=0.7, label="actual normal", color="seagreen")
plt.axvline(THR, color="black", linestyle="--", label=f"threshold {THR}")
plt.xlabel("Defect probability")
plt.ylabel("Images")
plt.legend()
plt.title("Model 2: defect probability")
plt.tight_layout()
plt.savefig("results/conf_condition.png", dpi=150)
plt.close()

# ---------- Text report ----------
with open("results/metrics.txt", "w") as f:
    f.write(f"MODEL 1: MATERIAL (test n={len(test)})\n{report1}\n{cm1}\n\n")
    f.write(f"MODEL 2: CONDITION (MVTec test n={len(t2)}, threshold={THR})\n{report2}\n{cm2}\n")

print(report1)
print(cm1)
print(f"\nMODEL 2 (threshold {THR}, n={len(t2)})")
print(report2)
print(cm2)
print("\nSaved: results/cm_material.png, cm_condition.png, conf_material.png, conf_condition.png, metrics.txt")