import json
import numpy as np
import streamlit as st
from PIL import Image

try:
    import keras
except ImportError:
    try:
        from importlib import import_module

        keras = import_module("tensorflow.keras")
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow/Keras is required to run this app. Install the required dependencies first."
        ) from exc

IMG = 224
MATERIAL_MIN_CONF = 0.60   # isse kam confidence par manual review

st.set_page_config(page_title="Industrial Quality Inspection", layout="centered")


@st.cache_resource
def load_all():
    material = keras.models.load_model("models/material.keras")
    condition = keras.models.load_model("models/condition.keras")
    m_classes = json.load(open("models/material_classes.json"))
    c_classes = json.load(open("models/condition_classes.json"))
    thr = json.load(open("models/condition_threshold.json"))["defective_threshold"]
    return material, condition, m_classes, c_classes, round(float(thr), 2)


def to_array(pil_img):
    # Models ke andar Rescaling layer hai, isliye 0-255 pixels hi bhejna hai
    img = pil_img.convert("RGB").resize((IMG, IMG))
    return np.array(img, dtype=np.float32)[None]


material_model, condition_model, m_classes, c_classes, THR = load_all()
DEF = c_classes.index("defective")

st.title("Industrial Quality Inspection")
st.caption("Material → Condition → Defect type")

uploaded = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption=f"Uploaded: {uploaded.name}", width="stretch")

    if st.button("Detect Defect", type="primary"):
        x = to_array(pil_img)

        # Model 1: material
        m_probs = material_model.predict(x, verbose=0)[0]
        m_idx = int(m_probs.argmax())
        material = m_classes[m_idx]
        m_conf = float(m_probs[m_idx])

        # Model 2: condition (tuned threshold)
        p_def = float(condition_model.predict(x, verbose=0)[0][DEF])
        is_defective = p_def >= THR
        condition = "Defective" if is_defective else "Normal"
        c_conf = p_def if is_defective else 1 - p_def

        # Final decision
        if m_conf < MATERIAL_MIN_CONF:
            result, note = "NEEDS MANUAL REVIEW", "Material confidence low hai."
        elif is_defective and p_def < 0.50:
            result, note = "NEEDS MANUAL REVIEW", "Borderline defect: threshold se upar, par 50% se neeche."
        elif is_defective:
            result, note = "REJECTED", ""
        else:
            result, note = "ACCEPTED", ""

        st.subheader("INDUSTRIAL QUALITY INSPECTION")
        st.write(f"**Uploaded Image:** {uploaded.name}")
        st.write(f"**Material:** {material.capitalize()}")
        st.write(f"**Condition:** {condition}")
        st.write("**Defect Type:** Not available yet")

        st.write(f"Material Confidence: {m_conf:.0%}")
        st.write(f"Condition Confidence: {c_conf:.0%}")
        st.write(f"Defect probability (raw): {p_def:.0%}  |  threshold: {THR:.2f}")

        if result == "ACCEPTED":
            st.success(f"Inspection Result: {result}")
        elif result == "REJECTED":
            st.error(f"Inspection Result: {result}")
        else:
            st.warning(f"Inspection Result: {result}. {note}")
else:
    st.info("Upload an image to start.")