from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from chest_xray import CLASS_NAMES, IMAGE_SIZE, build_models, preprocess_image

WEIGHTS_PATH = Path("artifacts/chest_xray.weights.h5")
DISPLAY_NAMES = {
    "NORMAL": "Normal",
    "PNEUMONIA": "Pneumonia-suspected",
}

st.set_page_config(page_title="Chest X-ray Research Prototype", page_icon="🩻")
st.title("AI-Based Chest X-ray Analysis")
st.warning(
    "Educational research prototype only — not a medical device. "
    "This model can be wrong and must not be used to diagnose, treat, or rule out "
    "pneumonia. Seek advice from a qualified healthcare professional."
)
st.write(
    "Upload a chest X-ray to see the model's predicted dataset class, its softmax "
    "score, and a Grad-CAM heatmap. The heatmap is a rough model explanation, "
    "not a clinically validated localization."
)


@st.cache_resource
def load_model(weights_path: str) -> tf.keras.Model:
    classifier, gradcam_model, _ = build_models(imagenet_weights=False)
    classifier.load_weights(weights_path)
    return gradcam_model


if not WEIGHTS_PATH.is_file():
    st.error(
        f"Model weights were not found at {WEIGHTS_PATH}. "
        "Train the model using the instructions in README.md."
    )
    st.stop()

gradcam_model = load_model(str(WEIGHTS_PATH))
uploaded_file = st.file_uploader(
    "Choose a chest X-ray image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    resized = cv2.resize(
        np.asarray(image), (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA
    )
    batch = preprocess_image(tf.convert_to_tensor(resized[None, ...], dtype=tf.float32))
    with tf.GradientTape() as tape:
        feature_maps, class_scores = gradcam_model(batch, training=False)
        predicted_index_tensor = tf.argmax(class_scores[0])
        target_score = tf.gather(class_scores, predicted_index_tensor, axis=1)
    gradients = tape.gradient(target_score, feature_maps)
    probabilities = class_scores.numpy()[0]
    predicted_index = int(predicted_index_tensor.numpy())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.subheader(f"Predicted class: {DISPLAY_NAMES[predicted_class]}")
    st.metric("Model score (not a calibrated clinical probability)", f"{confidence:.1%}")
    st.caption(
        "The labels reflect the dataset's NORMAL and PNEUMONIA folders. "
        "A predicted class or score cannot confirm or exclude disease."
    )

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap_max = tf.reduce_max(heatmap)
    if float(heatmap_max) > 0:
        heatmap = heatmap / heatmap_max
    heatmap = cv2.resize(
        heatmap.numpy(), (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_LINEAR
    )
    color_map = cv2.applyColorMap(
        np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET
    )
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(resized, 0.6, color_map, 0.4, 0)
    st.image(
        overlay,
        caption="Grad-CAM overlay (highlighted areas are not a diagnosis)",
        use_container_width=True,
    )

    with st.expander("Scores for both dataset classes"):
        for index, class_name in enumerate(CLASS_NAMES):
            st.write(f"{DISPLAY_NAMES[class_name]}: {probabilities[index]:.1%}")
