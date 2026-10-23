from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("NORMAL", "PNEUMONIA")


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    return keras.applications.mobilenet_v2.preprocess_input(image)


def build_models(
    imagenet_weights: bool = False,
) -> tuple[keras.Model, keras.Model, keras.Model]:
    """Build the classifier, its Grad-CAM view, and its transfer-learning backbone."""
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="xray")
    augmented = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.03),
            keras.layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )(inputs)

    backbone = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet" if imagenet_weights else None,
    )
    backbone.trainable = False
    feature_maps = backbone(augmented, training=False)
    pooled = keras.layers.GlobalAveragePooling2D(name="global_pool")(feature_maps)
    pooled = keras.layers.Dropout(0.25, name="dropout")(pooled)
    probabilities = keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="predictions"
    )(pooled)

    classifier = keras.Model(inputs, probabilities, name="chest_xray_classifier")
    gradcam_model = keras.Model(
        inputs, [feature_maps, probabilities], name="chest_xray_gradcam"
    )
    return classifier, gradcam_model, backbone
