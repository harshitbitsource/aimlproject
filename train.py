from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from chest_xray import CLASS_NAMES, IMAGE_SIZE, build_models, preprocess_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a chest X-ray educational classifier."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Dataset root containing train/, val/, and test/ class folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-layers", type=int, default=30)
    parser.add_argument(
        "--no-imagenet",
        action="store_true",
        help="Initialize MobileNetV2 randomly instead of loading ImageNet weights.",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        parser.error("--epochs must be positive and --fine-tune-epochs cannot be negative.")
    if args.batch_size < 1 or args.fine_tune_layers < 1:
        parser.error("--batch-size and --fine-tune-layers must be positive.")
    return args


def validate_dataset(data_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Required dataset split was not found: {split_dir}")
        actual_classes = {path.name for path in split_dir.iterdir() if path.is_dir()}
        if actual_classes != set(CLASS_NAMES):
            raise ValueError(
                f"{split_dir} must contain exactly these class folders: "
                f"{', '.join(CLASS_NAMES)}; found: {', '.join(sorted(actual_classes)) or 'none'}"
            )
        for class_name in CLASS_NAMES:
            image_count = sum(
                1
                for path in (split_dir / class_name).iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if image_count == 0:
                raise ValueError(f"No supported images found in {split_dir / class_name}")


def load_dataset(
    directory: Path, batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        class_names=list(CLASS_NAMES),
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=42,
    )
    return dataset.map(
        lambda images, labels: (preprocess_image(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def get_class_weights(train_dir: Path) -> dict[int, float]:
    counts = {
        index: sum(
            1
            for path in (train_dir / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }
    total = sum(counts.values())
    return {index: total / (len(CLASS_NAMES) * count) for index, count in counts.items()}


def evaluate_test_set(
    model: keras.Model, test_data: tf.data.Dataset
) -> dict[str, object]:
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    for images, labels in test_data:
        probabilities = model(images, training=False).numpy()
        true_labels.append(labels.numpy())
        predicted_labels.append(np.argmax(probabilities, axis=1))

    truth = np.concatenate(true_labels)
    predictions = np.concatenate(predicted_labels)
    confusion = [
        [
            int(np.sum((truth == actual) & (predictions == predicted)))
            for predicted in range(len(CLASS_NAMES))
        ]
        for actual in range(len(CLASS_NAMES))
    ]
    per_class_recall = {
        CLASS_NAMES[index]: (
            confusion[index][index] / sum(confusion[index])
            if sum(confusion[index])
            else 0.0
        )
        for index in range(len(CLASS_NAMES))
    }
    return {
        "test_accuracy": float(np.mean(truth == predictions)),
        "test_count": int(len(truth)),
        "confusion_matrix": confusion,
        "class_order": list(CLASS_NAMES),
        "recall_by_class": per_class_recall,
    }


def main() -> None:
    args = parse_args()
    validate_dataset(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    train_dir = args.data_dir / "train"
    train_data = load_dataset(train_dir, args.batch_size, shuffle=True)
    val_data = load_dataset(args.data_dir / "val", 4, shuffle=False)
    test_data = load_dataset(args.data_dir / "test", 4, shuffle=False)
    classifier, _, backbone = build_models(imagenet_weights=not args.no_imagenet)
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    head_history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        class_weight=get_class_weights(train_dir),
        callbacks=[early_stopping],
    )
    best_weights = classifier.get_weights()
    best_val_loss = min(head_history.history["val_loss"])

    fine_tune_history: dict[str, list[float]] = {}
    if args.fine_tune_epochs:
        backbone.trainable = True
        for layer in backbone.layers[:-args.fine_tune_layers]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        classifier.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        fine_tune_history = classifier.fit(
            train_data,
            validation_data=val_data,
            epochs=args.fine_tune_epochs,
            class_weight=get_class_weights(train_dir),
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                )
            ],
        ).history
        if min(fine_tune_history["val_loss"]) < best_val_loss:
            best_val_loss = min(fine_tune_history["val_loss"])
        else:
            classifier.set_weights(best_weights)

    classifier.save_weights(args.output_dir / "chest_xray.weights.h5")
    metrics = evaluate_test_set(classifier, test_data)
    metrics["best_validation_loss"] = float(best_val_loss)
    metrics["head_epochs_completed"] = len(head_history.history["loss"])
    metrics["fine_tune_epochs_completed"] = len(fine_tune_history.get("loss", []))
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Weights saved to {args.output_dir / 'chest_xray.weights.h5'}")


if __name__ == "__main__":
    main()
