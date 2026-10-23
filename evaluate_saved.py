import json
from pathlib import Path

import tensorflow as tf

from train import build_models, load_dataset, evaluate_test_set

# Allow TensorFlow to allocate GPU memory as needed.
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU memory configuration:", exc)

DATA_DIR = Path("data/chest_xray")
OUTPUT_DIR = Path("artifacts")
WEIGHTS = OUTPUT_DIR / "chest_xray.weights.h5"

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Saved weights not found: {WEIGHTS}")

if not (DATA_DIR / "test").is_dir():
    raise FileNotFoundError(f"Test dataset not found: {DATA_DIR / 'test'}")

print("Building model architecture...")
classifier, _, _ = build_models(imagenet_weights=False)

print("Loading saved weights...")
classifier.load_weights(WEIGHTS)

print("Loading test dataset with batch size 4...")
test_data = load_dataset(DATA_DIR / "test", 4, shuffle=False)

print("Evaluating saved model...")
metrics = evaluate_test_set(classifier, test_data)

metrics_path = OUTPUT_DIR / "metrics.json"
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nEvaluation results:")
print(json.dumps(metrics, indent=2, default=str))
print(f"\nMetrics saved to: {metrics_path}")
