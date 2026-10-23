# AI-Based Chest X-ray Analysis

An educational computer-vision prototype that trains a MobileNetV2 transfer-learning model to classify images into the `NORMAL` and `PNEUMONIA` folders of the Chest X-Ray Images (Pneumonia) dataset. The Streamlit app displays the predicted dataset class, its softmax score, and a Grad-CAM heatmap.

> **Important:** This is not a medical device or a validated diagnostic system. It can make incorrect predictions and must not be used to diagnose, treat, or rule out disease. Grad-CAM is only a rough explanation of model behavior, not a clinically validated lesion map. Consult a qualified healthcare professional for medical concerns.

## Dataset

Download and extract the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Pass the directory containing `train`, `val`, and `test` to the training script. Each split must contain exactly the two class directories shown below:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

Keep the dataset's provided train, validation, and test splits separate. The script uses the validation split for training decisions and reports final metrics on the test split.

## Setup and training

Python 3.10 or 3.11 is recommended. ImageNet weights are loaded by default for transfer learning, so the first run needs internet access.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data-dir /path/to/chest_xray
```

The script trains a classification head, fine-tunes the top backbone layers, applies class weighting, and selects the better training stage by validation loss. It saves:

- `artifacts/chest_xray.weights.h5` — model weights used by the app.
- `artifacts/metrics.json` — held-out test accuracy, class-wise recall, and confusion matrix.

Training options:

```bash
python train.py --help
```

For a run without downloading pretrained weights, add `--no-imagenet`; this starts the backbone with random weights and generally requires more data and training.

## Run the app

```bash
streamlit run app.py
```

Train the weights first. The app expects `artifacts/chest_xray.weights.h5` relative to the project directory. Upload a JPG, JPEG, or PNG chest X-ray to view its predicted class and Grad-CAM overlay.

## Evaluation and limitations

The reported test metrics describe performance only on this dataset's held-out split. They do not establish performance in a clinical setting or on images from other hospitals, scanners, patient populations, or acquisition conditions. Softmax scores are not calibrated clinical probabilities. The model is limited to the dataset's two labels and does not assess image quality or other conditions.
