"""
=========================================================
Enterprise Deep Learning Platform
ANN Prediction Pipeline
=========================================================
"""

import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


# =========================================================
# Load Model & Scaler
# =========================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ann_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "ann_scaler.pkl"
)

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)


# =========================================================
# Prediction Function
# =========================================================

def predict(sample):
    """
    Predict a single sample.

    Parameters
    ----------
    sample : list | numpy.ndarray | pandas.DataFrame

    Returns
    -------
    dict
    """

    if isinstance(sample, list):
        sample = np.array(sample).reshape(1, -1)

    elif isinstance(sample, np.ndarray):
        sample = sample.reshape(1, -1)

    elif isinstance(sample, pd.DataFrame):
        sample = sample.values

    sample = scaler.transform(sample)

    probability = model.predict(
        sample,
        verbose=0
    )[0][0]

    prediction = int(probability >= 0.5)

    label = "Positive" if prediction == 1 else "Negative"

    return {
        "prediction": prediction,
        "label": label,
        "confidence": float(probability)
    }


# =========================================================
# Example
# =========================================================

if __name__ == "__main__":

    sample = [
        0.25,
        0.81,
        0.45,
        0.32,
        0.77,
        0.91,
        0.15,
        0.48,
        0.67,
        0.28
    ]

    result = predict(sample)

    print("=" * 60)
    print("Prediction")
    print("=" * 60)

    print(result)