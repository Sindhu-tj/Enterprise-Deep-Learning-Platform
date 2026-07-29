"""
=========================================================
Enterprise Deep Learning Platform
ANN Evaluation Pipeline
=========================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =========================================================
# Load Dataset
# =========================================================

DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

df = pd.read_csv(DATASET_PATH)


# =========================================================
# Features & Target
# =========================================================

TARGET_COLUMN = "target"

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]


# =========================================================
# Load Scaler
# =========================================================

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "ann_scaler.pkl"
    )
)

X = scaler.transform(X)


# =========================================================
# Load Model
# =========================================================

model = load_model(
    os.path.join(
        MODEL_DIR,
        "ann_model.keras"
    )
)


# =========================================================
# Prediction
# =========================================================

predictions = model.predict(X)

predictions = (predictions > 0.5).astype(int)

predictions = predictions.flatten()


# =========================================================
# Metrics
# =========================================================

accuracy = accuracy_score(y, predictions)

precision = precision_score(y, predictions)

recall = recall_score(y, predictions)

f1 = f1_score(y, predictions)

cm = confusion_matrix(y, predictions)

report = classification_report(y, predictions)


# =========================================================
# Results
# =========================================================

print("=" * 60)

print("ANN MODEL EVALUATION")

print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("=" * 60)

print("Confusion Matrix")

print(cm)

print("=" * 60)

print("Classification Report")

print(report)

print("=" * 60)