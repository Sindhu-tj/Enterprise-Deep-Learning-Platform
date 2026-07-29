"""
=========================================================
Enterprise Deep Learning Platform
ANN Training Pipeline
=========================================================
"""

import os
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from src.ann.model import build_ann


# =========================================================
# Project Paths
# =========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# =========================================================
# Load Dataset
# =========================================================

DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

df = pd.read_csv(DATASET_PATH)

print(df.head())


# =========================================================
# Features & Target
# =========================================================

TARGET_COLUMN = "target"

X = df.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]


# =========================================================
# Train Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================================================
# Feature Scaling
# =========================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "ann_scaler.pkl")
)


# =========================================================
# Build Model
# =========================================================

model = build_ann(
    input_dim=X_train.shape[1],
    output_dim=1
)


# =========================================================
# Callbacks
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_DIR,
        "ann_model.keras"
    ),
    save_best_only=True,
    monitor="val_loss"
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=5
)

tensorboard = TensorBoard(
    log_dir=LOG_DIR
)


# =========================================================
# Training
# =========================================================

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr,
        tensorboard
    ],
    verbose=1
)


# =========================================================
# Save Final Model
# =========================================================

model.save(
    os.path.join(
        MODEL_DIR,
        "ann_model.keras"
    )
)


joblib.dump(
    history.history,
    os.path.join(
        MODEL_DIR,
        "ann_history.pkl"
    )
)


# =========================================================
# Evaluate
# =========================================================

loss, accuracy, precision, recall = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("=" * 50)
print(f"Loss      : {loss:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print("=" * 50)