import os
import random
import numpy as np
import torch

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================================================
# Device Configuration
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# ==========================================================
# Dataset
# ==========================================================

NUM_CLASSES = 10
IMAGE_SIZE = 32
INPUT_CHANNELS = 3

# ==========================================================
# Model Save Paths
# ==========================================================

CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
ANN_MODEL_PATH = os.path.join(MODEL_DIR, "ann_model.pth")
RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_model.pth")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.pth")
GRU_MODEL_PATH = os.path.join(MODEL_DIR, "gru_model.pth")