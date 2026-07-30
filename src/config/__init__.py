"""
Configuration Module

This package contains global configuration variables
used throughout the Enterprise Deep Learning Platform.
"""

from .settings import *

__all__ = [
    "DEVICE",
    "RANDOM_SEED",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "NUM_EPOCHS",
    "NUM_CLASSES",
    "MODEL_DIR",
    "DATA_DIR",
    "LOG_DIR",
]