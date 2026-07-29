"""
=========================================================
Enterprise Deep Learning Platform
Artificial Neural Network (ANN) Package
=========================================================

This package contains:

- model.py      : ANN architecture
- train.py      : Training pipeline
- evaluate.py   : Model evaluation
- predict.py    : Prediction pipeline
"""

from .model import build_ann
from .predict import predict

__all__ = [
    "build_ann",
    "predict",
]