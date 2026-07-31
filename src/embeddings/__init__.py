"""
Evaluation Module

This package provides evaluation metrics for Machine Learning
and Deep Learning models.

Modules:
- Classification Metrics
- Regression Metrics
- Confusion Matrix
- ROC-AUC Score
"""

from .classification_metrics import classification_metrics
from .regression_metrics import regression_metrics
from .confusion_matrix import plot_confusion_matrix
from .roc_auc import roc_auc_score_plot

__all__ = [
    "classification_metrics",
    "regression_metrics",
    "plot_confusion_matrix",
    "roc_auc_score_plot",
]