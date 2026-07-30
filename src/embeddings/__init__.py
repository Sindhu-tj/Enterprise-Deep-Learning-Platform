"""
Embeddings Module

This package provides utilities for creating, managing,
and visualizing embeddings used in deep learning and
natural language processing.

Modules:
- Embedding
- Visualize
"""

from .embedding import Embedding
from .visualize import visualize_embeddings

__all__ = [
    "Embedding",
    "visualize_embeddings",
]