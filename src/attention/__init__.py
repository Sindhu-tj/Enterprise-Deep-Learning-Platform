"""
Attention Module

This package provides implementations of various attention mechanisms
commonly used in deep learning and Transformer-based architectures.

Modules:
- Attention
- SelfAttention
- MultiHeadAttention
"""

from .attention import Attention
from .self_attention import SelfAttention
from .multi_head_attention import MultiHeadAttention

__all__ = [
    "Attention",
    "SelfAttention",
    "MultiHeadAttention",
]