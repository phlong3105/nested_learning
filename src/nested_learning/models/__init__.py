"""
Models for Nested Learning

This module provides transformer-like models that implement the nested learning framework:

Key classes:
- HOPE: Transformer with self-modifying attention and continuum memory
  - Self-modification: Attention weights change during forward pass (delta rule)
  - CMS: Multi-frequency MLP stack for temporal memory hierarchy
  - Designed to work with NestedLearningTrainer for full nested optimization

- SelfModifyingTitan: RNN-based model with online weight updates
- SelfModifyingAttention: Attention layer with true parameter self-modification
- LinearAttentionWithMemory: Linear attention with external memory (fast weights)

The key insight: different timescales need different treatment.
- Fast patterns: Self-modifying attention (immediate adaptation)
- Medium patterns: Fast CMS levels (working memory)
- Slow patterns: Slow CMS levels (long-term memory)
"""

from .hope import HOPE, HOPEBlock, LinearAttentionWithMemory, HOPEForSequenceClassification
from .titans import (
    SelfModifyingTitan,
    SelfModifyingLinear,
    SelfModifyingAttention,
)

__all__ = [
    # HOPE model and components
    "HOPE",
    "HOPEBlock",
    "HOPEForSequenceClassification",
    "LinearAttentionWithMemory",
    # Self-modifying models
    "SelfModifyingTitan",
    "SelfModifyingLinear",
    "SelfModifyingAttention",
]
