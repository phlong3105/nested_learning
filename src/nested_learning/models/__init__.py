"""
Models for Nested Learning

Includes:
- HOPE: Transformer-like model with linear attention and CMS (partial implementation)
- LinearAttentionWithMemory: Linear attention with fast weights (NOT self-modifying)
- SelfModifyingTitan: Sequence model with delta-rule weight updates (basic implementation)
- SelfModifyingLinear: Linear layer with online delta-rule weight updates
- SelfModifyingAttention: Attention with self-modifying parameters (in titans.py)

NOTE: "Self-modifying" in HOPE was a misnomer - it used external memory, not parameter updates.
The class has been renamed to LinearAttentionWithMemory for honesty.
"""

from .hope import HOPE, HOPEBlock, LinearAttentionWithMemory
from .titans import (
    SelfModifyingTitan,
    SelfModifyingLinear,
    SelfModifyingAttention,
)

__all__ = [
    "HOPE",
    "HOPEBlock",
    "LinearAttentionWithMemory",
    "SelfModifyingTitan",
    "SelfModifyingLinear",
    "SelfModifyingAttention",
]