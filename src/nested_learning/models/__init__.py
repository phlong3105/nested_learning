"""
Models for Nested Learning

Includes:
- HOPE: Self-referential learning module with continuum memory
- SelfModifyingTitan: Sequence model that learns its own update algorithm
- SelfModifyingLinear: Linear layer with online delta-rule weight updates
- SelfModifyingAttention: Attention with self-modifying parameters
"""

from .hope import HOPE, HOPEBlock
from .titans import (
    SelfModifyingTitan,
    SelfModifyingLinear,
    SelfModifyingAttention,
)

__all__ = [
    "HOPE",
    "HOPEBlock",
    "SelfModifyingTitan",
    "SelfModifyingLinear",
    "SelfModifyingAttention",
]