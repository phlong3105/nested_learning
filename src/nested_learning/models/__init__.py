"""
Models for Nested Learning

Includes:
- HOPE: Self-referential learning module with continuum memory
- SelfModifyingTitan: Sequence model that learns its own update algorithm
"""

from .hope import HOPE, HOPEBlock
from .titans import SelfModifyingTitan

__all__ = [
    "HOPE",
    "HOPEBlock",
    "SelfModifyingTitan",
]