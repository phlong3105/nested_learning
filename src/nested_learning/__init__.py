"""
Nested Learning: The Illusion of Deep Learning Architectures

Implementation of the NeurIPS 2025 paper by Behrouz et al.
"""

__version__ = "0.1.0"

from nested_learning.optimizers import (
    DeepMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum,
)
from nested_learning.memory import AssociativeMemory, ContinuumMemorySystem
from nested_learning.models import HOPE, SelfModifyingTitan

__all__ = [
    "DeepMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
    "AssociativeMemory",
    "ContinuumMemorySystem",
    "HOPE",
    "SelfModifyingTitan",
]