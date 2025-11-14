"""
Deep Optimizers: Gradient-based optimizers as associative memory modules.

This module implements optimizers from the Nested Learning perspective,
where optimizers are viewed as associative memories that compress gradients.
"""

from .deep_momentum import DeepMomentumGD
from .delta_rule import DeltaRuleMomentum
from .preconditioned import PreconditionedMomentum

__all__ = [
    "DeepMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
]