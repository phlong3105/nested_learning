"""
Deep Optimizers: Gradient-based optimizers as associative memory modules.

This module implements optimizers from the Nested Learning perspective,
where optimizers are viewed as associative memories that compress gradients.
"""

from .deep_momentum import DeepMomentumGD
from .delta_rule import DeltaRuleMomentum
from .preconditioned import PreconditionedMomentum
from .nested_dmgd import (
    NestedDeepMomentumGD,
    LearnedMemoryModule,
    create_meta_learning_task_distribution,
)

__all__ = [
    # Basic optimizers (partial implementations - see docstrings for limitations)
    "DeepMomentumGD",
    "DeltaRuleMomentum",
    "PreconditionedMomentum",
    # Experimental nested implementation (needs validation)
    "NestedDeepMomentumGD",
    "LearnedMemoryModule",
    "create_meta_learning_task_distribution",
]