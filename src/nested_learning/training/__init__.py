"""
Training utilities for nested learning.

Key classes:
- NestedLearningTrainer: Unified trainer integrating all components
  - DeepMomentumGD optimizer with internal loss
  - CMS multi-frequency gradient updates
  - Optional meta-learning
- ContinuumMemoryTrainer: Legacy trainer for CMS-only training
"""

from nested_learning.training.nested_trainer import (
    NestedLearningTrainer,
    create_nested_learning_setup,
)
from nested_learning.training.continuum_trainer import (
    ContinuumMemoryTrainer,
    visualize_update_schedule,
    create_continuum_model_example,
)

__all__ = [
    # Main trainer
    'NestedLearningTrainer',
    'create_nested_learning_setup',
    # Legacy CMS trainer
    'ContinuumMemoryTrainer',
    'visualize_update_schedule',
    'create_continuum_model_example',
]
