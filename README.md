# Nested Learning: The Illusion of Deep Learning Architectures

Implementation of the paper "Nested Learning: The Illusion of Deep Learning Architectures" by Behrouz et al. (NeurIPS 2025).

## Overview

This repository implements the Nested Learning (NL) paradigm, which represents machine learning models as integrated systems of nested, multi-level optimization problems. Each component has its own "context flow" and update frequency, enabling more expressive learning algorithms.

## Key Contributions

### 1. Deep Optimizers
Novel gradient-based optimizers that act as associative memory modules:
- **Deep Momentum Gradient Descent (DMGD)**: Extends momentum with deep memory
- **Delta-Rule Momentum**: Uses L2 regression for better gradient compression
- **Preconditioned Momentum**: More expressive associative memory with key-value mappings

### 2. Self-Modifying Titans
A sequence model that learns how to modify itself by learning its own update algorithm.

### 3. Continuum Memory System (CMS)
A novel formulation generalizing long-term/short-term memory with multi-frequency update schedules.

### 4. HOPE Architecture
Combines self-referential learning with the continuum memory system for:
- Language modeling
- Continual learning
- Long-context reasoning

## Architecture

```
nested-learning/
├── src/nested_learning/
│   ├── optimizers/          # Deep optimizers and variants
│   ├── memory/              # Memory systems (CMS, associative memory)
│   ├── models/              # HOPE and sequence models
│   └── utils/               # Helper functions
├── experiments/             # Training and evaluation scripts
├── configs/                 # Configuration files
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nested-learning.git
cd nested-learning

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Training HOPE Model

```python
from nested_learning.models import HOPE
from nested_learning.optimizers import DeepMomentumGD

# Initialize HOPE model
model = HOPE(
    dim=512,
    n_layers=12,
    n_heads=8,
    chunk_sizes=[16, 64, 256],  # Multi-frequency updates
    vocab_size=50257
)

# Train with deep optimizer
optimizer = DeepMomentumGD(model.parameters(), lr=1e-3)

# Your training loop here
```

### Using Deep Optimizers

```python
from nested_learning.optimizers import (
    DeepMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum
)

# Deep Momentum with MLP memory
optimizer = DeepMomentumGD(
    params=model.parameters(),
    lr=1e-3,
    memory_depth=2  # Depth of momentum memory module
)

# Delta-Rule based momentum
optimizer = DeltaRuleMomentum(
    params=model.parameters(),
    lr=1e-3,
    momentum=0.9
)
```

## Experiments

### Language Modeling

```bash
python experiments/train_lm.py \
    --model hope \
    --size 760M \
    --dataset wikitext \
    --optimizer deep_momentum
```

### Continual Learning

```bash
python experiments/continual_learning.py \
    --model hope \
    --benchmark split_cifar100
```

### Long Context Reasoning

```bash
python experiments/long_context.py \
    --model hope \
    --context_length 32768
```

## Key Concepts

### Nested Learning Paradigm

Traditional deep learning views models as stacked layers. Nested Learning reveals that models are actually **nested optimization problems** with different update frequencies:

- **Level 0** (Highest frequency): Working memory (e.g., attention states)
- **Level 1**: Projection layers
- **Level 2**: MLP layers (knowledge storage)
- **Level 3**: Optimizer state (momentum, second moments)

### Associative Memory

All components are **associative memories** that compress their context flow:

```python
# Simple example: Linear layer as associative memory
W* = argmin_W <W·x, ∇L(W; x)>
```

### Update Frequency

Components are ordered by update frequency `f_A`:
- Higher frequency → Updated more often per token
- Lower frequency → Updated less often (longer context compression)

## Results

Based on the paper (Table 1), HOPE achieves:

| Model | Size | Wikitext PPL ↓ | LAMBADA Acc ↑ | Avg. Reasoning ↑ |
|-------|------|----------------|---------------|------------------|
| HOPE | 760M | 20.53 | 39.02 | 52.26 |
| HOPE | 1.3B | 15.11 | 50.01 | 57.23 |
| Transformer++ | 1.3B | 18.53 | 42.60 | 52.25 |
| Titans (LMM) | 1.3B | 15.60 | 49.14 | 56.82 |

## Citation

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peiling and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Paper Reference

- **Paper**: Nested Learning: The Illusion of Deep Learning Architectures
- **Authors**: Ali Behrouz, Meisam Razaviyayn, Peiling Zhong, Vahab Mirrokni (Google Research)
- **Conference**: NeurIPS 2025
- **arXiv**: Coming November 13, 2024
- **Local Copy**: See [`paper.pdf`](paper.pdf) in this repository

## License

MIT License (adjust as needed)

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

This implementation is based on the NeurIPS 2025 paper by Google Research. Special thanks to the authors for their groundbreaking work on Nested Learning.