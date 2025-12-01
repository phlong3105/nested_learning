# Nested Learning: Implementation Study

## Nov 2025 Update

Major cleanup: fixed gradient flow bugs, honest documentation, squashed history.
If you forked before this date, recommend re-cloning.

---

> **Partial Implementation**: This repo implements the API structure and several components from the NeurIPS 2025 paper "Nested Learning: The Illusion of Deep Learning Architectures" by Behrouz et al. (Google Research), but core concepts (nested optimization, multi-frequency training, self-modifying parameters) are **not fully implemented**. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed assessment.

**What works**: Linear attention, associative memory, basic optimizer variants, HOPE model structure
**What's missing**: Nested optimization loops, memory module training, multi-frequency update integration

## Implementation Status

| Component | API | Runs | Paper-Faithful | Notes |
|-----------|-----|------|----------------|-------|
| DeepMomentumGD | Yes | Yes | No | Static MLP, no nested learning |
| DeltaRuleMomentum | Yes | Yes | Partial | Simplified, preconditioner unused |
| PreconditionedMomentum | Yes | Yes | Yes | Honest, basic implementation |
| HOPE Model | Yes | Yes | Partial | Structure only, no self-modification |
| ContinuumMemorySystem | Yes | Yes | No | Multi-freq code exists but unused |
| LinearAttention | Yes | Yes | Yes | Works correctly |
| AssociativeMemory | Yes | Yes | Yes | Works correctly |
| SelfModifyingTitan | Yes | Yes | Partial | Basic delta-rule, needs validation |

### What This Repo Actually Demonstrates

- Paper reading and comprehension
- PyTorch engineering (custom optimizers, transformers)
- Identifying theory-practice gaps in ML research
- Honest technical assessment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nested-learning.git
cd nested-learning

# Install in development mode
pip install -e .
```

### Validate Installation

```bash
python validate_installation.py
```

### Run Demo

```bash
python demo_simple.py
```

## Architecture

```
nested-learning/
├── src/nested_learning/
│   ├── optimizers/          # Deep optimizers (API works, training incomplete)
│   ├── memory/              # Memory systems (CMS multi-freq unused)
│   ├── models/              # HOPE and sequence models
│   └── utils/               # Helper functions
├── experiments/             # Training scripts
├── tests/                   # Unit tests (smoke tests, not paper validation)
└── docs/                    # Documentation
```

## About This Implementation

This started as an attempt to implement the full Nested Learning framework. Through the process, I learned that:

1. **The paper assumes infrastructure that isn't described**: Meta-learning for optimizer memory, multi-timescale training loops
2. **"Self-modifying" is harder than it sounds**: True parameter self-modification during inference requires careful gradient handling
3. **Reading a paper is not the same as implementing it**: The gap is substantial for theoretical ML work

### This repo is useful for:

- Understanding the Nested Learning paper's architecture
- Starting point for a full implementation
- Learning PyTorch custom optimizer patterns
- Seeing honest assessment of implementation challenges

### This repo is NOT:

- A complete reproduction of the paper
- Production-ready code
- Experimentally validated against paper's results

## Key Concepts from the Paper

### Nested Learning Paradigm

Traditional deep learning views models as stacked layers. Nested Learning reveals that models are **nested optimization problems** with different update frequencies:

- **Level 0** (Highest frequency): Working memory (e.g., attention states)
- **Level 1**: Projection layers
- **Level 2**: MLP layers (knowledge storage)
- **Level 3**: Optimizer state (momentum, second moments)

### What We Implemented vs. What the Paper Describes

| Paper Concept | Our Implementation |
|---------------|-------------------|
| Deep memory trained via internal loss L^(2) | Static random MLP (never trained) |
| Multi-frequency CMS updates | Code exists but training loop doesn't call it |
| Self-modifying parameters W_t -> W_{t+1} | External memory buffer (fast weights) |
| Nested optimization alternating task/memory | Standard single-loss backprop |

## Usage Examples

### Using Deep Optimizers (with caveats)

```python
from nested_learning.optimizers import DeepMomentumGD

# NOTE: Memory MLPs are randomly initialized and NOT trained
# This is nonlinear momentum, not the paper's learned optimization
optimizer = DeepMomentumGD(
    params=model.parameters(),
    lr=1e-3,
    memory_depth=2
)

# Use like any PyTorch optimizer
for x, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
```

### Using HOPE Model

```python
from nested_learning.models import HOPE

# HOPE is a working transformer with custom attention
# But doesn't implement true self-referential learning
model = HOPE(
    dim=512,
    n_layers=12,
    n_heads=8,
    vocab_size=50257
)

# Standard language modeling
input_ids = torch.randint(0, 50257, (batch, seq_len))
logits = model(input_ids)
```

### Using Memory Systems

```python
from nested_learning.memory import AssociativeMemory, ContinuumMemorySystem

# AssociativeMemory works correctly
memory = AssociativeMemory(dim_key=64, dim_value=64)
memory.store(keys, values)
retrieved = memory.retrieve(query)

# CMS works as MLP stack, but multi-freq updates never triggered
cms = ContinuumMemorySystem(dim=128, num_levels=3)
output = cms(input)  # Just sequential MLP, not multi-timescale
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### High-Value Contributions Needed

1. **Nested Optimization** (High Impact): Implement internal loss L^(2) and alternating optimization
2. **Multi-Frequency Training** (Medium Impact): Integrate CMS update methods into training loop
3. **Experimental Reproduction** (Medium Impact): Scripts reproducing paper's Tables 1-3

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and Razaviyayn, Meisam and Zhong, Peiling and Mirrokni, Vahab},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Documentation

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Detailed technical assessment
- [STATUS.md](STATUS.md) - Component status overview
- [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md) - Test results

## License

MIT License - See LICENSE file for details

## Acknowledgments

This implementation is based on the NeurIPS 2025 paper by Google Research. All credit for the theoretical contributions goes to the original authors. This repository provides an independent, partial implementation for educational and research purposes.
