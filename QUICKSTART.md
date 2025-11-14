# Quick Start Guide

## Installation

```bash
cd nested-learning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Running Examples

### 1. Simple Examples

Run the simple example script to understand basic concepts:

```bash
python examples/simple_example.py
```

This demonstrates:
- Deep Momentum optimizer
- Delta-Rule optimizer
- Continuum Memory System
- HOPE model basics

### 2. Training a Language Model

Train a small HOPE model on WikiText:

```bash
# Train 340M model (requires GPU)
python experiments/train_lm.py \
    --model hope \
    --size 340M \
    --dataset wikitext \
    --optimizer deep_momentum \
    --epochs 10 \
    --batch_size 8

# Use wandb for logging (optional)
python experiments/train_lm.py \
    --model hope \
    --size 340M \
    --wandb \
    --project_name my-nested-learning
```

### 3. Running Tests

Verify installation with tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_optimizers.py -v
pytest tests/test_models.py -v
```

## Using in Your Code

### Deep Optimizers

```python
from nested_learning.optimizers import DeepMomentumGD

# Your model
model = YourModel()

# Deep momentum optimizer
optimizer = DeepMomentumGD(
    model.parameters(),
    lr=1e-3,
    momentum=0.9,
    memory_depth=2,  # Depth of MLP for momentum memory
)

# Training loop
for batch in dataloader:
    loss = compute_loss(model, batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### HOPE Model

```python
from nested_learning.models import HOPE

# Create model
model = HOPE(
    dim=512,
    n_layers=12,
    n_heads=8,
    vocab_size=50257,
    chunk_sizes=[16, 64, 256],  # Multi-frequency CMS
    max_seq_len=2048,
)

# Training
loss, logits = model(input_ids, labels=labels)
loss.backward()

# Generation
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
)
```

### Continuum Memory System

```python
from nested_learning.memory import ContinuumMemorySystem

# Create CMS with 3 levels
cms = ContinuumMemorySystem(
    dim=512,
    hidden_dim=2048,
    num_levels=3,
    chunk_sizes=[8, 32, 128],  # Different update frequencies
)

# Use in your model
output = cms(input_tensor)
```

## Configuration Files

You can also use YAML config files:

```bash
# configs/hope_340M.yaml already provided
python experiments/train_lm.py --config configs/hope_340M.yaml
```

## Next Steps

1. Read `docs/CONCEPTS.md` for deeper understanding
2. Explore paper: "Nested Learning: The Illusion of Deep Learning Architectures"
3. Experiment with different optimizers and CMS configurations
4. Try scaling to larger models (760M, 1.3B)

## Common Issues

### Out of Memory

- Reduce `batch_size`
- Reduce `max_length`
- Use gradient accumulation
- Use smaller model size

### Slow Training

- Ensure GPU is being used (`--device cuda`)
- Reduce `memory_depth` in DeepMomentumGD
- Use mixed precision training

### Import Errors

Make sure to install in development mode:
```bash
pip install -e .
```

## Resources

- Paper: NeurIPS 2025, Behrouz et al.
- GitHub Issues: Report bugs and ask questions
- CONTRIBUTING.md: Guidelines for contributors