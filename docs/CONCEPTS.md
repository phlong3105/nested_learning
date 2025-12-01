# Core Concepts of Nested Learning

## Overview

Nested Learning (NL) is a paradigm that views deep learning models as integrated systems of **nested, multi-level optimization problems**, each with its own "context flow" and update frequency.

## Key Insights

### 1. Learning vs. Memorization

Following neuropsychology literature:
- **Memory**: A neural update caused by an input
- **Learning**: The process for acquiring effective and useful memory

### 2. Associative Memory

All components are associative memories that map keys to values:

```
M* = argmin_M L(M(K); V)
```

Where:
- K: Keys (e.g., input tokens, gradients)
- V: Values (e.g., target outputs, updates)
- L: Objective function (dot-product, L2, etc.)

### 3. Update Frequency Hierarchy

Components are ordered by their update frequency:

- **Level 0** (Highest frequency): Attention states - updated every token
- **Level 1**: Projection weights - updated every batch
- **Level 2**: MLP weights - updated every chunk
- **Level 3**: Optimizer state - updated every batch

## Deep Optimizers

### Gradient Descent with Momentum

Traditional view:
```python
W_{t+1} = W_t - lr * m_t
m_t = momentum * m_{t-1} + grad_t
```

NL view: Momentum is an associative memory compressing gradients:
```python
m_t = argmin_m <m, grad_t> + regularization
```

### Deep Momentum

Replace linear momentum with MLP:
```python
m_t = MLP(grad_t)  # Non-linear compression of gradients
W_{t+1} = W_t - lr * m_t
```

### Delta-Rule Momentum

Use L2 objective instead of dot-product:
```python
m_t = argmin_m ||m * grad_t - target||²
```

This provides better capacity management.

## Continuum Memory System

Instead of binary long-term/short-term memory, CMS uses multiple levels with different update frequencies:

```python
# Level 1: Update every 16 tokens
MLP_1: Updated when step % 16 == 0

# Level 2: Update every 64 tokens
MLP_2: Updated when step % 64 == 0

# Level 3: Update every 256 tokens
MLP_3: Updated when step % 256 == 0
```

Each level compresses information at its own timescale.

## HOPE Architecture

HOPE combines:

1. **Self-Modifying Attention**: Learns its own update rule
2. **Continuum Memory**: Multi-frequency knowledge storage
3. **Novel Gradient Descent**: Better handling of token dependencies

```python
class HOPEBlock:
    def forward(x):
        # Self-modifying attention
        x = x + SelfModAttn(x)

        # Continuum memory (3 levels)
        x = x + MLP_f1(x)  # High frequency
        x = x + MLP_f2(x)  # Medium frequency
        x = x + MLP_f3(x)  # Low frequency

        return x
```

## Connections to Neuroscience

### Memory Consolidation

The brain consolidates memories through two processes:

1. **Online Consolidation** (Synaptic): Rapid, during wakefulness
2. **Offline Consolidation** (Systems): Slower, during sleep/replay

CMS mirrors this with multi-frequency updates.

### Neuroplasticity

The brain adapts through:
- **Uniform structure**: Reusable cortical columns
- **Multi-timescale updates**: Different frequencies for different areas

NL captures this through:
- **Uniform blocks**: Repeated HOPE blocks
- **Frequency hierarchy**: Different update rates per level

## Mathematical Foundations

### Optimization Perspective

A model with optimizer can be viewed as:

```
Level 0: max-frequency (attention)
    min_M L_0(M; context)

Level 1: mid-frequency (projections)
    min_W L_1(W; data_chunk)

Level 2: low-frequency (MLP)
    min_θ L_2(θ; large_context)

Level 3: min-frequency (optimizer)
    min_m L_3(m; gradients)
```

Each level optimizes its own objective over its own context.

### Context Flow Compression

Each component learns to **compress** its context flow:

```
Input Context → Component → Compressed Representation
    ↓                ↓              ↓
  Tokens         Attention       Memory State
  Batches        Weights         Parameters
  Gradients      Momentum        Compressed Grad
```

## Practical Implications

### 1. More Expressive Optimizers

Instead of hand-crafted update rules, learn the optimizer:
- Deep memory modules (MLPs instead of matrices)
- Better objectives (L2 instead of dot-product)
- Adaptive preconditioning

### 2. Continual Learning

Multi-frequency updates enable:
- Fast adaptation (high-frequency levels)
- Stable long-term memory (low-frequency levels)
- No catastrophic forgetting

### 3. In-Context Learning

Higher-order in-context learning emerges from:
- Multiple nested levels
- Each level learning to learn
- Hierarchical abstraction

## Further Reading

- Original paper: "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025)
- Related: Fast Weight Programmers, Meta-Learning, Test-Time Training