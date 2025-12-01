#!/usr/bin/env python3
"""
Simple example demonstrating Nested Learning concepts.

This script shows:
1. How to use Deep Optimizers
2. How to train a HOPE model
3. How to use Continuum Memory System
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nested_learning.optimizers import DeepMomentumGD, DeltaRuleMomentum
from nested_learning.models import HOPE
from nested_learning.memory import ContinuumMemorySystem


def example_1_deep_optimizer():
    """Example 1: Using Deep Momentum optimizer."""
    print("=" * 60)
    print("Example 1: Deep Momentum Optimizer")
    print("=" * 60)

    # Simple regression task
    true_weights = torch.randn(10, 5)
    X = torch.randn(100, 10)
    y = X @ true_weights + 0.1 * torch.randn(100, 5)

    # Model
    model = nn.Linear(10, 5, bias=False)

    # Deep Momentum optimizer
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        memory_depth=2,  # Use 2-layer MLP for momentum memory
    )

    print(f"Training with Deep Momentum (memory_depth=2)...")

    # Train
    for epoch in range(50):
        # Forward
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}\n")


def example_2_delta_rule():
    """Example 2: Delta-Rule Momentum optimizer."""
    print("=" * 60)
    print("Example 2: Delta-Rule Momentum")
    print("=" * 60)

    # Same task as Example 1
    true_weights = torch.randn(10, 5)
    X = torch.randn(100, 10)
    y = X @ true_weights + 0.1 * torch.randn(100, 5)

    model = nn.Linear(10, 5, bias=False)

    # Delta-Rule optimizer (uses L2 objective for momentum)
    optimizer = DeltaRuleMomentum(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        precondition=True,  # Use preconditioning
    )

    print(f"Training with Delta-Rule Momentum...")

    for epoch in range(50):
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}\n")


def example_3_continuum_memory():
    """Example 3: Continuum Memory System."""
    print("=" * 60)
    print("Example 3: Continuum Memory System")
    print("=" * 60)

    # Create CMS with 3 levels
    cms = ContinuumMemorySystem(
        dim=64,
        hidden_dim=256,
        num_levels=3,
        chunk_sizes=[8, 32, 128],  # Different update frequencies
    )

    print("CMS Configuration:")
    print(f"  Levels: 3")
    print(f"  Level 1: Updates every 8 steps")
    print(f"  Level 2: Updates every 32 steps")
    print(f"  Level 3: Updates every 128 steps")

    # Dummy input
    batch_size = 4
    seq_len = 16
    x = torch.randn(batch_size, seq_len, 64)

    # Forward pass
    output = cms(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Each level compresses information at different timescales!\n")


def example_4_hope_model():
    """Example 4: Training a small HOPE model."""
    print("=" * 60)
    print("Example 4: HOPE Model")
    print("=" * 60)

    # Small HOPE model for demo
    model = HOPE(
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=100,
        chunk_sizes=[8, 16],  # 2-level CMS
        max_seq_len=256,
        dropout=0.0,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created HOPE model with {n_params:,} parameters")
    print(f"  Dimension: 128")
    print(f"  Layers: 4")
    print(f"  Heads: 4")
    print(f"  CMS levels: 2 (chunk_sizes=[8, 16])")

    # Dummy training data
    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = input_ids.clone()

    # Forward pass
    loss, logits = model(input_ids, labels=labels)

    print(f"\nForward pass:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Quick training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\nQuick training (10 steps):")
    for step in range(10):
        # Random batch
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        labels = input_ids.clone()

        loss, _ = model(input_ids, labels=labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 2 == 0:
            print(f"  Step {step+1}: Loss = {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    # Run all examples
    example_1_deep_optimizer()
    example_2_delta_rule()
    example_3_continuum_memory()
    example_4_hope_model()

    print("\nTo train a full model, run:")
    print("  python experiments/train_lm.py --model hope --size 340M")