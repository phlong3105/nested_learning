"""Tests for optimizers."""

import torch
import torch.nn as nn
import pytest

from nested_learning.optimizers import (
    DeepMomentumGD,
    DeltaRuleMomentum,
    PreconditionedMomentum,
)


def test_deep_momentum_basic():
    """Test basic DeepMomentumGD functionality."""
    # Simple model
    model = nn.Linear(10, 5)
    optimizer = DeepMomentumGD(model.parameters(), lr=0.01)

    # Dummy data
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    # Forward pass
    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Check that parameters were updated
    assert model.weight.grad is not None


def test_delta_rule_momentum():
    """Test DeltaRuleMomentum optimizer."""
    model = nn.Linear(10, 5)
    optimizer = DeltaRuleMomentum(model.parameters(), lr=0.01, momentum=0.9)

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.weight.grad is not None


def test_preconditioned_momentum():
    """Test PreconditionedMomentum optimizer."""
    model = nn.Linear(10, 5)
    optimizer = PreconditionedMomentum(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        precondition_type='diagonal',
    )

    x = torch.randn(32, 10)
    y = torch.randn(32, 5)

    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.weight.grad is not None


def test_optimizer_convergence():
    """Test that optimizer can minimize a simple function."""
    # Create simple quadratic function to minimize
    target = torch.randn(5)
    param = nn.Parameter(torch.zeros(5))

    optimizer = DeepMomentumGD([param], lr=0.1, memory_depth=1)

    initial_loss = float(((param - target) ** 2).sum())

    # Optimize for a few steps
    for _ in range(100):
        loss = ((param - target) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = float(((param - target) ** 2).sum())

    # Loss should decrease
    assert final_loss < initial_loss
    # Should get close to target
    assert final_loss < 0.1


if __name__ == '__main__':
    pytest.main([__file__])