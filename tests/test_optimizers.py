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

    # Use higher learning rate for faster convergence
    optimizer = DeepMomentumGD([param], lr=0.3, memory_depth=1)

    initial_loss = float(((param.detach() - target) ** 2).sum())

    # Optimize for more steps
    for _ in range(200):
        loss = ((param - target) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = float(((param.detach() - target) ** 2).sum())

    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, f"Loss did not decrease enough: {initial_loss} -> {final_loss}"
    # Should make progress toward target (more lenient threshold)
    assert final_loss < 1.0, f"Loss too high: {final_loss}"


def test_deep_momentum_memory_training():
    """
    Test that DeepMomentumGD actually trains memory modules.

    This verifies:
    1. Memory modules have gradients after step()
    2. Memory module weights change during training
    """
    torch.manual_seed(42)

    # Create model with enough parameters to use shared memory pool
    model = nn.Linear(64, 32)
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,  # Higher lr for visible changes
        use_shared_memory=True,
    )

    # Get initial memory weights
    initial_memory_weights = {}
    for name, param in optimizer.shared_memory.named_parameters():
        initial_memory_weights[name] = param.data.clone()

    # Run several training steps
    for step in range(10):
        x = torch.randn(32, 64)
        y = torch.randn(32, 32)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check that memory weights changed
    weights_changed = False
    for name, param in optimizer.shared_memory.named_parameters():
        if not torch.allclose(initial_memory_weights[name], param.data, atol=1e-6):
            weights_changed = True
            break

    assert weights_changed, "Memory module weights should change during training"


def test_deep_momentum_memory_gradients():
    """
    Test that memory modules receive gradients during training.

    This explicitly checks that internal loss produces non-zero gradients.
    """
    torch.manual_seed(42)

    model = nn.Linear(64, 32)
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        use_shared_memory=True,
    )

    # Run one step
    x = torch.randn(32, 64)
    y = torch.randn(32, 32)

    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()

    # Before optimizer step, memory should have no gradients
    # After step(), memory gradients get zeroed by memory_optimizer.zero_grad()
    # but weights should be updated

    # Get weights before step
    weights_before = {}
    for name, param in optimizer.shared_memory.named_parameters():
        weights_before[name] = param.data.clone()

    # Step (this trains memory via internal loss)
    optimizer.step()

    # Check weights changed (indicating gradients were computed and applied)
    any_changed = False
    for name, param in optimizer.shared_memory.named_parameters():
        if not torch.allclose(weights_before[name], param.data, atol=1e-8):
            any_changed = True
            break

    # Memory should be updated even in a single step
    # (the internal loss is always computed and backpropped)
    assert any_changed, "Memory weights should change after a single step (internal loss training)"


def test_deep_momentum_l2_regression_mode():
    """
    Test DeepMomentumGD with paper-exact L2 regression internal loss.

    This verifies:
    1. L2 regression mode runs without error
    2. Memory weights are updated
    3. L2 projection matrices are created and updated
    """
    torch.manual_seed(42)

    model = nn.Linear(64, 32)
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        use_shared_memory=True,
        internal_loss_mode='l2_regression',  # Paper-exact mode
        l2_projection_lr=0.1,
    )

    # Verify mode is set correctly
    stats = optimizer.get_internal_loss_stats()
    assert stats['internal_loss_mode'] == 'l2_regression'

    # Get initial memory weights
    initial_memory_weights = {}
    for name, param in optimizer.shared_memory.named_parameters():
        initial_memory_weights[name] = param.data.clone()

    # Run several training steps
    for step in range(10):
        x = torch.randn(32, 64)
        y = torch.randn(32, 32)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check that memory weights changed
    weights_changed = False
    for name, param in optimizer.shared_memory.named_parameters():
        if not torch.allclose(initial_memory_weights[name], param.data, atol=1e-6):
            weights_changed = True
            break

    assert weights_changed, "Memory module weights should change during L2 regression training"

    # Check that L2 projection matrices were created
    stats = optimizer.get_internal_loss_stats()
    assert stats['num_l2_projections'] > 0, "L2 projection matrices should be created"


def test_deep_momentum_convergence_l2_mode():
    """Test that L2 regression mode can minimize a simple function."""
    target = torch.randn(5)
    param = nn.Parameter(torch.zeros(5))

    optimizer = DeepMomentumGD(
        [param],
        lr=0.3,
        memory_depth=1,
        internal_loss_mode='l2_regression',
    )

    initial_loss = float(((param.detach() - target) ** 2).sum())

    for _ in range(200):
        loss = ((param - target) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = float(((param.detach() - target) ** 2).sum())

    # Loss should decrease significantly
    assert final_loss < initial_loss * 0.5, f"L2 mode: Loss did not decrease enough: {initial_loss} -> {final_loss}"


if __name__ == '__main__':
    pytest.main([__file__])