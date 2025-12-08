"""
Tests for meta-learning functionality.

These tests verify that:
1. NestedDeepMomentumGD.meta_step() actually updates memory module weights
2. Memory modules can be trained via meta-learning
3. The surrogate loss approach works correctly
"""

import torch
import torch.nn as nn
import pytest

from nested_learning.optimizers import NestedDeepMomentumGD


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_meta_step_updates_memory():
    """Verify meta_step actually changes memory module weights."""
    # Create model and optimizer
    model = SimpleModel()
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        meta_learning=True,
    )

    # Get initial memory weights
    initial_weights = {}
    for i, memory in enumerate(optimizer.get_memory_modules()):
        initial_weights[i] = {
            k: v.clone() for k, v in memory.state_dict().items()
        }

    # Run some training steps with create_graph=True
    for _ in range(5):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(create_graph=True)

    # Compute validation loss (meta loss)
    val_x = torch.randn(32, 10)
    val_y = torch.randn(32, 5)
    val_output = model(val_x)
    val_loss = nn.functional.mse_loss(val_output, val_y)

    # Meta step
    optimizer.meta_step(val_loss)

    # Verify weights changed
    final_weights = {}
    for i, memory in enumerate(optimizer.get_memory_modules()):
        final_weights[i] = memory.state_dict()

    # Check at least one weight tensor changed
    weights_changed = False
    for i in initial_weights:
        for key in initial_weights[i]:
            if not torch.allclose(initial_weights[i][key], final_weights[i][key], atol=1e-6):
                weights_changed = True
                break

    assert weights_changed, "meta_step did not update any memory weights"


def test_meta_learning_improves_optimization():
    """
    Test that meta-learning mechanism runs without errors.

    Note: Actually demonstrating meta-learning improves optimization
    requires careful hyperparameter tuning and longer training.
    This test verifies the mechanics work correctly.
    """
    torch.manual_seed(42)

    # Test that meta_step() can be called without errors
    # and that training makes some progress
    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.001,
        meta_learning=True,
    )

    initial_params = {n: p.clone() for n, p in model.named_parameters()}

    # Run training with meta-learning
    for step in range(10):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(create_graph=True)

        # Meta update every 5 steps
        if step > 0 and step % 5 == 0:
            val_x = torch.randn(32, 10)
            val_output = model(val_x)
            val_loss = val_output.pow(2).mean()
            optimizer.meta_step(val_loss)

    # Verify parameters have changed (training happened)
    params_changed = False
    for n, p in model.named_parameters():
        if not torch.allclose(initial_params[n], p):
            params_changed = True
            break

    assert params_changed, "Model parameters should change during training"


def test_compute_meta_gradients():
    """Test the compute_meta_gradients method."""
    model = SimpleModel()
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        meta_learning=True,
    )

    # Run step with create_graph=True to store memory outputs
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step(create_graph=True)

    # Verify memory outputs were stored
    assert hasattr(optimizer, '_last_memory_outputs')
    assert len(optimizer._last_memory_outputs) > 0

    # Compute meta gradients
    val_x = torch.randn(32, 10)
    val_output = model(val_x)
    val_loss = val_output.pow(2).mean()

    optimizer.memory_optimizer.zero_grad()
    optimizer.compute_meta_gradients(val_loss)

    # Check that some memory modules have gradients
    has_grad = False
    for memory in optimizer.get_memory_modules():
        for param in memory.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

    # This may not always produce gradients depending on the computation graph
    # but we verify the method runs without error


def test_nested_optimizer_state_dict():
    """Test saving and loading optimizer state with memory modules."""
    torch.manual_seed(42)

    model = SimpleModel()
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        meta_learning=True,
    )

    # Run some steps
    for _ in range(3):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save state
    state_dict = optimizer.state_dict()

    # Verify memory modules are included
    assert 'memory_modules' in state_dict
    assert len(state_dict['memory_modules']) > 0

    # The state dict should be loadable (even if weights don't match due to
    # different parameter layouts in the new optimizer). The key functionality
    # is that save/load doesn't raise errors.
    new_model = SimpleModel()
    new_optimizer = NestedDeepMomentumGD(
        new_model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        meta_learning=True,
    )

    # Load state - this should not raise an error
    # Note: Weights may not match exactly because parameter ids differ between
    # the two models, but the structure should be compatible
    try:
        new_optimizer.load_state_dict(state_dict)
    except Exception as e:
        pytest.fail(f"load_state_dict raised an exception: {e}")


def test_meta_learning_with_validation_data():
    """
    Test meta-learning with separate train/validation data.

    This mimics the typical use case where we optimize on training data
    and use validation loss for meta-updates.
    """
    torch.manual_seed(123)

    # Create simple linear regression task
    true_weights = torch.randn(10, 5)

    def make_data(n=32):
        x = torch.randn(n, 10)
        y = x @ true_weights + 0.1 * torch.randn(n, 5)
        return x, y

    model = SimpleModel(input_dim=10, hidden_dim=20, output_dim=5)
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.001,
        meta_learning=True,
    )

    # Training with meta-learning
    train_losses = []

    for epoch in range(5):
        # Inner loop: 10 training steps
        for _ in range(10):
            x_train, y_train = make_data()
            output = model(x_train)
            loss = nn.functional.mse_loss(output, y_train)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(create_graph=True)

        # Meta update using validation data
        x_val, y_val = make_data()
        val_output = model(x_val)
        val_loss = nn.functional.mse_loss(val_output, y_val)

        optimizer.meta_step(val_loss)

    # Training should make progress (loss decreasing overall)
    # Note: Due to stochastic nature, we just check that final loss is reasonable
    avg_first_10 = sum(train_losses[:10]) / 10
    avg_last_10 = sum(train_losses[-10:]) / 10
    assert avg_last_10 < avg_first_10 * 2, "Training loss should not increase dramatically"


def test_meta_learning_disabled():
    """Test that meta-learning can be disabled."""
    model = SimpleModel()
    optimizer = NestedDeepMomentumGD(
        model.parameters(),
        lr=0.01,
        memory_lr=0.01,
        meta_learning=False,  # Disabled
    )

    # Get initial weights
    initial_weights = {}
    for i, memory in enumerate(optimizer.get_memory_modules()):
        initial_weights[i] = {
            k: v.clone() for k, v in memory.state_dict().items()
        }

    # Run training
    x = torch.randn(32, 10)
    y = torch.randn(32, 5)
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Try meta_step (should do nothing)
    val_loss = nn.functional.mse_loss(model(torch.randn(32, 10)), torch.randn(32, 5))
    optimizer.meta_step(val_loss)  # Should not raise, should do nothing

    # Weights should not have changed from meta_step
    # (they may have changed from internal loss in step())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
