"""Tests for models."""

import torch
import pytest

from nested_learning.models import HOPE


def test_hope_forward():
    """Test HOPE forward pass."""
    model = HOPE(
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=100,
        chunk_sizes=[8, 16],
        max_seq_len=512,
    )

    # Create dummy input
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, 100)


def test_hope_training_step():
    """Test HOPE training step."""
    model = HOPE(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        chunk_sizes=[8],
    )

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = input_ids.clone()

    # Forward with loss
    loss, logits = model(input_ids, labels=labels)

    # Check shapes
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar
    assert logits.shape == (batch_size, seq_len, 100)

    # Backward pass should work
    loss.backward()


def test_hope_generation():
    """Test HOPE text generation."""
    model = HOPE(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        max_seq_len=512,
    )
    model.eval()

    # Starting prompt
    input_ids = torch.randint(0, 100, (1, 10))

    # Generate
    output = model.generate(input_ids, max_new_tokens=20)

    # Check that tokens were generated
    assert output.shape[1] == 30  # 10 + 20


def test_hope_adaptation_scope():
    """Test HOPE adaptation_scope context manager restores weights."""
    model = HOPE(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        max_seq_len=512,
        use_self_modification=True,
    )
    model.eval()

    # Get initial state
    initial_state = model.get_self_mod_state()

    # Input that will trigger self-modification
    input_ids = torch.randint(0, 100, (1, 20))

    # Run forward pass within adaptation_scope
    with model.adaptation_scope():
        # Multiple forward passes to accumulate modifications
        for _ in range(3):
            _ = model(input_ids, enable_self_modification=True)
            model.apply_pending_updates()

        # Verify weights changed within scope
        modified_state = model.get_self_mod_state()
        weights_changed = False
        for key in initial_state:
            if not torch.allclose(initial_state[key], modified_state[key], atol=1e-6):
                weights_changed = True
                break
        assert weights_changed, "Self-modification should have changed weights within scope"

    # After scope exits, weights should be restored
    restored_state = model.get_self_mod_state()
    for key in initial_state:
        assert torch.allclose(initial_state[key], restored_state[key], atol=1e-8), \
            f"Weight {key} was not properly restored after adaptation_scope"


def test_hope_adaptation_scope_isolation():
    """Test that separate adaptation scopes are isolated."""
    model = HOPE(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=100,
        use_self_modification=True,
    )
    model.eval()

    input_ids_1 = torch.randint(0, 100, (1, 20))
    input_ids_2 = torch.randint(0, 100, (1, 20))

    initial_state = model.get_self_mod_state()

    # First adaptation scope
    with model.adaptation_scope():
        _ = model(input_ids_1, enable_self_modification=True)
        model.apply_pending_updates()

    # State should be restored
    mid_state = model.get_self_mod_state()
    for key in initial_state:
        assert torch.allclose(initial_state[key], mid_state[key], atol=1e-8)

    # Second adaptation scope
    with model.adaptation_scope():
        _ = model(input_ids_2, enable_self_modification=True)
        model.apply_pending_updates()

    # State should still match initial (scopes are isolated)
    final_state = model.get_self_mod_state()
    for key in initial_state:
        assert torch.allclose(initial_state[key], final_state[key], atol=1e-8)


if __name__ == '__main__':
    pytest.main([__file__])