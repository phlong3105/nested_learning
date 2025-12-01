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


if __name__ == '__main__':
    pytest.main([__file__])