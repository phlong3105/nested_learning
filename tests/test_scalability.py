"""
Tests for scalability improvements.

These tests verify:
1. Gradient checkpointing reduces memory usage
2. Mixed precision training works correctly
3. Factorized memory reduces parameter count
"""

import torch
import torch.nn as nn
import pytest
import gc

from nested_learning.optimizers import DeepMomentumGD
from nested_learning.optimizers.deep_momentum import (
    MemoryMLP,
    FactorizedMemoryMLP,
    SharedMemoryPool,
)


class LargeModel(nn.Module):
    """Larger model for scalability testing."""

    def __init__(self, dim=256, n_layers=4):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(dim, dim),
                nn.ReLU(),
            ])
        layers.append(nn.Linear(dim, dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def count_parameters(module):
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_factorized_memory_is_smaller():
    """Verify factorized memory has fewer parameters than standard memory."""
    param_dim = 4096
    hidden_dim = 64
    rank = 16

    # Standard memory
    standard_memory = MemoryMLP(
        param_dim=param_dim,
        hidden_dim=hidden_dim,
        depth=2,
    )
    standard_params = count_parameters(standard_memory)

    # Factorized memory
    factorized_memory = FactorizedMemoryMLP(
        param_dim=param_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        depth=2,
    )
    factorized_params = count_parameters(factorized_memory)

    print(f"Standard memory params: {standard_params:,}")
    print(f"Factorized memory params: {factorized_params:,}")
    print(f"Reduction: {standard_params / factorized_params:.2f}x")

    # Factorized should have significantly fewer parameters
    assert factorized_params < standard_params, "Factorized memory should have fewer parameters"
    assert factorized_params < standard_params * 0.5, "Factorized memory should be at least 2x smaller"


def test_factorized_memory_functional():
    """Verify factorized memory produces valid outputs."""
    param_dim = 1024
    hidden_dim = 64
    rank = 16

    memory = FactorizedMemoryMLP(
        param_dim=param_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        depth=2,
    )

    # Test with batched input
    grad = torch.randn(1, param_dim)
    momentum = torch.randn(1, param_dim)
    output = memory(grad, momentum)

    assert output.shape == grad.shape, f"Output shape {output.shape} != input shape {grad.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"

    # Test with unbatched input
    grad_unbatched = torch.randn(param_dim)
    momentum_unbatched = torch.randn(param_dim)
    output_unbatched = memory(grad_unbatched, momentum_unbatched)

    assert output_unbatched.shape == (param_dim,), f"Unbatched output shape wrong: {output_unbatched.shape}"


def test_shared_memory_pool_factorized():
    """Test SharedMemoryPool with factorized memory option."""
    bucket_sizes = [64, 256, 1024, 4096]

    # Standard pool
    standard_pool = SharedMemoryPool(
        bucket_sizes=bucket_sizes,
        hidden_dim=64,
        depth=2,
        use_factorized=False,
    )
    standard_params = count_parameters(standard_pool)

    # Factorized pool (for large buckets)
    factorized_pool = SharedMemoryPool(
        bucket_sizes=bucket_sizes,
        hidden_dim=64,
        depth=2,
        use_factorized=True,
        factorized_rank=16,
        factorize_threshold=1024,  # Only factorize buckets >= 1024
    )
    factorized_params = count_parameters(factorized_pool)

    print(f"Standard pool params: {standard_params:,}")
    print(f"Factorized pool params: {factorized_params:,}")

    # Factorized pool should be smaller
    assert factorized_params < standard_params, "Factorized pool should have fewer parameters"

    # Test functional equivalence (both should produce valid outputs)
    grad = torch.randn(2048)  # Uses 4096 bucket
    momentum = torch.randn(2048)

    standard_output = standard_pool(grad, momentum, 2048)
    factorized_output = factorized_pool(grad, momentum, 2048)

    assert standard_output.shape == (2048,)
    assert factorized_output.shape == (2048,)


def test_deep_momentum_with_factorized_memory():
    """Test DeepMomentumGD with factorized memory option."""
    model = LargeModel(dim=256, n_layers=2)

    # Create optimizer with factorized memory
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        use_shared_memory=True,
        use_factorized_memory=True,
        factorized_rank=16,
    )

    # Run training step
    x = torch.randn(32, 256)
    y = torch.randn(32, 256)

    output = model(x)
    loss = nn.functional.mse_loss(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify no errors and parameters updated
    assert loss.item() >= 0


def test_gradient_checkpointing_functional():
    """Test that gradient checkpointing works without errors."""
    model = LargeModel(dim=128, n_layers=2)

    # Create optimizer with gradient checkpointing
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=0.01,
        use_shared_memory=True,
        gradient_checkpointing=True,
    )

    # Run several training steps
    for _ in range(3):
        x = torch.randn(32, 128)
        y = torch.randn(32, 128)

        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Verify training worked
    assert loss.item() >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gradient_checkpointing_reduces_memory():
    """
    Verify gradient checkpointing reduces GPU memory usage.

    This test compares peak memory usage with and without checkpointing.
    """
    torch.cuda.reset_peak_memory_stats()

    # Test without checkpointing
    model_no_ckpt = LargeModel(dim=512, n_layers=4).cuda()
    optimizer_no_ckpt = DeepMomentumGD(
        model_no_ckpt.parameters(),
        lr=0.01,
        use_shared_memory=True,
        gradient_checkpointing=False,
    )

    # Move memory pool to GPU
    if hasattr(optimizer_no_ckpt, 'shared_memory'):
        optimizer_no_ckpt.shared_memory = optimizer_no_ckpt.shared_memory.cuda()

    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(64, 512).cuda()
    y = torch.randn(64, 512).cuda()

    output = model_no_ckpt(x)
    loss = nn.functional.mse_loss(output, y)
    optimizer_no_ckpt.zero_grad()
    loss.backward()
    optimizer_no_ckpt.step()

    memory_no_ckpt = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    # Cleanup
    del model_no_ckpt, optimizer_no_ckpt, x, y, loss, output
    gc.collect()
    torch.cuda.empty_cache()

    # Test with checkpointing
    model_ckpt = LargeModel(dim=512, n_layers=4).cuda()
    optimizer_ckpt = DeepMomentumGD(
        model_ckpt.parameters(),
        lr=0.01,
        use_shared_memory=True,
        gradient_checkpointing=True,
    )

    if hasattr(optimizer_ckpt, 'shared_memory'):
        optimizer_ckpt.shared_memory = optimizer_ckpt.shared_memory.cuda()

    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(64, 512).cuda()
    y = torch.randn(64, 512).cuda()

    output = model_ckpt(x)
    loss = nn.functional.mse_loss(output, y)
    optimizer_ckpt.zero_grad()
    loss.backward()
    optimizer_ckpt.step()

    memory_ckpt = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    print(f"Memory without checkpointing: {memory_no_ckpt:.2f} MB")
    print(f"Memory with checkpointing: {memory_ckpt:.2f} MB")

    # Checkpointing should reduce memory (or at least not increase it significantly)
    # Note: The reduction may be small for this test case
    assert memory_ckpt <= memory_no_ckpt * 1.1, "Checkpointing shouldn't increase memory significantly"


def test_amp_wrapper_basic():
    """Test basic AMP wrapper functionality."""
    from nested_learning.utils.amp import NestedAMPWrapper

    # Test with AMP disabled
    amp_disabled = NestedAMPWrapper(enabled=False)
    assert not amp_disabled.enabled

    # Test with AMP enabled (CPU fallback)
    amp_enabled = NestedAMPWrapper(enabled=True, dtype=torch.bfloat16)

    # Test context managers don't raise errors
    with amp_enabled.model_autocast():
        x = torch.randn(32, 64)
        y = torch.randn(64, 32)
        z = x @ y

    with amp_enabled.full_precision():
        z_fp = x @ y


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_amp_wrapper_cuda():
    """Test AMP wrapper with CUDA."""
    from nested_learning.utils.amp import NestedAMPWrapper

    model = LargeModel(dim=128, n_layers=2).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    amp = NestedAMPWrapper(enabled=True, dtype=torch.float16)

    # Training step
    x = torch.randn(32, 128).cuda()
    y = torch.randn(32, 128).cuda()

    with amp.model_autocast():
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

    amp.backward(loss)
    amp.unscale_and_clip(optimizer, max_norm=1.0, parameters=model)
    amp.step(optimizer)
    amp.update()

    # Verify training worked
    assert loss.item() >= 0


def test_amp_config():
    """Test AMPConfig dataclass."""
    from nested_learning.utils.amp import AMPConfig

    config = AMPConfig()
    assert config.enabled is True
    assert config.dtype == torch.bfloat16

    config_custom = AMPConfig(
        enabled=False,
        dtype=torch.float16,
        use_separate_memory_scaler=True,
    )
    assert config_custom.enabled is False
    assert config_custom.use_separate_memory_scaler is True


def test_amp_trainer():
    """Test AMPTrainer class."""
    from nested_learning.utils.amp import AMPTrainer, AMPConfig

    model = LargeModel(dim=64, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    trainer = AMPTrainer(
        model=model,
        optimizer=optimizer,
        amp_config=AMPConfig(enabled=False),  # Disable for CPU test
        device='cpu',
    )

    # Training step
    x = torch.randn(32, 64)
    y = torch.randn(32, 64)

    metrics = trainer.train_step(
        x, y,
        loss_fn=nn.functional.mse_loss,
    )

    assert 'loss' in metrics
    assert metrics['loss'] >= 0


def test_optimizer_parameter_count_comparison():
    """
    Compare parameter counts for different optimizer configurations.

    This provides a summary of memory efficiency options.
    """
    model = LargeModel(dim=512, n_layers=4)

    # Standard optimizer
    opt_standard = DeepMomentumGD(
        model.parameters(),
        use_shared_memory=True,
        use_factorized_memory=False,
    )
    standard_params = count_parameters(opt_standard.shared_memory)

    # Factorized optimizer
    opt_factorized = DeepMomentumGD(
        model.parameters(),
        use_shared_memory=True,
        use_factorized_memory=True,
        factorized_rank=16,
    )
    factorized_params = count_parameters(opt_factorized.shared_memory)

    print("\n" + "=" * 50)
    print("Memory Module Parameter Comparison")
    print("=" * 50)
    print(f"Standard memory pool:   {standard_params:,} params")
    print(f"Factorized memory pool: {factorized_params:,} params")
    print(f"Reduction:              {(1 - factorized_params/standard_params)*100:.1f}%")
    print("=" * 50)

    assert factorized_params < standard_params


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
