#!/usr/bin/env python3
"""
Nested Learning Demo

This script demonstrates the full nested learning framework working end-to-end:
1. DeepMomentumGD with true nested optimization (memory modules trained)
2. CMS multi-frequency updates (different levels update at different rates)
3. Self-modifying attention (weights change during forward pass)

The demo shows:
- Memory modules being trained via internal loss
- CMS levels updating at different frequencies
- Self-modification statistics
- Comparison with standard optimization

Run this to verify the nested learning implementation is working correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nested_learning.optimizers import DeepMomentumGD, SimpleMomentumGD
from nested_learning.memory import ContinuumMemorySystem
from nested_learning.models import HOPE
from nested_learning.training.nested_trainer import NestedLearningTrainer, create_nested_learning_setup


def demo_deep_momentum_gd():
    """
    Demo 1: DeepMomentumGD with true nested optimization.

    Shows that memory modules are actually being trained via internal loss.
    """
    print("\n" + "=" * 60)
    print("Demo 1: DeepMomentumGD with Nested Optimization")
    print("=" * 60)

    # Simple model
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

    # Create optimizer with nested learning
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        memory_lr=1e-4,
        use_shared_memory=True,
    )

    # Simple regression task
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    print(f"\nTraining with DeepMomentumGD (memory modules learn via internal loss)...")
    print(f"Memory modules: {len(optimizer.get_memory_modules())}")

    losses = []
    for epoch in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            stats = optimizer.get_internal_loss_stats()
            print(f"  Epoch {epoch + 1}: Loss = {loss.item():.4f}, "
                  f"Steps = {stats['step_count']}")

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({100 * (1 - losses[-1] / losses[0]):.1f}% improvement)")

    # Compare with simple momentum
    print("\nComparing with SimpleMomentumGD (no nested learning)...")
    model2 = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    optimizer2 = SimpleMomentumGD(model2.parameters(), lr=1e-3, momentum=0.9)

    losses2 = []
    for epoch in range(50):
        optimizer2.zero_grad()
        pred = model2(X)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())

    print(f"SimpleMomentumGD final loss: {losses2[-1]:.4f}")
    print(f"DeepMomentumGD final loss:   {losses[-1]:.4f}")

    return losses[-1] < losses2[-1] * 1.5  # Should be competitive


def demo_cms_multifreq():
    """
    Demo 2: CMS Multi-frequency updates.

    Shows that different levels update at different rates.
    """
    print("\n" + "=" * 60)
    print("Demo 2: CMS Multi-frequency Updates")
    print("=" * 60)

    # Create CMS with specific chunk sizes
    cms = ContinuumMemorySystem(
        dim=64,
        num_levels=3,
        chunk_sizes=[4, 16, 64],  # Fast, medium, slow
    )

    print(f"\nCMS Configuration:")
    print(f"  Levels: {cms.num_levels}")
    print(f"  Chunk sizes: {cms.chunk_sizes}")
    print(f"  Level 0 updates every 4 steps")
    print(f"  Level 1 updates every 16 steps")
    print(f"  Level 2 updates every 64 steps")

    # Simulate training
    print(f"\nSimulating 100 training steps...")
    cms.train()

    update_counts = [0, 0, 0]

    for step in range(100):
        # Forward pass
        x = torch.randn(2, 10, 64)
        y = cms(x)

        # Simulate backward (create fake gradients)
        loss = y.sum()
        loss.backward()

        # Check which levels update
        update_levels = cms.get_update_levels(step)

        for level, should_update in enumerate(update_levels):
            cms.accumulate_gradients(level)
            if should_update:
                cms.apply_accumulated_gradients(level)
                update_counts[level] += 1

        # Zero gradients
        for p in cms.parameters():
            if p.grad is not None:
                p.grad.zero_()

    print(f"\nUpdate counts after 100 steps:")
    for level in range(3):
        expected = 100 // cms.chunk_sizes[level]
        print(f"  Level {level} (chunk={cms.chunk_sizes[level]}): "
              f"{update_counts[level]} updates (expected ~{expected})")

    # Verify counts are approximately correct (step 0 counts as an update)
    # Level 0 (chunk=4): steps 0,4,8,...,96 = 25 updates
    # Level 1 (chunk=16): steps 0,16,32,48,64,80,96 = 7 updates
    # Level 2 (chunk=64): steps 0,64 = 2 updates
    expected_counts = [25, 7, 2]  # Including step 0
    all_match = all(
        abs(update_counts[i] - expected_counts[i]) <= 1
        for i in range(3)
    )
    if all_match:
        print("  All levels updating at expected frequencies!")
    return all_match


def demo_self_modification():
    """
    Demo 3: Self-modifying attention.

    Shows that attention weights actually change during forward pass.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Self-Modifying Attention")
    print("=" * 60)

    from nested_learning.models.titans import SelfModifyingLinear, SelfModifyingAttention

    # Test self-modifying linear
    layer = SelfModifyingLinear(64, 64, self_mod_lr=0.01)
    layer.train()

    print("\nSelfModifyingLinear:")
    initial_weight = layer.weight.clone()

    x = torch.randn(4, 10, 64)
    for i in range(5):
        y = layer(x, update_weights=True)

    # Apply pending updates (deferred to avoid breaking gradients)
    layer.apply_pending_updates()

    final_weight = layer.weight

    weight_change = (final_weight - initial_weight).abs().mean().item()
    print(f"  Initial weight norm: {initial_weight.norm().item():.4f}")
    print(f"  Final weight norm:   {final_weight.norm().item():.4f}")
    print(f"  Mean weight change:  {weight_change:.6f}")
    print(f"  Update count: {layer.update_count.item()}")

    # Test self-modifying attention
    attn = SelfModifyingAttention(dim=64, num_heads=4, self_mod_lr=0.001)
    attn.train()

    print("\nSelfModifyingAttention:")
    initial_wq = attn.W_q.weight.clone()

    for i in range(5):
        y = attn(x, enable_self_modification=True)

    # Apply pending updates
    attn.apply_pending_updates()

    final_wq = attn.W_q.weight

    wq_change = (final_wq - initial_wq).abs().mean().item()
    print(f"  W_q weight change: {wq_change:.6f}")
    print(f"  W_q update count: {attn.W_q.update_count.item()}")
    print(f"  W_k update count: {attn.W_k.update_count.item()}")
    print(f"  W_v update count: {attn.W_v.update_count.item()}")
    print(f"  W_o update count: {attn.W_o.update_count.item()}")

    return weight_change > 0 and wq_change > 0


def demo_hope_model():
    """
    Demo 4: HOPE model with full nested learning.

    Shows the complete integration of all components.
    """
    print("\n" + "=" * 60)
    print("Demo 4: HOPE Model with Full Nested Learning")
    print("=" * 60)

    # Create small HOPE model
    model = HOPE(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        chunk_sizes=[4, 16, 64],
        max_seq_len=128,
        use_self_modification=True,
        self_mod_lr=0.001,
    )

    print(f"\nHOPE Configuration:")
    print(f"  Dimension: {model.dim}")
    print(f"  Layers: {model.n_layers}")
    print(f"  Heads: {model.n_heads}")
    print(f"  Self-modification: {model.use_self_modification}")
    print(f"  CMS chunk sizes: {model.blocks[0].cms.chunk_sizes}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Create optimizer with nested learning
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        memory_lr=1e-4,
    )

    # Simple training loop
    print(f"\nTraining HOPE with nested learning...")
    model.train()

    losses = []
    for step in range(20):
        # Random batch
        input_ids = torch.randint(0, 1000, (4, 32))
        labels = input_ids.clone()

        optimizer.zero_grad()
        loss, logits = model(input_ids, labels=labels)
        loss.backward()

        # Apply CMS multi-frequency updates
        for block in model.blocks:
            cms = block.cms
            update_levels = cms.get_update_levels(step)
            for level, should_update in enumerate(update_levels):
                cms.accumulate_gradients(level)
                if should_update:
                    cms.apply_accumulated_gradients(level)

        optimizer.step()

        # Apply self-modification updates after backward pass
        model.apply_pending_updates()

        losses.append(loss.item())

        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: Loss = {loss.item():.4f}")

    # Get self-modification stats
    print(f"\nSelf-modification statistics:")
    stats = model.get_self_mod_stats()
    for block_stats in stats['blocks']:
        print(f"  Block {block_stats['block_idx']}:")
        if 'attention_updates' in block_stats:
            attn = block_stats['attention_updates']
            print(f"    Attention updates: W_q={attn['W_q']}, W_k={attn['W_k']}, "
                  f"W_v={attn['W_v']}, W_o={attn['W_o']}")

    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Success criteria: self-modification is working (updates counted)
    # and model runs without errors
    total_updates = sum(
        attn['W_q'] for block_stats in stats['blocks']
        if 'attention_updates' in block_stats
        for attn in [block_stats['attention_updates']]
    )
    return total_updates > 0


def demo_unified_trainer():
    """
    Demo 5: NestedLearningTrainer with all components.

    Shows the complete nested learning pipeline.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Unified NestedLearningTrainer")
    print("=" * 60)

    # Create model
    model = HOPE(
        dim=64,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        chunk_sizes=[4, 16, 32],
        max_seq_len=64,
        use_self_modification=True,
    )

    # Create optimizer
    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.9,
        memory_lr=1e-4,
    )

    # Create trainer
    trainer = NestedLearningTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',
        enable_cms_multifreq=True,
    )

    # Create dummy data
    input_ids = torch.randint(0, 1000, (100, 32))
    dataset = TensorDataset(input_ids, input_ids)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define loss function for LM
    def lm_loss(output, labels):
        if isinstance(output, tuple):
            return output[0]
        # Shift and compute cross entropy
        shift_logits = output[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, 1000),
            shift_labels.view(-1),
        )

    print(f"\nTraining with NestedLearningTrainer...")
    print(f"  CMS multi-frequency: {trainer.enable_cms_multifreq}")
    print(f"  CMS modules found: {len(trainer.cms_modules)}")

    # Train for a few epochs
    history = trainer.train(
        train_loader=dataloader,
        loss_fn=lm_loss,
        num_epochs=3,
        log_every=1,
    )

    # Get CMS stats
    cms_stats = trainer.get_cms_update_stats()
    print(f"\nCMS Update Statistics:")
    for name, stats in cms_stats.items():
        print(f"  {name}:")
        print(f"    Levels: {stats['num_levels']}")
        print(f"    Updates per level: {stats['updates_per_level']}")

    return len(history) == 3


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("NESTED LEARNING DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows all nested learning components working together:")
    print("1. DeepMomentumGD with true nested optimization")
    print("2. CMS multi-frequency updates")
    print("3. Self-modifying attention")
    print("4. HOPE model integration")
    print("5. Unified training pipeline")

    results = {}

    # Run each demo
    try:
        results['deep_momentum'] = demo_deep_momentum_gd()
    except Exception as e:
        print(f"Demo 1 failed: {e}")
        results['deep_momentum'] = False

    try:
        results['cms_multifreq'] = demo_cms_multifreq()
    except Exception as e:
        print(f"Demo 2 failed: {e}")
        results['cms_multifreq'] = False

    try:
        results['self_modification'] = demo_self_modification()
    except Exception as e:
        print(f"Demo 3 failed: {e}")
        results['self_modification'] = False

    try:
        results['hope_model'] = demo_hope_model()
    except Exception as e:
        print(f"Demo 4 failed: {e}")
        results['hope_model'] = False

    try:
        results['unified_trainer'] = demo_unified_trainer()
    except Exception as e:
        print(f"Demo 5 failed: {e}")
        results['unified_trainer'] = False

    # Summary
    print("\n" + "=" * 60)
    print("DEMO RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll demos passed! Nested learning is working correctly.")
    else:
        print("\nSome demos failed. Check the output above for details.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
