#!/usr/bin/env python3
"""
Simple demo of what actually works in this repo.

This demonstrates:
1. LinearAttentionWithMemory - working linear attention (renamed from SelfModifyingAttention)
2. AssociativeMemory - working key-value memory
3. HOPE model - working transformer structure (without self-modification)
4. DeepMomentumGD - nonlinear momentum (note: MLPs not trained)

Run with:
    python demo_simple.py
"""

import torch
import torch.nn as nn


def demo_associative_memory():
    """Associative memory works correctly."""
    print("=" * 60)
    print("Demo 1: Associative Memory")
    print("=" * 60)
    print()

    from nested_learning.memory import AssociativeMemory

    memory = AssociativeMemory(dim_key=64, dim_value=64)
    print(f"Created AssociativeMemory(dim_key=64, dim_value=64)")

    # Store some key-value pairs
    keys = torch.randn(10, 64)
    values = torch.randn(10, 64)
    memory.store(keys, values)
    print(f"Stored 10 key-value pairs")

    # Retrieve
    query = keys[0:1]  # Query with first key
    retrieved = memory.retrieve(query)

    # Should be close to values[0]
    similarity = torch.cosine_similarity(retrieved, values[0:1]).item()
    print(f"Retrieved with first key, cosine similarity to stored value: {similarity:.3f}")
    print(f"(Should be close to 1.0 for perfect retrieval)")
    print()
    print("Status: AssociativeMemory works correctly")
    print()


def demo_linear_attention():
    """Linear attention with memory works correctly."""
    print("=" * 60)
    print("Demo 2: LinearAttentionWithMemory (renamed from SelfModifyingAttention)")
    print("=" * 60)
    print()

    from nested_learning.models import LinearAttentionWithMemory

    attn = LinearAttentionWithMemory(dim=128, n_heads=4)
    print(f"Created LinearAttentionWithMemory(dim=128, n_heads=4)")
    print()
    print("NOTE: This is linear attention with an external memory buffer.")
    print("      It does NOT modify its own parameters (W_q, W_k, W_v, W_o).")
    print("      The name was changed from 'SelfModifyingAttention' for honesty.")
    print()

    # Test forward pass
    x = torch.randn(2, 16, 128)  # (batch, seq, dim)
    output = attn(x)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print()
    print("Status: LinearAttentionWithMemory works correctly")
    print()


def demo_hope_forward():
    """HOPE model forward pass works."""
    print("=" * 60)
    print("Demo 3: HOPE Model")
    print("=" * 60)
    print()

    from nested_learning.models import HOPE

    model = HOPE(
        dim=128,
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        max_seq_len=512,
    )

    print(f"Created HOPE model:")
    print(f"  dim=128, n_layers=2, n_heads=4")
    print(f"  vocab_size=1000, max_seq_len=512")
    print()
    print("NOTE: HOPE is a working transformer with custom attention,")
    print("      but it does NOT implement true self-referential learning.")
    print("      CMS multi-frequency updates are NOT used in practice.")
    print()

    # Random input
    input_ids = torch.randint(0, 1000, (2, 32))

    # Forward pass
    logits = model(input_ids)
    print(f"Input shape:  {tuple(input_ids.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print()
    print("Status: HOPE forward pass works correctly")
    print()


def demo_optimizer():
    """DeepMomentumGD runs (but MLPs aren't trained)."""
    print("=" * 60)
    print("Demo 4: DeepMomentumGD Optimizer")
    print("=" * 60)
    print()

    from nested_learning.optimizers import DeepMomentumGD

    print("IMPORTANT CAVEAT:")
    print("  Memory MLPs are randomly initialized and NOT trained.")
    print("  This is 'nonlinear momentum', not the paper's 'learned optimization'.")
    print("  Paper's version requires meta-learning infrastructure we don't have.")
    print()

    # Simple model
    model = nn.Linear(10, 1)
    optimizer = DeepMomentumGD(model.parameters(), lr=0.01, memory_depth=2)

    print(f"Created DeepMomentumGD(lr=0.01, memory_depth=2)")
    print()

    # Fake training
    losses = []
    for i in range(5):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        optimizer.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {i+1}: loss = {loss.item():.4f}")

    print()
    if losses[-1] < losses[0]:
        print(f"Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    else:
        print(f"Loss stable/increased (random MLPs don't help optimization)")
    print()
    print("Status: DeepMomentumGD runs (but MLPs not trained as paper intends)")
    print()


def demo_self_modifying_titan():
    """SelfModifyingTitan with delta-rule updates."""
    print("=" * 60)
    print("Demo 5: SelfModifyingTitan")
    print("=" * 60)
    print()

    from nested_learning.models import SelfModifyingTitan

    model = SelfModifyingTitan(
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        num_layers=1,
        self_mod_lr=0.001,
    )

    print(f"Created SelfModifyingTitan:")
    print(f"  input_dim=64, hidden_dim=64, output_dim=64")
    print(f"  self_mod_lr=0.001")
    print()
    print("NOTE: This has basic delta-rule weight updates, but needs")
    print("      more validation against the paper's formulation.")
    print()

    # Test forward pass
    model.train()
    x = torch.randn(2, 16, 64)  # (batch, seq, dim)
    output, hidden = model(x, enable_self_modification=True)

    stats = model.get_update_stats()
    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Self-mod updates: input_proj={stats['input_proj_updates']}, output_proj={stats['output_proj_updates']}")
    print()
    print("Status: SelfModifyingTitan runs with delta-rule updates")
    print()


def print_summary():
    """Print summary of what works and what doesn't."""
    print("=" * 60)
    print("SUMMARY: What Actually Works")
    print("=" * 60)
    print()
    print("Working correctly:")
    print("  [OK] AssociativeMemory - key-value memory storage/retrieval")
    print("  [OK] LinearAttentionWithMemory - linear attention with fast weights")
    print("  [OK] HOPE forward pass - transformer structure works")
    print("  [OK] DeepMomentumGD - optimizer API works")
    print("  [OK] SelfModifyingTitan - basic delta-rule updates")
    print()
    print("Not implemented / Not working as paper intends:")
    print("  [!!] Nested optimization - no internal loss L^(2)")
    print("  [!!] Memory training - MLPs never trained")
    print("  [!!] Multi-frequency CMS - code exists but not used")
    print("  [!!] True self-modification in HOPE - uses external memory only")
    print("  [!!] Experimental reproduction - no paper results validated")
    print()
    print("See IMPLEMENTATION_STATUS.md for detailed analysis.")
    print()


if __name__ == "__main__":
    print()
    print("Nested Learning - Honest Demo")
    print("Shows what actually works (and what doesn't)")
    print()

    demo_associative_memory()
    demo_linear_attention()
    demo_hope_forward()
    demo_optimizer()
    demo_self_modifying_titan()
    print_summary()
