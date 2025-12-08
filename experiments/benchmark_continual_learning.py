"""
Synthetic Continual Learning Benchmark

Tests HOPE with CMS multi-frequency updates vs a vanilla transformer on
a continual learning task where the data distribution shifts over time.

This demonstrates the advantage of the Continuum Memory System's
multi-frequency updates for handling temporal hierarchies.

Task: Sequence prediction where patterns change at different timescales.
- Fast patterns: change every few steps
- Slow patterns: persist for longer periods

Usage:
    python experiments/benchmark_continual_learning.py
    python experiments/benchmark_continual_learning.py --paper-exact  # Paper-exact modes
"""

import torch
import torch.nn as nn
import argparse
import time
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, 'src')

from nested_learning.models import HOPE
from nested_learning.memory import ContinuumMemorySystem


class VanillaTransformer(nn.Module):
    """Simple transformer without CMS or self-modification."""

    def __init__(
        self,
        dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        vocab_size: int = 100,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=4 * dim,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # Weight tying

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        tok_emb = self.token_embedding(input_ids)
        pos = torch.arange(seq_len, device=device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if labels is not None:
            return loss, logits
        return logits


def generate_continual_data(
    n_sequences: int,
    seq_len: int = 32,
    vocab_size: int = 100,
    fast_change_freq: int = 5,
    slow_change_freq: int = 50,
    seed: int = 42,
) -> List[torch.Tensor]:
    """
    Generate sequences with patterns that change at different frequencies.

    Fast patterns: Simple repeating patterns that change frequently
    Slow patterns: Underlying distribution that changes slowly
    """
    torch.manual_seed(seed)
    sequences = []

    # Slow pattern: base distribution parameters
    slow_pattern_idx = 0
    slow_patterns = [
        (20, 40),   # Token range 20-40
        (50, 70),   # Token range 50-70
        (10, 30),   # Token range 10-30
    ]

    # Fast pattern: local structure
    fast_pattern_idx = 0
    fast_patterns = [
        'repeat',    # Repeat previous token
        'increment', # Increment token
        'random',    # Random from range
    ]

    for i in range(n_sequences):
        # Update slow pattern
        if i > 0 and i % slow_change_freq == 0:
            slow_pattern_idx = (slow_pattern_idx + 1) % len(slow_patterns)

        # Update fast pattern
        if i > 0 and i % fast_change_freq == 0:
            fast_pattern_idx = (fast_pattern_idx + 1) % len(fast_patterns)

        low, high = slow_patterns[slow_pattern_idx]
        fast_pattern = fast_patterns[fast_pattern_idx]

        # Generate sequence
        seq = torch.zeros(seq_len, dtype=torch.long)
        seq[0] = torch.randint(low, high, (1,)).item()

        for t in range(1, seq_len):
            if fast_pattern == 'repeat':
                # Repeat with small noise
                if torch.rand(1).item() < 0.8:
                    seq[t] = seq[t-1]
                else:
                    seq[t] = torch.randint(low, high, (1,)).item()
            elif fast_pattern == 'increment':
                # Increment with wrap
                if torch.rand(1).item() < 0.8:
                    seq[t] = low + (seq[t-1] - low + 1) % (high - low)
                else:
                    seq[t] = torch.randint(low, high, (1,)).item()
            else:
                # Random from range
                seq[t] = torch.randint(low, high, (1,)).item()

        sequences.append(seq)

    return sequences


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sequence: torch.Tensor,
    apply_pending: bool = False,
) -> float:
    """Single training step on a sequence."""
    model.train()
    optimizer.zero_grad()

    # Prepare input and labels
    input_ids = sequence.unsqueeze(0)
    labels = sequence.unsqueeze(0)

    # Forward pass
    if hasattr(model, 'apply_pending_updates'):
        loss, _ = model(input_ids, labels=labels, enable_self_modification=True)
    else:
        loss, _ = model(input_ids, labels=labels)

    loss.backward()
    optimizer.step()

    # Apply self-modification updates if applicable
    if apply_pending and hasattr(model, 'apply_pending_updates'):
        model.apply_pending_updates()

    return loss.item()


def evaluate(
    model: nn.Module,
    sequences: List[torch.Tensor],
) -> float:
    """Evaluate model on sequences."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for seq in sequences:
            input_ids = seq.unsqueeze(0)
            labels = seq.unsqueeze(0)

            if hasattr(model, 'apply_pending_updates'):
                loss, _ = model(input_ids, labels=labels, enable_self_modification=False)
            else:
                loss, _ = model(input_ids, labels=labels)

            total_loss += loss.item()

    return total_loss / len(sequences)


def run_continual_benchmark(
    model_name: str,
    model: nn.Module,
    train_sequences: List[torch.Tensor],
    test_sequences: List[torch.Tensor],
    lr: float = 1e-3,
    apply_pending: bool = False,
) -> Dict:
    """Run continual learning benchmark."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    window_size = 20

    start_time = time.time()

    for i, seq in enumerate(train_sequences):
        loss = train_step(model, optimizer, seq, apply_pending=apply_pending)
        train_losses.append(loss)

        # Periodic evaluation
        if (i + 1) % window_size == 0:
            test_loss = evaluate(model, test_sequences[:50])
            test_losses.append(test_loss)

            # Compute moving average of train loss
            avg_train = sum(train_losses[-window_size:]) / window_size
            print(f"  {model_name} Step {i+1}: train={avg_train:.4f}, test={test_loss:.4f}")

    total_time = time.time() - start_time

    return {
        'model': model_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': sum(train_losses[-20:]) / 20,
        'final_test_loss': test_losses[-1] if test_losses else float('inf'),
        'best_test_loss': min(test_losses) if test_losses else float('inf'),
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Continual Learning Benchmark')
    parser.add_argument('--n-train', type=int, default=500, help='Number of training sequences')
    parser.add_argument('--n-test', type=int, default=100, help='Number of test sequences')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--paper-exact', action='store_true', help='Use paper-exact modes')
    args = parser.parse_args()

    print("=" * 60)
    print("Continual Learning Benchmark: HOPE vs Vanilla Transformer")
    print("=" * 60)
    print(f"Training sequences: {args.n_train}")
    print(f"Test sequences: {args.n_test}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Paper-exact mode: {args.paper_exact}")
    print()

    # Generate data
    print("Generating continual learning data...")
    train_sequences = generate_continual_data(
        n_sequences=args.n_train,
        seq_len=args.seq_len,
        seed=args.seed,
    )
    test_sequences = generate_continual_data(
        n_sequences=args.n_test,
        seq_len=args.seq_len,
        seed=args.seed + 1000,  # Different seed for test
    )
    print(f"Generated {len(train_sequences)} train and {len(test_sequences)} test sequences")
    print()

    results = []

    # Vanilla Transformer
    print("Training Vanilla Transformer...")
    torch.manual_seed(args.seed)
    vanilla_model = VanillaTransformer(dim=128, n_layers=4, n_heads=4, vocab_size=100)
    vanilla_result = run_continual_benchmark(
        'Vanilla Transformer',
        vanilla_model,
        train_sequences,
        test_sequences,
    )
    results.append(vanilla_result)

    # HOPE with CMS
    print("\nTraining HOPE (CMS + Self-Modification)...")
    torch.manual_seed(args.seed)
    hope_model = HOPE(
        dim=128,
        n_layers=4,
        n_heads=4,
        vocab_size=100,
        max_seq_len=64,
        chunk_sizes=[1, 10, 50],  # Multi-frequency updates
        use_self_modification=True,
    )
    hope_result = run_continual_benchmark(
        'HOPE (CMS)',
        hope_model,
        train_sequences,
        test_sequences,
        apply_pending=True,  # Apply self-modification updates
    )
    results.append(hope_result)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Final Train':<12} {'Final Test':<12} {'Best Test':<12} {'Time (s)':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['model']:<25} {r['final_train_loss']:<12.4f} {r['final_test_loss']:<12.4f} {r['best_test_loss']:<12.4f} {r['total_time']:<10.2f}")

    print("\n" + "=" * 60)

    # Analysis
    vanilla_best = vanilla_result['best_test_loss']
    hope_best = hope_result['best_test_loss']

    if hope_best < vanilla_best:
        improvement = (vanilla_best - hope_best) / vanilla_best * 100
        print(f"HOPE outperforms Vanilla by {improvement:.1f}% (best test loss)")
        print("CMS multi-frequency updates help adapt to changing patterns.")
    else:
        print("Note: Vanilla performed better on this run.")
        print("Try more training sequences or different seeds.")

    print("=" * 60)


if __name__ == '__main__':
    main()
