"""
Synthetic Pattern Learning Benchmark

Tests DeepMomentumGD vs standard SGD on a simple pattern learning task.
This demonstrates the advantage of learned gradient transformation.

Task: Learn to predict a nonlinear function from noisy data.
The memory modules should learn to compress gradients effectively.

Usage:
    python experiments/benchmark_pattern_learning.py
    python experiments/benchmark_pattern_learning.py --l2-mode  # Paper-exact mode
"""

import torch
import torch.nn as nn
import argparse
import time
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, 'src')

from nested_learning.optimizers import DeepMomentumGD, SimpleMomentumGD


def create_synthetic_data(
    n_samples: int = 1000,
    input_dim: int = 20,
    noise_std: float = 0.1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic regression data with nonlinear patterns."""
    torch.manual_seed(seed)

    X = torch.randn(n_samples, input_dim)

    # Nonlinear target function
    y = (
        torch.sin(X[:, 0] * 2) +
        torch.cos(X[:, 1] * 3) +
        X[:, 2] ** 2 +
        X[:, 3] * X[:, 4] +
        torch.tanh(X[:, 5:10].sum(dim=1))
    )
    y = y.unsqueeze(1)

    # Add noise
    y = y + torch.randn_like(y) * noise_std

    return X, y


class SimpleNet(nn.Module):
    """Simple MLP for regression."""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 32,
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Shuffle data
    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]

    for i in range(0, X.size(0), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]

        optimizer.zero_grad()
        pred = model(batch_X)
        loss = nn.functional.mse_loss(pred, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Evaluate model and return MSE loss."""
    model.eval()
    with torch.no_grad():
        pred = model(X)
        loss = nn.functional.mse_loss(pred, y)
    return loss.item()


def run_benchmark(
    optimizer_name: str,
    optimizer_cls,
    optimizer_kwargs: Dict,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_epochs: int = 100,
    seed: int = 42,
) -> Dict:
    """Run benchmark for a single optimizer."""
    torch.manual_seed(seed)

    model = SimpleNet()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    train_losses = []
    test_losses = []
    times = []

    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        train_loss = train_epoch(model, optimizer, X_train, y_train)
        epoch_time = time.time() - epoch_start

        test_loss = evaluate(model, X_test, y_test)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        times.append(epoch_time)

        if (epoch + 1) % 20 == 0:
            print(f"  {optimizer_name} Epoch {epoch+1}: train={train_loss:.4f}, test={test_loss:.4f}")

    total_time = time.time() - start_time

    return {
        'optimizer': optimizer_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': min(test_losses),
        'total_time': total_time,
        'avg_epoch_time': sum(times) / len(times),
    }


def main():
    parser = argparse.ArgumentParser(description='Pattern Learning Benchmark')
    parser.add_argument('--n-train', type=int, default=800, help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=200, help='Number of test samples')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--l2-mode', action='store_true', help='Use paper-exact L2 regression mode for DMGD')
    args = parser.parse_args()

    print("=" * 60)
    print("Pattern Learning Benchmark: DMGD vs SGD")
    print("=" * 60)
    print(f"Training samples: {args.n_train}")
    print(f"Test samples: {args.n_test}")
    print(f"Epochs: {args.epochs}")
    print(f"DMGD mode: {'L2 regression (paper-exact)' if args.l2_mode else 'Surrogate (default)'}")
    print()

    # Create data
    X, y = create_synthetic_data(n_samples=args.n_train + args.n_test, seed=args.seed)
    X_train, X_test = X[:args.n_train], X[args.n_train:]
    y_train, y_test = y[:args.n_train], y[args.n_train:]

    results = []

    # SGD with momentum
    print("Training with SGD + Momentum...")
    sgd_result = run_benchmark(
        'SGD+Momentum',
        SimpleMomentumGD,
        {'lr': 0.01, 'momentum': 0.9},
        X_train, y_train, X_test, y_test,
        n_epochs=args.epochs,
        seed=args.seed,
    )
    results.append(sgd_result)

    # DeepMomentumGD (surrogate mode)
    print("\nTraining with DeepMomentumGD (surrogate)...")
    dmgd_kwargs = {
        'lr': 0.01,
        'momentum': 0.9,
        'memory_lr': 0.001,
        'use_shared_memory': True,
        'internal_loss_mode': 'surrogate',
    }
    dmgd_result = run_benchmark(
        'DMGD (surrogate)',
        DeepMomentumGD,
        dmgd_kwargs,
        X_train, y_train, X_test, y_test,
        n_epochs=args.epochs,
        seed=args.seed,
    )
    results.append(dmgd_result)

    # DeepMomentumGD (L2 regression mode) - if requested
    if args.l2_mode:
        print("\nTraining with DeepMomentumGD (L2 regression)...")
        dmgd_l2_kwargs = {
            'lr': 0.01,
            'momentum': 0.9,
            'memory_lr': 0.001,
            'use_shared_memory': True,
            'internal_loss_mode': 'l2_regression',
            'l2_projection_lr': 0.01,
        }
        dmgd_l2_result = run_benchmark(
            'DMGD (L2 regression)',
            DeepMomentumGD,
            dmgd_l2_kwargs,
            X_train, y_train, X_test, y_test,
            n_epochs=args.epochs,
            seed=args.seed,
        )
        results.append(dmgd_l2_result)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Optimizer':<25} {'Final Train':<12} {'Final Test':<12} {'Best Test':<12} {'Time (s)':<10}")
    print("-" * 60)

    for r in results:
        print(f"{r['optimizer']:<25} {r['final_train_loss']:<12.4f} {r['final_test_loss']:<12.4f} {r['best_test_loss']:<12.4f} {r['total_time']:<10.2f}")

    print("\n" + "=" * 60)

    # Check if DMGD beats SGD
    sgd_best = sgd_result['best_test_loss']
    dmgd_best = dmgd_result['best_test_loss']

    if dmgd_best < sgd_best:
        improvement = (sgd_best - dmgd_best) / sgd_best * 100
        print(f"DMGD outperforms SGD by {improvement:.1f}% (best test loss)")
    else:
        print("Note: SGD performed better on this run. Results vary by seed.")

    print("=" * 60)


if __name__ == '__main__':
    main()
