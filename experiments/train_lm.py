#!/usr/bin/env python3
"""
Train HOPE model on language modeling task.

Example usage:
    python experiments/train_lm.py --model hope --size 760M --dataset wikitext
"""

import argparse
import torch
import wandb
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nested_learning.models import HOPE
from nested_learning.optimizers import DeepMomentumGD, DeltaRuleMomentum
from nested_learning.utils import train_epoch, evaluate, get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='Train HOPE language model')

    # Model args
    parser.add_argument('--model', type=str, default='hope', choices=['hope'])
    parser.add_argument('--size', type=str, default='340M',
                        choices=['340M', '760M', '1.3B'])
    parser.add_argument('--dim', type=int, default=None, help='Model dimension (overrides --size)')
    parser.add_argument('--n_layers', type=int, default=None, help='Number of layers')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')

    # Data args
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=1024)

    # Training args
    parser.add_argument('--optimizer', type=str, default='deep_momentum',
                        choices=['deep_momentum', 'delta_rule', 'adam'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    # System args
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)

    # Logging
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--project_name', type=str, default='nested-learning')
    parser.add_argument('--save_dir', type=str, default='checkpoints')

    return parser.parse_args()


def get_model_config(size):
    """Get model configuration based on size."""
    configs = {
        '340M': {'dim': 512, 'n_layers': 12, 'n_heads': 8},
        '760M': {'dim': 768, 'n_layers': 16, 'n_heads': 12},
        '1.3B': {'dim': 1024, 'n_layers': 24, 'n_heads': 16},
    }
    return configs[size]


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project=args.project_name, config=vars(args))

    # Get model config
    if args.dim is None:
        config = get_model_config(args.size)
        args.dim = config['dim']
        args.n_layers = args.n_layers or config['n_layers']
        args.n_heads = args.n_heads or config['n_heads']

    print(f"Training HOPE model: {args.size}")
    print(f"  dim={args.dim}, n_layers={args.n_layers}, n_heads={args.n_heads}")

    # Create model
    model = HOPE(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=50257,  # GPT-2 vocab size
        chunk_sizes=[16, 64, 256],  # Multi-frequency updates
        dropout=0.1,
    )

    model = model.to(args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {n_params / 1e6:.1f}M")

    # Create optimizer
    if args.optimizer == 'deep_momentum':
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            memory_depth=2,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'delta_rule':
        optimizer = DeltaRuleMomentum(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:  # adam
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    # Create dataloaders
    print("Loading datasets...")
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split='train',
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    val_loader = get_dataloader(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split='validation',
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    # Training loop
    best_val_loss = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            max_grad_norm=args.max_grad_norm,
        )

        # Evaluate
        val_loss = evaluate(model, val_loader, device=args.device)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log to wandb
        if args.wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = save_dir / f'hope_{args.size}_best.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args),
            }, save_path)
            print(f"Saved best model to {save_path}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()