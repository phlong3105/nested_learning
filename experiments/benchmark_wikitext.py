#!/usr/bin/env python3
"""
WikiText-103 Language Modeling Benchmark

Reproduces Table 1 results from the Nested Learning paper:
- Model: HOPE (340M, 760M, 1.3B parameters)
- Metric: Perplexity on WikiText-103 test set
- Comparison: HOPE vs Transformer vs baseline optimizers

Paper's HOPE configurations (from Appendix):
- HOPE-340M: dim=512, n_layers=12, n_heads=8
- HOPE-760M: dim=768, n_layers=16, n_heads=12
- HOPE-1.3B: dim=1024, n_layers=24, n_heads=16

Training settings from paper:
- Batch size: 32
- Learning rate: 3e-4 with cosine decay
- Warmup: 2000 steps
- Total steps: 100K
- Memory LR: 1e-4

Usage:
    # Full training run
    python experiments/benchmark_wikitext.py --size 340M --epochs 10

    # Quick test (1 epoch, small batch)
    python experiments/benchmark_wikitext.py --test

    # Evaluate pre-trained checkpoint
    python experiments/benchmark_wikitext.py --eval_only --checkpoint path/to/model.pt
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nested_learning.models import HOPE
from nested_learning.optimizers import DeepMomentumGD
from nested_learning.training import NestedLearningTrainer

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    HAS_HF = True
except ImportError:
    HAS_HF = False


# Model configurations matching paper
MODEL_CONFIGS = {
    '340M': {
        'dim': 512,
        'n_layers': 12,
        'n_heads': 8,
        'chunk_sizes': [16, 64, 256],
        'vocab_size': 50257,
        'max_seq_len': 1024,
    },
    '760M': {
        'dim': 768,
        'n_layers': 16,
        'n_heads': 12,
        'chunk_sizes': [32, 128, 512],
        'vocab_size': 50257,
        'max_seq_len': 1024,
    },
    '1.3B': {
        'dim': 1024,
        'n_layers': 24,
        'n_heads': 16,
        'chunk_sizes': [64, 256, 1024],
        'vocab_size': 50257,
        'max_seq_len': 2048,
    },
}

# Training configurations from paper
TRAIN_CONFIGS = {
    '340M': {
        'batch_size': 32,
        'lr': 3e-4,
        'memory_lr': 1e-4,
        'warmup_steps': 2000,
        'total_steps': 100000,
        'weight_decay': 0.01,
    },
    '760M': {
        'batch_size': 24,
        'lr': 2e-4,
        'memory_lr': 1e-4,
        'warmup_steps': 2000,
        'total_steps': 100000,
        'weight_decay': 0.01,
    },
    '1.3B': {
        'batch_size': 16,
        'lr': 1e-4,
        'memory_lr': 5e-5,
        'warmup_steps': 3000,
        'total_steps': 150000,
        'weight_decay': 0.01,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='WikiText-103 Benchmark')

    # Model
    parser.add_argument('--size', type=str, default='340M',
                        choices=['340M', '760M', '1.3B'],
                        help='Model size')

    # Data
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--dataset_config', type=str, default='wikitext-103-raw-v1')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Max sequence length (default: from config)')

    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')
    parser.add_argument('--memory_lr', type=float, default=None)
    parser.add_argument('--warmup_steps', type=int, default=None)
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max training steps (overrides epochs)')
    parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluate every N steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='deep_momentum',
                        choices=['deep_momentum', 'adam', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use cosine LR scheduler with warmup')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing for memory efficiency')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints/wikitext')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from or evaluate')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only evaluate, do not train')

    # Logging
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--project', type=str, default='nested-learning-wikitext')
    parser.add_argument('--run_name', type=str, default=None)

    # Testing
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode (1 epoch, small data)')

    return parser.parse_args()


def compute_perplexity(model, dataloader, device, max_batches=None):
    """
    Compute perplexity on a dataset.

    Perplexity = exp(average cross-entropy loss)

    Args:
        model: The language model
        dataloader: DataLoader with tokenized data
        device: Device to use
        max_batches: Maximum number of batches to evaluate (for speed)

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Computing perplexity')):
            if max_batches and i >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Forward pass
            loss, _ = model(input_ids, labels=labels)

            # Count non-padding tokens
            if attention_mask is not None:
                num_tokens = attention_mask.sum().item()
            else:
                num_tokens = input_ids.numel()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def create_wikitext_dataloader(
    dataset_name='wikitext',
    dataset_config='wikitext-103-raw-v1',
    split='train',
    tokenizer_name='gpt2',
    batch_size=8,
    max_length=1024,
    num_workers=4,
    test_mode=False,
):
    """
    Create a DataLoader for WikiText language modeling.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration
        split: Dataset split
        tokenizer_name: Tokenizer to use
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        test_mode: If True, use small subset for testing

    Returns:
        DataLoader
    """
    if not HAS_HF:
        raise ImportError("Please install transformers and datasets: "
                          "pip install transformers datasets")

    # Load dataset
    print(f"Loading {dataset_name}/{dataset_config} ({split})...")
    dataset = load_dataset(dataset_name, dataset_config, split=split)

    if test_mode:
        # Use small subset for testing
        dataset = dataset.select(range(min(1000, len(dataset))))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Filter out empty strings
    dataset = dataset.filter(lambda x: len(x['text'].strip()) > 0)

    # Tokenize
    def tokenize_function(examples):
        # Tokenize
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        # Labels are same as input_ids for language modeling
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc='Tokenizing',
    )

    tokenized_dataset.set_format(type='torch')

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


def create_model(size, device, gradient_checkpointing=False):
    """Create HOPE model with specified configuration."""
    config = MODEL_CONFIGS[size]

    model = HOPE(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        vocab_size=config['vocab_size'],
        chunk_sizes=config['chunk_sizes'],
        max_seq_len=config['max_seq_len'],
        dropout=0.1,
        use_self_modification=True,
    )

    # Enable gradient checkpointing if requested
    if gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created HOPE-{size} with {n_params / 1e6:.1f}M parameters")

    return model


def create_optimizer(model, args, train_config):
    """Create optimizer based on arguments."""
    lr = args.lr or train_config['lr']
    memory_lr = args.memory_lr or train_config['memory_lr']
    weight_decay = train_config['weight_decay']

    if args.optimizer == 'deep_momentum':
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            memory_lr=memory_lr,
            memory_hidden_dim=64,
            memory_depth=2,
            weight_decay=weight_decay,
            use_shared_memory=True,
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    return optimizer


def train_step(model, batch, optimizer, device, scaler=None):
    """
    Perform a single training step.

    Returns:
        loss value
    """
    model.train()

    input_ids = batch['input_ids'].to(device)
    labels = batch.get('labels', input_ids).to(device)

    optimizer.zero_grad()

    if scaler is not None:
        # Mixed precision training
        with torch.cuda.amp.autocast():
            loss, _ = model(input_ids, labels=labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, _ = model(input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Apply self-modification updates
    if hasattr(model, 'apply_pending_updates'):
        model.apply_pending_updates()

    return loss.item()


def train(model, train_loader, val_loader, test_loader, optimizer, scheduler, args):
    """
    Main training loop.

    Returns:
        Dictionary of final metrics
    """
    device = args.device
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Mixed precision scaler
    scaler = None
    if args.mixed_precision and device == 'cuda':
        scaler = torch.cuda.amp.GradScaler()

    # Training state
    global_step = 0
    best_val_ppl = float('inf')
    train_losses = []

    # Determine total steps
    steps_per_epoch = len(train_loader)
    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * steps_per_epoch

    print(f"\nTraining Configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Batch size: {args.batch_size or TRAIN_CONFIGS[args.size]['batch_size']}")
    print(f"  Learning rate: {args.lr or TRAIN_CONFIGS[args.size]['lr']}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Device: {device}")
    print()

    # Training loop
    epoch = 0
    pbar = tqdm(total=total_steps, desc='Training')

    while global_step < total_steps:
        epoch += 1

        for batch in train_loader:
            if global_step >= total_steps:
                break

            # Training step
            loss = train_step(model, batch, optimizer, device, scaler)
            train_losses.append(loss)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            pbar.update(1)

            # Logging
            if global_step % args.log_interval == 0:
                avg_loss = sum(train_losses[-100:]) / min(100, len(train_losses))
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ppl': f'{math.exp(avg_loss):.2f}',
                    'lr': f'{lr:.2e}',
                })

                if args.wandb and HAS_WANDB:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/perplexity': math.exp(avg_loss),
                        'train/lr': lr,
                        'step': global_step,
                    })

            # Evaluation
            if global_step % args.eval_interval == 0:
                val_ppl = compute_perplexity(
                    model, val_loader, device,
                    max_batches=50 if args.test else None
                )
                print(f"\nStep {global_step}: Validation Perplexity = {val_ppl:.2f}")

                if args.wandb and HAS_WANDB:
                    wandb.log({
                        'val/perplexity': val_ppl,
                        'step': global_step,
                    })

                # Save best model
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    checkpoint_path = save_dir / f'hope_{args.size}_best.pt'
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_perplexity': val_ppl,
                        'args': vars(args),
                    }, checkpoint_path)
                    print(f"  Saved best model (val_ppl={val_ppl:.2f})")

                model.train()

    pbar.close()

    # Final evaluation on test set
    print("\nFinal Evaluation on Test Set...")
    test_ppl = compute_perplexity(model, test_loader, device)
    print(f"Test Perplexity: {test_ppl:.2f}")

    if args.wandb and HAS_WANDB:
        wandb.log({
            'test/perplexity': test_ppl,
        })

    return {
        'test_perplexity': test_ppl,
        'best_val_perplexity': best_val_ppl,
        'final_train_loss': sum(train_losses[-100:]) / min(100, len(train_losses)),
    }


def main():
    args = parse_args()

    # Test mode overrides
    if args.test:
        args.epochs = 1
        args.batch_size = args.batch_size or 4
        args.eval_interval = 50
        args.log_interval = 10
        args.max_steps = 100
        print("Running in TEST MODE (quick validation)")

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Get configs
    model_config = MODEL_CONFIGS[args.size]
    train_config = TRAIN_CONFIGS[args.size]

    # Override with args
    batch_size = args.batch_size or train_config['batch_size']
    max_length = args.max_length or model_config['max_seq_len']

    # Initialize wandb
    if args.wandb and HAS_WANDB:
        run_name = args.run_name or f"hope-{args.size}-{args.optimizer}"
        wandb.init(
            project=args.project,
            name=run_name,
            config={
                **vars(args),
                'model_config': model_config,
                'train_config': train_config,
            }
        )

    # Create model
    model = create_model(args.size, args.device, args.gradient_checkpointing)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create data loaders
    print("\nLoading datasets...")
    train_loader = create_wikitext_dataloader(
        args.dataset, args.dataset_config, 'train',
        batch_size=batch_size, max_length=max_length,
        num_workers=args.num_workers, test_mode=args.test,
    )
    val_loader = create_wikitext_dataloader(
        args.dataset, args.dataset_config, 'validation',
        batch_size=batch_size, max_length=max_length,
        num_workers=args.num_workers, test_mode=args.test,
    )
    test_loader = create_wikitext_dataloader(
        args.dataset, args.dataset_config, 'test',
        batch_size=batch_size, max_length=max_length,
        num_workers=args.num_workers, test_mode=args.test,
    )

    # Evaluation only
    if args.eval_only:
        print("\nEvaluation Mode")
        val_ppl = compute_perplexity(model, val_loader, args.device)
        test_ppl = compute_perplexity(model, test_loader, args.device)
        print(f"Validation Perplexity: {val_ppl:.2f}")
        print(f"Test Perplexity: {test_ppl:.2f}")
        return

    # Create optimizer
    optimizer = create_optimizer(model, args, train_config)

    # Create scheduler
    scheduler = None
    if args.use_scheduler and HAS_HF:
        warmup_steps = args.warmup_steps or train_config['warmup_steps']
        total_steps = args.max_steps or (args.epochs * len(train_loader))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    # Train
    metrics = train(
        model, train_loader, val_loader, test_loader,
        optimizer, scheduler, args
    )

    # Print final results
    print("\n" + "=" * 60)
    print("WikiText-103 Benchmark Results")
    print("=" * 60)
    print(f"Model: HOPE-{args.size}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Test Perplexity: {metrics['test_perplexity']:.2f}")
    print(f"Best Validation Perplexity: {metrics['best_val_perplexity']:.2f}")
    print("=" * 60)

    if args.wandb and HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
