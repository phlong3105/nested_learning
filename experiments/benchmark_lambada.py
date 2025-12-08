#!/usr/bin/env python3
"""
LAMBADA Benchmark - Zero-shot Last Word Prediction

LAMBADA tests a model's ability to use long-range context to predict
the final word of a passage. This is particularly relevant for the
Nested Learning paper because:
1. CMS multi-frequency updates should help capture long-range dependencies
2. Self-modifying attention adapts online to the context

The paper reports HOPE outperforms baselines on this long-range task.

Metric: Accuracy (percentage of passages where the model correctly
        predicts the final word)

Usage:
    # Evaluate a model on LAMBADA
    python experiments/benchmark_lambada.py --checkpoint path/to/model.pt

    # Quick test
    python experiments/benchmark_lambada.py --test

    # Evaluate different model sizes
    python experiments/benchmark_lambada.py --size 760M --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nested_learning.models import HOPE

# Optional imports
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# Model configurations
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


def parse_args():
    parser = argparse.ArgumentParser(description='LAMBADA Benchmark')

    # Model
    parser.add_argument('--size', type=str, default='340M',
                        choices=['340M', '760M', '1.3B'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')

    # Evaluation
    parser.add_argument('--max_length', type=int, default=None,
                        help='Max context length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')

    # System
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)

    # Logging
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project', type=str, default='nested-learning-lambada')

    # Testing
    parser.add_argument('--test', action='store_true',
                        help='Quick test mode')

    return parser.parse_args()


def load_lambada_dataset(tokenizer, max_length=1024, test_mode=False):
    """
    Load and prepare LAMBADA dataset.

    LAMBADA format: Each example is a passage where the task is to
    predict the last word given the preceding context.

    Returns:
        List of (context_ids, target_word, target_id) tuples
    """
    if not HAS_HF:
        raise ImportError("Please install datasets: pip install datasets")

    print("Loading LAMBADA dataset...")
    dataset = load_dataset('lambada', split='test')

    if test_mode:
        dataset = dataset.select(range(min(100, len(dataset))))

    examples = []
    skipped = 0

    for item in tqdm(dataset, desc='Preparing examples'):
        text = item['text']

        # Split into context and target (last word)
        words = text.split()
        if len(words) < 2:
            skipped += 1
            continue

        target_word = words[-1]
        context = ' '.join(words[:-1])

        # Tokenize context
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        # Tokenize target (with space prefix to get correct tokenization)
        target_with_space = ' ' + target_word
        target_ids = tokenizer.encode(target_with_space, add_special_tokens=False)

        if len(target_ids) == 0:
            skipped += 1
            continue

        # Get first token of target (main prediction target)
        target_id = target_ids[0]

        # Truncate context if needed
        if len(context_ids) > max_length - 1:
            context_ids = context_ids[-(max_length - 1):]

        examples.append({
            'context_ids': context_ids,
            'target_word': target_word,
            'target_id': target_id,
            'full_target_ids': target_ids,
        })

    print(f"Prepared {len(examples)} examples (skipped {skipped})")
    return examples


def evaluate_lambada(model, examples, tokenizer, device, batch_size=1):
    """
    Evaluate model on LAMBADA dataset.

    For each example:
    1. Feed context to model
    2. Get prediction for next token
    3. Check if predicted token matches target

    Returns:
        Dictionary with accuracy and other metrics
    """
    model.eval()
    correct = 0
    correct_full = 0  # Correct when considering full target word
    total = 0

    with torch.no_grad():
        for example in tqdm(examples, desc='Evaluating'):
            context_ids = example['context_ids']
            target_id = example['target_id']
            full_target_ids = example['full_target_ids']

            # Prepare input
            input_ids = torch.tensor([context_ids], device=device)

            # Forward pass
            _, logits = model(input_ids)

            # Get prediction for next token (last position)
            next_token_logits = logits[0, -1, :]
            predicted_id = next_token_logits.argmax().item()

            # Check accuracy
            if predicted_id == target_id:
                correct += 1

            # Check if we predict the full target word correctly
            # (greedy decoding for length of target)
            predicted_ids = []
            current_input = input_ids

            for _ in range(len(full_target_ids)):
                _, logits = model(current_input)
                next_id = logits[0, -1, :].argmax().item()
                predicted_ids.append(next_id)
                current_input = torch.cat([
                    current_input,
                    torch.tensor([[next_id]], device=device)
                ], dim=1)

            if predicted_ids == full_target_ids:
                correct_full += 1

            total += 1

    accuracy = correct / total * 100
    accuracy_full = correct_full / total * 100

    return {
        'accuracy': accuracy,
        'accuracy_full_word': accuracy_full,
        'correct': correct,
        'correct_full': correct_full,
        'total': total,
    }


def evaluate_lambada_with_self_mod(model, examples, tokenizer, device):
    """
    Evaluate with self-modification enabled.

    This tests whether the model's online weight adaptation helps
    with LAMBADA's long-range dependencies.
    """
    model.eval()

    # Enable self-modification for inference
    if hasattr(model, 'enable_self_modification'):
        model.enable_self_modification()

    correct = 0
    total = 0

    with torch.no_grad():
        for example in tqdm(examples, desc='Evaluating (self-mod)'):
            context_ids = example['context_ids']
            target_id = example['target_id']

            # Reset any accumulated state
            if hasattr(model, 'reset_memory'):
                model.reset_memory()

            # Prepare input
            input_ids = torch.tensor([context_ids], device=device)

            # Forward pass with self-modification
            _, logits = model(input_ids)

            # Apply pending updates (self-modification)
            if hasattr(model, 'apply_pending_updates'):
                model.apply_pending_updates()

            # Get prediction
            next_token_logits = logits[0, -1, :]
            predicted_id = next_token_logits.argmax().item()

            if predicted_id == target_id:
                correct += 1

            total += 1

    accuracy = correct / total * 100

    return {
        'accuracy_self_mod': accuracy,
        'correct_self_mod': correct,
        'total': total,
    }


def create_model(size, device, checkpoint_path=None):
    """Create HOPE model and optionally load checkpoint."""
    config = MODEL_CONFIGS[size]

    model = HOPE(
        dim=config['dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        vocab_size=config['vocab_size'],
        chunk_sizes=config['chunk_sizes'],
        max_seq_len=config['max_seq_len'],
        dropout=0.0,  # No dropout for evaluation
        use_self_modification=True,
    )

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created HOPE-{size} with {n_params / 1e6:.1f}M parameters")

    return model


def main():
    args = parse_args()

    if not HAS_HF:
        print("Error: transformers and datasets required")
        print("Install with: pip install transformers datasets")
        return

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Get config
    config = MODEL_CONFIGS[args.size]
    max_length = args.max_length or config['max_seq_len']

    # Initialize wandb
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.project,
            config=vars(args),
        )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    examples = load_lambada_dataset(
        tokenizer,
        max_length=max_length,
        test_mode=args.test,
    )

    # Create model
    model = create_model(args.size, args.device, args.checkpoint)

    # Standard evaluation
    print("\n" + "=" * 50)
    print("Standard Evaluation")
    print("=" * 50)

    results = evaluate_lambada(
        model, examples, tokenizer, args.device, args.batch_size
    )

    print(f"\nResults:")
    print(f"  Accuracy (first token): {results['accuracy']:.2f}%")
    print(f"  Accuracy (full word):   {results['accuracy_full_word']:.2f}%")
    print(f"  Correct: {results['correct']}/{results['total']}")

    # Evaluation with self-modification
    print("\n" + "=" * 50)
    print("Evaluation with Self-Modification")
    print("=" * 50)

    results_selfmod = evaluate_lambada_with_self_mod(
        model, examples, tokenizer, args.device
    )

    print(f"\nResults (self-mod):")
    print(f"  Accuracy: {results_selfmod['accuracy_self_mod']:.2f}%")
    print(f"  Correct: {results_selfmod['correct_self_mod']}/{results_selfmod['total']}")

    # Combine results
    all_results = {**results, **results_selfmod}

    if args.wandb and HAS_WANDB:
        wandb.log(all_results)
        wandb.finish()

    # Final summary
    print("\n" + "=" * 60)
    print("LAMBADA Benchmark Results")
    print("=" * 60)
    print(f"Model: HOPE-{args.size}")
    print(f"Checkpoint: {args.checkpoint or 'None (random init)'}")
    print(f"\nAccuracy (first token): {results['accuracy']:.2f}%")
    print(f"Accuracy (full word):   {results['accuracy_full_word']:.2f}%")
    print(f"Accuracy (self-mod):    {results_selfmod['accuracy_self_mod']:.2f}%")
    print("=" * 60)

    return all_results


if __name__ == '__main__':
    main()
