"""Training utilities."""

import torch
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device='cuda', max_grad_norm=1.0):
    """
    Train for one epoch.

    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else input_ids

        # Forward pass
        loss, _ = model(input_ids, labels=labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, device='cuda'):
    """
    Evaluate the model.

    Args:
        model: The model to evaluate
        dataloader: Evaluation dataloader
        device: Device to evaluate on

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device) if 'labels' in batch else input_ids

        # Forward pass
        loss, _ = model(input_ids, labels=labels)

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches