"""
Training utilities for Continuum Memory System with multi-frequency updates.

This implements the paper's multi-timescale update schedule where different
memory levels update at different frequencies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm


class ContinuumMemoryTrainer:
    """
    Trainer that properly handles multi-frequency updates for ContinuumMemorySystem.

    Each level of the CMS updates at a different frequency (chunk_size).
    This trainer:
    1. Accumulates gradients for each level
    2. Applies updates only when step_count % chunk_size == 0
    3. Uses averaged gradients for stability
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
    ):
        """
        Args:
            model: Model containing ContinuumMemorySystem modules
            optimizer: Base optimizer for standard parameters
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.step_count = 0

        # Find all CMS modules in model
        self.cms_modules = self._find_cms_modules()

        print(f"Found {len(self.cms_modules)} ContinuumMemorySystem modules")
        for name, cms in self.cms_modules.items():
            print(f"  {name}: {cms.num_levels} levels, chunk_sizes={cms.chunk_sizes}")

    def _find_cms_modules(self) -> Dict[str, nn.Module]:
        """Find all ContinuumMemorySystem modules in the model."""
        from nested_learning.memory import ContinuumMemorySystem

        cms_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, ContinuumMemorySystem):
                cms_modules[name] = module

        return cms_modules

    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: callable,
    ) -> float:
        """
        Perform one training step with multi-frequency updates.

        Args:
            batch: Input batch
            labels: Target labels
            loss_fn: Loss function

        Returns:
            Loss value
        """
        self.model.train()
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        output = self.model(batch)
        loss = loss_fn(output, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Handle multi-frequency updates for CMS modules
        for name, cms in self.cms_modules.items():
            self._apply_multifreq_update(cms, self.step_count)

        # Update standard parameters
        self.optimizer.step()

        self.step_count += 1

        return loss.item()

    def _apply_multifreq_update(
        self,
        cms: nn.Module,
        step: int,
    ):
        """
        Apply multi-frequency gradient updates to CMS.

        Args:
            cms: ContinuumMemorySystem module
            step: Current training step
        """
        # Get which levels should update at this step
        update_levels = cms.get_update_levels(step)

        for level_idx, should_update in enumerate(update_levels):
            # Accumulate gradients for all levels
            cms.accumulate_gradients(level_idx)

            # Apply accumulated gradients for levels that should update
            if should_update:
                cms.apply_accumulated_gradients(level_idx)

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: callable,
        epoch: int,
    ) -> float:
        """
        Train for one epoch with multi-frequency updates.

        Args:
            dataloader: Training dataloader
            loss_fn: Loss function
            epoch: Current epoch number

        Returns:
            Average loss for epoch
        """
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch, labels in pbar:
            loss = self.train_step(batch, labels, loss_fn)
            total_loss += loss
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss, 'avg_loss': total_loss / num_batches})

        return total_loss / num_batches

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: callable,
    ) -> float:
        """
        Evaluate model on validation set.

        Args:
            dataloader: Validation dataloader
            loss_fn: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch, labels in dataloader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.model(batch)
                loss = loss_fn(output, labels)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        loss_fn: callable,
        num_epochs: int,
        log_every: int = 1,
    ) -> List[Dict[str, float]]:
        """
        Full training loop with multi-frequency updates.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader (optional)
            loss_fn: Loss function
            num_epochs: Number of epochs to train
            log_every: Log every N epochs

        Returns:
            Training history (list of dicts with 'train_loss', 'val_loss', etc.)
        """
        history = []

        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Multi-frequency update schedule:")
        for name, cms in self.cms_modules.items():
            print(f"  {name}: {cms.chunk_sizes}")

        for epoch in range(1, num_epochs + 1):
            # Train epoch
            train_loss = self.train_epoch(train_loader, loss_fn, epoch)

            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss_fn)

            # Log
            if epoch % log_every == 0:
                log_str = f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    log_str += f", Val Loss: {val_loss:.4f}"
                print(log_str)

            # Record history
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'step_count': self.step_count,
            })

        return history


def visualize_update_schedule(
    chunk_sizes: List[int],
    num_steps: int = 1000,
):
    """
    Visualize which levels update at which steps.

    Args:
        chunk_sizes: List of chunk sizes for each level
        num_steps: Number of steps to visualize
    """
    import matplotlib.pyplot as plt
    import numpy as np

    num_levels = len(chunk_sizes)
    update_matrix = np.zeros((num_levels, num_steps))

    for step in range(num_steps):
        for level, chunk_size in enumerate(chunk_sizes):
            if step % chunk_size == 0:
                update_matrix[level, step] = 1

    plt.figure(figsize=(15, 6))
    plt.imshow(update_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Update?')
    plt.xlabel('Training Step')
    plt.ylabel('Memory Level')
    plt.title(f'Multi-Frequency Update Schedule\nChunk Sizes: {chunk_sizes}')
    plt.yticks(range(num_levels), [f'Level {i} (C={c})' for i, c in enumerate(chunk_sizes)])
    plt.tight_layout()
    plt.savefig('results/update_schedule.png', dpi=150)
    print("Saved update schedule visualization to results/update_schedule.png")


# Example usage function

def create_continuum_model_example():
    """
    Create example model with ContinuumMemorySystem for testing.
    """
    from nested_learning.memory import ContinuumMemorySystem

    class SimpleModelWithCMS(nn.Module):
        def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
            super().__init__()

            self.input_proj = nn.Linear(input_dim, hidden_dim)

            # Continuum Memory System with multi-frequency updates
            self.cms = ContinuumMemorySystem(
                dim=hidden_dim,
                num_levels=3,
                chunk_sizes=[8, 32, 128],  # Different update frequencies
            )

            self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Flatten input
            x = x.view(x.size(0), -1)

            # Project
            x = self.input_proj(x)
            x = torch.relu(x)

            # Add sequence dimension for CMS
            x = x.unsqueeze(1)  # (batch, 1, dim)

            # Process through continuum memory
            x = self.cms(x)

            # Remove sequence dimension
            x = x.squeeze(1)

            # Output
            return self.output(x)

    return SimpleModelWithCMS()
