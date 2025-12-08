"""
Unified Nested Learning Trainer

Integrates all components of the nested learning framework:
1. Deep momentum optimizer with true nested optimization (memory modules trained)
2. CMS multi-frequency updates (different levels update at different rates)
3. Optional meta-learning for optimizer improvement

This trainer ensures that all the paper's concepts work together:
- Outer loop: model parameter updates via learned optimizer
- Inner loop: memory module training via internal loss
- Multi-frequency: CMS levels updated at different timescales
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Tuple, Any
from tqdm import tqdm
import math


class NestedLearningTrainer:
    """
    Unified trainer for the nested learning framework.

    This trainer properly integrates:
    1. DeepMomentumGD with internal loss for memory training
    2. CMS multi-frequency gradient accumulation and application
    3. Optional validation-based meta-learning for optimizer tuning

    The key insight: everything updates at its own timescale
    - Model parameters: every step (fast)
    - Memory modules: every step via internal loss (fast, but different objective)
    - CMS level 0: frequently (fast context)
    - CMS level 1: less frequently (medium context)
    - CMS level N: rarely (slow context / long-term memory)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        enable_cms_multifreq: bool = True,
        enable_meta_learning: bool = False,
        meta_val_steps: int = 5,
    ):
        """
        Args:
            model: The model to train (e.g., HOPE)
            optimizer: The optimizer (e.g., DeepMomentumGD)
            device: Device to train on
            enable_cms_multifreq: Enable CMS multi-frequency updates
            enable_meta_learning: Enable meta-learning for optimizer
            meta_val_steps: Steps between meta-learning validation checks
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.enable_cms_multifreq = enable_cms_multifreq
        self.enable_meta_learning = enable_meta_learning
        self.meta_val_steps = meta_val_steps

        self.step_count = 0
        self.epoch_count = 0

        # Find CMS modules for multi-frequency updates
        self.cms_modules = self._find_cms_modules()
        if self.cms_modules:
            print(f"Found {len(self.cms_modules)} CMS modules for multi-frequency training")
            for name, cms in self.cms_modules.items():
                print(f"  {name}: {cms.num_levels} levels, chunk_sizes={cms.chunk_sizes}")

        # Track metrics
        self.metrics_history: List[Dict[str, float]] = []

    def _find_cms_modules(self) -> Dict[str, nn.Module]:
        """Find all ContinuumMemorySystem modules in the model."""
        from nested_learning.memory import ContinuumMemorySystem

        cms_modules = {}
        for name, module in self.model.named_modules():
            if isinstance(module, ContinuumMemorySystem):
                cms_modules[name] = module

        return cms_modules

    def _apply_cms_multifreq_update(self, step: int):
        """
        Apply multi-frequency gradient updates to all CMS modules.

        This is the key to the paper's continuum memory concept:
        - Level 0 updates frequently (captures fast-changing patterns)
        - Level N updates rarely (captures slow-changing / long-term patterns)
        """
        if not self.enable_cms_multifreq:
            return

        for name, cms in self.cms_modules.items():
            # Get which levels should update at this step
            update_levels = cms.get_update_levels(step)

            for level_idx, should_update in enumerate(update_levels):
                # Always accumulate gradients for all levels
                cms.accumulate_gradients(level_idx)

                # Apply accumulated gradients only when it's time for this level
                if should_update:
                    cms.apply_accumulated_gradients(level_idx)

    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Perform one training step with full nested learning.

        This implements the complete nested optimization:
        1. Forward pass through model
        2. Compute task loss
        3. Backward pass
        4. CMS multi-frequency gradient accumulation
        5. Optimizer step (includes memory module training via internal loss)

        Args:
            batch: Input batch
            labels: Target labels
            loss_fn: Loss function

        Returns:
            Dictionary of metrics for this step
        """
        self.model.train()
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        # Forward pass
        output = self.model(batch)

        # Handle different output formats
        if isinstance(output, tuple):
            # HOPE returns (loss, logits) when labels provided
            if len(output) == 2 and output[0].dim() == 0:
                loss = output[0]
            else:
                loss = loss_fn(output[0], labels)
        else:
            loss = loss_fn(output, labels)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Apply CMS multi-frequency updates BEFORE optimizer step
        # This ensures gradients are accumulated/applied per level
        self._apply_cms_multifreq_update(self.step_count)

        # Optimizer step - this includes memory module training
        # DeepMomentumGD will compute internal loss and update memory
        self.optimizer.step()

        # Apply self-modification updates after backward/optimizer step
        # This is necessary because self-modification defers updates to avoid
        # breaking gradient computation
        if hasattr(self.model, 'apply_pending_updates'):
            self.model.apply_pending_updates()

        self.step_count += 1

        # Collect metrics
        metrics = {
            'loss': loss.item(),
            'step': self.step_count,
        }

        # Add optimizer-specific metrics if available
        if hasattr(self.optimizer, 'get_internal_loss_stats'):
            metrics.update(self.optimizer.get_internal_loss_stats())

        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        val_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch with full nested learning.

        Args:
            dataloader: Training dataloader
            loss_fn: Loss function
            val_dataloader: Optional validation dataloader for meta-learning

        Returns:
            Dictionary of epoch metrics
        """
        self.epoch_count += 1
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch_count}")

        for batch_data in pbar:
            # Handle different dataloader formats
            if isinstance(batch_data, (list, tuple)):
                if len(batch_data) == 2:
                    batch, labels = batch_data
                else:
                    batch = batch_data[0]
                    labels = batch_data[0]  # Self-supervised
            else:
                batch = batch_data
                labels = batch_data

            metrics = self.train_step(batch, labels, loss_fn)
            total_loss += metrics['loss']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'avg': f"{total_loss / num_batches:.4f}",
            })

            # Meta-learning validation check
            if (self.enable_meta_learning and
                val_dataloader is not None and
                self.step_count % self.meta_val_steps == 0):
                self._meta_learning_step(val_dataloader, loss_fn)

        epoch_metrics = {
            'epoch': self.epoch_count,
            'train_loss': total_loss / num_batches,
            'total_steps': self.step_count,
        }

        self.metrics_history.append(epoch_metrics)

        return epoch_metrics

    def _meta_learning_step(
        self,
        val_dataloader: DataLoader,
        loss_fn: Callable,
    ):
        """
        Perform meta-learning update based on validation performance.

        This is an additional layer of nested optimization:
        - The memory modules are already trained via internal loss in each step
        - This provides an additional signal from validation performance
        """
        if not hasattr(self.optimizer, 'meta_step'):
            return

        # Compute validation loss
        self.model.eval()
        val_loss = 0.0
        num_val = 0

        with torch.enable_grad():  # Need gradients for meta-learning
            for batch_data in val_dataloader:
                if isinstance(batch_data, (list, tuple)):
                    batch, labels = batch_data[0], batch_data[-1]
                else:
                    batch = labels = batch_data

                batch = batch.to(self.device)
                labels = labels.to(self.device)

                output = self.model(batch)
                if isinstance(output, tuple):
                    loss = output[0] if output[0].dim() == 0 else loss_fn(output[0], labels)
                else:
                    loss = loss_fn(output, labels)

                val_loss = val_loss + loss
                num_val += 1

                if num_val >= 3:  # Limit validation batches for efficiency
                    break

        if num_val > 0:
            avg_val_loss = val_loss / num_val
            # Meta-step to improve optimizer based on validation
            self.optimizer.meta_step(avg_val_loss)

        self.model.train()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: Evaluation dataloader
            loss_fn: Loss function

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_data in dataloader:
            if isinstance(batch_data, (list, tuple)):
                batch, labels = batch_data[0], batch_data[-1]
            else:
                batch = labels = batch_data

            batch = batch.to(self.device)
            labels = labels.to(self.device)

            output = self.model(batch)
            if isinstance(output, tuple):
                loss = output[0] if output[0].dim() == 0 else loss_fn(output[0], labels)
                logits = output[1] if len(output) > 1 else output[0]
            else:
                loss = loss_fn(output, labels)
                logits = output

            total_loss += loss.item() * batch.size(0)
            total_samples += batch.size(0)

            # Compute accuracy for classification
            if logits.dim() > 1 and logits.size(-1) > 1:
                preds = logits.argmax(dim=-1)
                if labels.dim() == logits.dim():
                    labels_for_acc = labels.argmax(dim=-1)
                else:
                    labels_for_acc = labels
                total_correct += (preds == labels_for_acc).sum().item()

        metrics = {
            'eval_loss': total_loss / total_samples if total_samples > 0 else 0,
            'eval_samples': total_samples,
        }

        if total_correct > 0:
            metrics['eval_accuracy'] = total_correct / total_samples

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        log_every: int = 1,
        save_path: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        """
        Full training loop with nested learning.

        Args:
            train_loader: Training dataloader
            loss_fn: Loss function
            num_epochs: Number of epochs
            val_loader: Optional validation dataloader
            log_every: Log every N epochs
            save_path: Optional path to save best model

        Returns:
            Training history
        """
        print(f"\n{'='*60}")
        print("Nested Learning Training")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"CMS Multi-frequency: {self.enable_cms_multifreq}")
        print(f"Meta-learning: {self.enable_meta_learning}")
        print(f"Device: {self.device}")

        if self.cms_modules:
            print(f"\nCMS Update Schedule:")
            for name, cms in self.cms_modules.items():
                for i, chunk in enumerate(cms.chunk_sizes):
                    print(f"  {name} level {i}: every {chunk} steps")

        print(f"{'='*60}\n")

        best_val_loss = float('inf')
        history = []

        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(
                train_loader,
                loss_fn,
                val_dataloader=val_loader if self.enable_meta_learning else None,
            )

            # Evaluate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, loss_fn)
                train_metrics.update(val_metrics)

            # Log
            if (epoch + 1) % log_every == 0:
                log_str = f"Epoch {epoch + 1}/{num_epochs}"
                log_str += f" | Train Loss: {train_metrics['train_loss']:.4f}"
                if 'eval_loss' in val_metrics:
                    log_str += f" | Val Loss: {val_metrics['eval_loss']:.4f}"
                if 'eval_accuracy' in val_metrics:
                    log_str += f" | Val Acc: {val_metrics['eval_accuracy']:.4f}"
                print(log_str)

            # Save best model
            if save_path and 'eval_loss' in val_metrics:
                if val_metrics['eval_loss'] < best_val_loss:
                    best_val_loss = val_metrics['eval_loss']
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': train_metrics,
                    }, save_path)
                    print(f"  Saved best model (val_loss: {best_val_loss:.4f})")

            history.append(train_metrics)

        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"Final Train Loss: {history[-1]['train_loss']:.4f}")
        if 'eval_loss' in history[-1]:
            print(f"Final Val Loss: {history[-1]['eval_loss']:.4f}")
        print(f"{'='*60}\n")

        return history

    def get_cms_update_stats(self) -> Dict[str, Any]:
        """Get statistics about CMS update frequencies."""
        stats = {}
        for name, cms in self.cms_modules.items():
            stats[name] = {
                'num_levels': cms.num_levels,
                'chunk_sizes': cms.chunk_sizes,
                'step_count': cms.step_count.item(),
                'updates_per_level': [
                    self.step_count // chunk for chunk in cms.chunk_sizes
                ],
            }
        return stats


def create_nested_learning_setup(
    model: nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.9,
    memory_lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    enable_cms_multifreq: bool = True,
) -> Tuple[nn.Module, torch.optim.Optimizer, NestedLearningTrainer]:
    """
    Convenience function to create a complete nested learning setup.

    Args:
        model: Model to train
        lr: Learning rate
        momentum: Momentum coefficient
        memory_lr: Learning rate for memory modules
        device: Device to use
        enable_cms_multifreq: Enable CMS multi-frequency updates

    Returns:
        Tuple of (model, optimizer, trainer)
    """
    from nested_learning.optimizers import DeepMomentumGD

    model = model.to(device)

    optimizer = DeepMomentumGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        memory_lr=memory_lr,
        use_shared_memory=True,
    )

    trainer = NestedLearningTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        enable_cms_multifreq=enable_cms_multifreq,
    )

    return model, optimizer, trainer
