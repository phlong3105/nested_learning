"""
Automatic Mixed Precision (AMP) utilities for nested learning.

Handles the complexity of mixed precision training with nested optimization:
1. Model forward pass can use FP16/BF16 for efficiency
2. Memory modules need careful handling for stable internal loss
3. Separate gradient scaling may be needed for memory optimizer

Usage:
    from nested_learning.utils.amp import NestedAMPWrapper, AMPConfig

    # Create wrapper
    amp_wrapper = NestedAMPWrapper(
        enabled=True,
        dtype=torch.bfloat16,  # or torch.float16
    )

    # Training loop
    with amp_wrapper.model_autocast():
        output = model(input_ids)
        loss = criterion(output, labels)

    amp_wrapper.backward(loss)
    amp_wrapper.unscale_and_clip(optimizer, max_norm=1.0)
    amp_wrapper.step(optimizer)
    amp_wrapper.update()

    # For memory optimizer (if using separate scaler)
    amp_wrapper.step_memory_optimizer(memory_optimizer)
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Union, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class AMPConfig:
    """Configuration for mixed precision training."""

    enabled: bool = True
    dtype: torch.dtype = torch.bfloat16
    use_separate_memory_scaler: bool = False
    init_scale: float = 65536.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000


class NestedAMPWrapper:
    """
    Handles mixed precision training for nested optimization.

    Key features:
    1. Autocast context for model forward pass
    2. Gradient scaling for FP16 stability (optional with BF16)
    3. Separate scaler option for memory optimizer
    4. Safe gradient unscaling and clipping

    Example:
        amp = NestedAMPWrapper(enabled=True)

        for batch in dataloader:
            with amp.model_autocast():
                loss, logits = model(batch)

            amp.backward(loss)
            amp.unscale_and_clip(optimizer)
            amp.step(optimizer)
            amp.update()
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        use_separate_memory_scaler: bool = False,
        init_scale: float = 65536.0,
        device_type: str = 'cuda',
    ):
        """
        Args:
            enabled: Whether to enable mixed precision
            dtype: Data type for autocast (bfloat16 or float16)
            use_separate_memory_scaler: Use separate scaler for memory optimizer
            init_scale: Initial scale factor for GradScaler
            device_type: Device type for autocast ('cuda' or 'cpu')
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        self.device_type = device_type
        self.use_separate_memory_scaler = use_separate_memory_scaler

        # BF16 doesn't need gradient scaling
        self.use_scaler = self.enabled and dtype == torch.float16

        if self.use_scaler:
            self.scaler = GradScaler(init_scale=init_scale)
            if use_separate_memory_scaler:
                self.memory_scaler = GradScaler(init_scale=init_scale)
            else:
                self.memory_scaler = None
        else:
            self.scaler = None
            self.memory_scaler = None

        # Track state
        self._gradients_unscaled = False

    @contextmanager
    def model_autocast(self):
        """
        Context manager for model forward pass with mixed precision.

        Usage:
            with amp_wrapper.model_autocast():
                output = model(input)
                loss = criterion(output, target)
        """
        if self.enabled:
            with autocast(device_type=self.device_type, dtype=self.dtype):
                yield
        else:
            yield

    @contextmanager
    def memory_autocast(self):
        """
        Context manager for memory module operations.

        By default, uses the same precision as model. Override if memory
        modules need different precision for stability.
        """
        if self.enabled:
            # Memory modules typically work fine with the same precision
            with autocast(device_type=self.device_type, dtype=self.dtype):
                yield
        else:
            yield

    @contextmanager
    def full_precision(self):
        """
        Context manager to force full precision operations.

        Useful for numerically sensitive operations.
        """
        if self.enabled:
            with autocast(device_type=self.device_type, enabled=False):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor, retain_graph: bool = False):
        """
        Backward pass with gradient scaling.

        Args:
            loss: The loss tensor to backpropagate
            retain_graph: Whether to retain computation graph
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            loss.backward(retain_graph=retain_graph)

        self._gradients_unscaled = False

    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """
        Unscale gradients before clipping or inspection.

        Call this before gradient clipping if using the scaler.
        """
        if self.scaler is not None and not self._gradients_unscaled:
            self.scaler.unscale_(optimizer)
            self._gradients_unscaled = True

    def unscale_and_clip(
        self,
        optimizer: torch.optim.Optimizer,
        max_norm: float = 1.0,
        parameters: Optional[torch.nn.Module] = None,
    ):
        """
        Unscale gradients and apply gradient clipping.

        Args:
            optimizer: The optimizer
            max_norm: Maximum gradient norm
            parameters: Parameters to clip (if None, uses optimizer params)
        """
        self.unscale_gradients(optimizer)

        if parameters is None:
            # Get parameters from optimizer
            params = []
            for group in optimizer.param_groups:
                params.extend(group['params'])
        else:
            params = list(parameters.parameters()) if hasattr(parameters, 'parameters') else list(parameters)

        torch.nn.utils.clip_grad_norm_(params, max_norm)

    def step(self, optimizer: torch.optim.Optimizer):
        """
        Optimizer step with gradient scaling handling.

        Args:
            optimizer: The optimizer to step
        """
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def step_memory_optimizer(self, memory_optimizer: torch.optim.Optimizer):
        """
        Step the memory optimizer, potentially with separate scaler.

        Args:
            memory_optimizer: The memory module optimizer
        """
        if self.memory_scaler is not None:
            self.memory_scaler.step(memory_optimizer)
        elif self.scaler is not None:
            # Use main scaler
            self.scaler.step(memory_optimizer)
        else:
            memory_optimizer.step()

    def update(self):
        """Update scaler(s) after optimizer step."""
        if self.scaler is not None:
            self.scaler.update()
        if self.memory_scaler is not None:
            self.memory_scaler.update()

        self._gradients_unscaled = False

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {'enabled': self.enabled}
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        if self.memory_scaler is not None:
            state['memory_scaler'] = self.memory_scaler.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if 'scaler' in state_dict and self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler'])
        if 'memory_scaler' in state_dict and self.memory_scaler is not None:
            self.memory_scaler.load_state_dict(state_dict['memory_scaler'])

    def get_scale(self) -> Optional[float]:
        """Get current gradient scale value."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return None


def create_amp_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    amp_wrapper: NestedAMPWrapper,
    max_grad_norm: float = 1.0,
):
    """
    Create a training step function with AMP support.

    Returns a function that handles:
    1. Mixed precision forward pass
    2. Scaled backward pass
    3. Gradient clipping
    4. Optimizer step
    5. Scale update

    Example:
        train_step = create_amp_training_step(model, optimizer, amp_wrapper)

        for batch in dataloader:
            loss = train_step(batch, labels, loss_fn)
    """
    def train_step(
        batch: torch.Tensor,
        labels: torch.Tensor,
        loss_fn,
    ) -> float:
        model.train()
        optimizer.zero_grad()

        with amp_wrapper.model_autocast():
            output = model(batch)
            if isinstance(output, tuple):
                loss = output[0] if output[0].dim() == 0 else loss_fn(output[0], labels)
            else:
                loss = loss_fn(output, labels)

        amp_wrapper.backward(loss)
        amp_wrapper.unscale_and_clip(optimizer, max_norm=max_grad_norm, parameters=model)
        amp_wrapper.step(optimizer)
        amp_wrapper.update()

        # Apply self-modification updates if present
        if hasattr(model, 'apply_pending_updates'):
            model.apply_pending_updates()

        return loss.item()

    return train_step


class AMPTrainer:
    """
    High-level trainer with AMP support for nested learning.

    Combines NestedAMPWrapper with training utilities for easy use.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory_optimizer: Optional[torch.optim.Optimizer] = None,
        amp_config: Optional[AMPConfig] = None,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
    ):
        """
        Args:
            model: The model to train
            optimizer: Main optimizer
            memory_optimizer: Optional memory module optimizer
            amp_config: AMP configuration
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.memory_optimizer = memory_optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        if amp_config is None:
            amp_config = AMPConfig()

        self.amp = NestedAMPWrapper(
            enabled=amp_config.enabled,
            dtype=amp_config.dtype,
            use_separate_memory_scaler=amp_config.use_separate_memory_scaler,
            init_scale=amp_config.init_scale,
        )

    def train_step(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        loss_fn,
    ) -> Dict[str, float]:
        """
        Perform one training step with AMP.

        Returns:
            Dictionary with loss and other metrics
        """
        self.model.train()
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()
        if self.memory_optimizer is not None:
            self.memory_optimizer.zero_grad()

        # Forward pass with autocast
        with self.amp.model_autocast():
            output = self.model(batch)
            if isinstance(output, tuple):
                loss = output[0] if output[0].dim() == 0 else loss_fn(output[0], labels)
            else:
                loss = loss_fn(output, labels)

        # Backward
        self.amp.backward(loss)

        # Gradient clipping
        self.amp.unscale_and_clip(
            self.optimizer,
            max_norm=self.max_grad_norm,
            parameters=self.model,
        )

        # Optimizer steps
        self.amp.step(self.optimizer)
        if self.memory_optimizer is not None:
            self.amp.step_memory_optimizer(self.memory_optimizer)
        self.amp.update()

        # Self-modification updates
        if hasattr(self.model, 'apply_pending_updates'):
            self.model.apply_pending_updates()

        metrics = {'loss': loss.item()}

        if self.amp.get_scale() is not None:
            metrics['grad_scale'] = self.amp.get_scale()

        return metrics

    def state_dict(self) -> Dict[str, Any]:
        """Get full state dict for checkpointing."""
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': self.amp.state_dict(),
        }
        if self.memory_optimizer is not None:
            state['memory_optimizer'] = self.memory_optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load from checkpoint."""
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if 'amp' in state_dict:
            self.amp.load_state_dict(state_dict['amp'])
        if 'memory_optimizer' in state_dict and self.memory_optimizer is not None:
            self.memory_optimizer.load_state_dict(state_dict['memory_optimizer'])
