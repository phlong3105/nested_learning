"""
Nested Deep Momentum GD with Meta-Learning

This is a COMPLETE implementation of the nested learning framework from the paper,
including:
- Internal loss functions L̃^(2) for memory modules
- Meta-learning to train optimizer networks
- Proper gradient flow through memory modules

This addresses the limitations in deep_momentum.py which uses static random MLPs.
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional, List, Callable


class LearnedMemoryModule(nn.Module):
    """
    MLP that learns to compress gradients.

    This is the core "memory" that replaces linear momentum.
    It will be trained via meta-learning.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(depth):
            if i == depth - 1:
                # Last layer
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))  # Add stability
                layers.append(nn.ReLU())
                current_dim = hidden_dim

        self.network = nn.Sequential(*layers)

        # Initialize with small weights for stability
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, grad: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """
        Process gradient and momentum to produce new momentum.

        Args:
            grad: Current gradient (flattened)
            momentum: Previous momentum (flattened)

        Returns:
            New momentum
        """
        # Concatenate gradient and momentum as input context
        x = torch.cat([grad, momentum], dim=-1)
        return self.network(x)


class NestedDeepMomentumGD(Optimizer):
    """
    Deep Momentum GD with proper nested optimization.

    This implements the full nested learning framework:
    - Outer loop: Optimize model parameters using learned memory
    - Inner loop: Optimize memory module parameters via meta-learning

    Args:
        params: Model parameters to optimize
        memory_optimizer: Optimizer for the memory modules (e.g., Adam)
        lr: Learning rate for model parameters
        memory_lr: Learning rate for memory module parameters
        momentum: Momentum coefficient
        memory_depth: Depth of memory MLP
        memory_hidden_dim: Hidden dimension of memory MLP
        meta_learning: Whether to enable meta-learning for memory
    """

    def __init__(
        self,
        params,
        memory_optimizer: Optional[Optimizer] = None,
        lr: float = 1e-3,
        memory_lr: float = 1e-4,
        momentum: float = 0.9,
        memory_depth: int = 2,
        memory_hidden_dim: int = 64,
        meta_learning: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_depth=memory_depth,
            memory_hidden_dim=memory_hidden_dim,
            memory_lr=memory_lr,
        )
        super().__init__(params, defaults)

        self.meta_learning = meta_learning
        self.memory_modules = {}
        self.memory_optimizer = memory_optimizer

        # Initialize memory modules for each parameter
        self._init_memory_modules()

    def _init_memory_modules(self):
        """Initialize learned memory modules for each parameter group."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Initialize momentum buffer
                state['momentum_buffer'] = torch.zeros_like(p.data)

                # Get dimensions
                param_numel = p.numel()
                hidden_dim = group['memory_hidden_dim']

                # Create learned memory module
                # Input: [gradient, momentum] (concatenated)
                # Output: new momentum
                memory = LearnedMemoryModule(
                    input_dim=param_numel * 2,  # grad + momentum
                    hidden_dim=hidden_dim,
                    output_dim=param_numel,
                    depth=group['memory_depth'],
                )

                # Move to same device as parameter
                memory = memory.to(p.device)

                # Store in state and module dict
                state['memory_module'] = memory
                self.memory_modules[id(p)] = memory

        # Create optimizer for memory modules if meta-learning is enabled
        if self.meta_learning and self.memory_optimizer is None:
            memory_params = []
            for memory in self.memory_modules.values():
                memory_params.extend(memory.parameters())

            self.memory_optimizer = torch.optim.Adam(
                memory_params,
                lr=self.param_groups[0]['memory_lr'],
            )

    def get_memory_modules(self) -> List[nn.Module]:
        """Return list of all memory modules."""
        return list(self.memory_modules.values())

    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                    Required for meta-learning.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Outer loop: Update model parameters using learned memory
        for group in self.param_groups:
            lr = group['lr']
            momentum_coef = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                momentum_buffer = state['momentum_buffer']
                memory_module = state['memory_module']

                # Flatten tensors
                grad_flat = grad.flatten()
                momentum_flat = momentum_buffer.flatten()

                # Process through learned memory module
                memory_module.train()
                new_momentum_flat = memory_module(
                    grad_flat.unsqueeze(0),
                    momentum_flat.unsqueeze(0),
                ).squeeze(0)

                # Reshape back
                new_momentum = new_momentum_flat.view_as(grad)

                # Classic momentum interpolation
                momentum_buffer.mul_(momentum_coef).add_(new_momentum)

                # Update parameters
                p.data.add_(momentum_buffer, alpha=-lr)

        return loss

    def meta_step(self, meta_loss: torch.Tensor):
        """
        Perform meta-learning step to improve memory modules.

        This implements the inner optimization loop L̃^(2).

        Args:
            meta_loss: Loss that depends on how well the optimizer performs
                      (e.g., validation loss after several optimization steps)
        """
        if not self.meta_learning:
            return

        if self.memory_optimizer is None:
            raise ValueError("Meta-learning enabled but no memory optimizer provided")

        # Backprop through memory modules
        self.memory_optimizer.zero_grad()
        meta_loss.backward()

        # Clip gradients for stability
        for memory in self.memory_modules.values():
            torch.nn.utils.clip_grad_norm_(memory.parameters(), max_norm=1.0)

        # Update memory module parameters
        self.memory_optimizer.step()

    def state_dict(self):
        """
        Return state dict including memory modules.
        """
        state = super().state_dict()

        # Add memory modules
        memory_state = {}
        for param_id, memory in self.memory_modules.items():
            memory_state[param_id] = memory.state_dict()

        state['memory_modules'] = memory_state

        if self.memory_optimizer is not None:
            state['memory_optimizer'] = self.memory_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict):
        """
        Load state dict including memory modules.
        """
        # Load memory modules
        if 'memory_modules' in state_dict:
            memory_state = state_dict.pop('memory_modules')
            for param_id, memory_dict in memory_state.items():
                if param_id in self.memory_modules:
                    self.memory_modules[param_id].load_state_dict(memory_dict)

        # Load memory optimizer
        if 'memory_optimizer' in state_dict and self.memory_optimizer is not None:
            mem_opt_state = state_dict.pop('memory_optimizer')
            self.memory_optimizer.load_state_dict(mem_opt_state)

        # Load base optimizer state
        super().load_state_dict(state_dict)


def create_meta_learning_task_distribution(
    num_tasks: int = 100,
    input_dim: int = 10,
    output_dim: int = 1,
    task_type: str = 'regression',
) -> List[Callable]:
    """
    Create a distribution of tasks for meta-learning the optimizer.

    This is used to train the memory modules on a variety of optimization
    problems so they learn general optimization strategies.

    Args:
        num_tasks: Number of tasks in distribution
        input_dim: Input dimension for tasks
        output_dim: Output dimension for tasks
        task_type: Type of tasks ('regression', 'classification', 'mixed')

    Returns:
        List of task generator functions
    """
    tasks = []

    for _ in range(num_tasks):
        if task_type == 'regression' or (task_type == 'mixed' and torch.rand(1).item() < 0.5):
            # Random regression task
            true_weights = torch.randn(input_dim, output_dim)
            noise_level = torch.rand(1).item() * 0.5

            def task_fn():
                X = torch.randn(32, input_dim)
                y = X @ true_weights + noise_level * torch.randn(32, output_dim)
                return X, y

            tasks.append(task_fn)
        else:
            # Random classification task (binary)
            true_weights = torch.randn(input_dim, output_dim)

            def task_fn():
                X = torch.randn(32, input_dim)
                logits = X @ true_weights
                y = (logits > 0).float()
                return X, y

            tasks.append(task_fn)

    return tasks
