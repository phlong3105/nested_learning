"""
Deep Momentum Gradient Descent (DMGD)

Extends standard momentum with a deep memory module (MLP) that learns
to compress gradients more effectively.

Based on Section 2.3 of the paper, Equation 23:
    W_{i+1} = W_i + m_{i+1}(u_i)
    m_{i+1} = epsilon_{i+1} * m_i - eta_t * grad_L^(2)(m_i; u_i, I)

where u_i = grad_L(W_i; x_i) and m(·) is an MLP instead of linear matrix.

IMPORTANT LIMITATIONS:
- TODO: Memory MLPs are randomly initialized and NEVER TRAINED (no internal loss L^(2))
- TODO: Only feeds gradient, not [gradient, momentum] context as paper suggests
- TODO: Per-parameter MLPs create scalability issues for large models
- TODO: Needs meta-learning infrastructure to actually learn optimization

Current implementation is a STATIC NONLINEAR MOMENTUM, not the learned optimizer
described in the paper. This demonstrates the API design but not the full concept.
"""

import torch
from torch.optim import Optimizer
import torch.nn as nn


class DeepMomentumGD(Optimizer):
    """
    Deep Momentum Gradient Descent optimizer.

    This optimizer treats momentum as a neural network (MLP) rather than
    a simple matrix, enabling it to learn more complex functions of past
    gradients.

    IMPORTANT - THIS IS NOT THE PAPER'S VERSION:
    - Memory MLPs are randomly initialized and NEVER TRAINED
    - No internal loss L^(2) is computed or optimized
    - This is "nonlinear momentum", not "learned optimization"
    - Paper's version requires meta-learning infrastructure we don't have

    To implement the paper's actual approach, you would need:
    1. Define internal loss for memory module (Eq. 23)
    2. Alternate between task optimization and memory optimization
    3. Or use meta-learning to pretrain memory on task distribution

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        momentum: Momentum coefficient (default: 0.9)
        memory_depth: Depth of the momentum memory module (default: 2)
        memory_hidden_dim: Hidden dimension of momentum MLP (default: None, uses param dim)
        weight_decay: Weight decay coefficient (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        memory_depth=2,
        memory_hidden_dim=None,
        weight_decay=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if memory_depth < 1:
            raise ValueError(f"Invalid memory depth: {memory_depth}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            memory_depth=memory_depth,
            memory_hidden_dim=memory_hidden_dim,
            weight_decay=weight_decay,
        )
        super(DeepMomentumGD, self).__init__(params, defaults)

        # Initialize deep momentum memories for each parameter group
        self._init_momentum_memories()

    def _init_momentum_memories(self):
        """Initialize MLP-based momentum memory for each parameter."""
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                # Initialize momentum buffer
                state['momentum_buffer'] = torch.zeros_like(p.data)

                # Get dimensions
                param_numel = p.numel()
                hidden_dim = group['memory_hidden_dim']
                if hidden_dim is None:
                    hidden_dim = min(param_numel, 1024)  # Cap for efficiency

                # Create MLP memory module: input is flattened gradient
                layers = []
                in_dim = param_numel

                for i in range(group['memory_depth']):
                    if i == group['memory_depth'] - 1:
                        # Last layer outputs same size as input
                        layers.append(nn.Linear(hidden_dim if i > 0 else in_dim, param_numel))
                    else:
                        # Hidden layers
                        out_dim = hidden_dim
                        layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out_dim))
                        layers.append(nn.ReLU())

                state['memory_module'] = nn.Sequential(*layers)

                # Move memory module to same device as parameter
                state['memory_module'] = state['memory_module'].to(p.device)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Add weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                momentum_buffer = state['momentum_buffer']
                memory_module = state['memory_module']

                # Flatten gradient for MLP processing
                grad_flat = grad.flatten()

                # Process gradient through deep memory module
                # This implements: m_{i+1} = f(grad) where f is learned MLP
                with torch.enable_grad():
                    # Enable grad for the memory module update
                    memory_module.train()
                    processed_grad = memory_module(grad_flat)

                # Reshape back to parameter shape
                processed_grad = processed_grad.view_as(grad)

                # Update momentum buffer with processed gradient
                # m_{i+1} = momentum * m_i + processed_grad
                momentum_buffer.mul_(momentum).add_(processed_grad)

                # Update parameters: W_{i+1} = W_i - lr * m_{i+1}
                p.add_(momentum_buffer, alpha=-lr)

        return loss


class SimpleMomentumGD(Optimizer):
    """
    Standard momentum gradient descent for comparison.

    This is the baseline optimizer without deep memory.
    """

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SimpleMomentumGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                p.add_(buf, alpha=-group['lr'])

        return loss