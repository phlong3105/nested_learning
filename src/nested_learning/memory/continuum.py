"""
Continuum Memory System (CMS)

Generalizes the traditional long-term/short-term memory viewpoint.
Multiple MLP layers with different update frequencies, each compressing
information at different timescales.

Based on Section 3, Equations 30-31:
    y_t = MLP^(f_k)(...MLP^(f_1)(x_t))
    θ^(f_ℓ)_{i+1} = θ^(f_ℓ)_i - sum(...) if i ≡ 0 (mod C^(ℓ))
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ContinuumMemorySystem(nn.Module):
    """
    Continuum Memory System with multi-frequency updates.

    Each MLP layer updates at different frequencies, allowing the model
    to compress information at multiple timescales.

    Args:
        dim: Model dimension
        hidden_dim: Hidden dimension for MLPs
        num_levels: Number of frequency levels
        chunk_sizes: List of chunk sizes for each level (update frequency)
        activation: Activation function (default: 'relu')
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_levels: int = 3,
        chunk_sizes: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.num_levels = num_levels

        # Default chunk sizes: exponentially increasing
        if chunk_sizes is None:
            chunk_sizes = [2 ** (i + 3) for i in range(num_levels)]  # [8, 16, 32, ...]
        assert len(chunk_sizes) == num_levels

        self.chunk_sizes = chunk_sizes

        # Create MLP for each frequency level
        self.mlps = nn.ModuleList()
        for i in range(num_levels):
            mlp = MLP(
                dim=dim,
                hidden_dim=self.hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            self.mlps.append(mlp)

        # Gradient accumulators for each level
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))
        self.grad_accumulators = [None] * num_levels

    def get_update_levels(self, step: int) -> List[bool]:
        """
        Determine which levels should be updated at current step.

        Returns list of booleans indicating which levels to update.
        """
        return [step % chunk_size == 0 for chunk_size in self.chunk_sizes]

    def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
        """
        Forward pass through all MLP levels.

        Args:
            x: Input tensor (batch, seq_len, dim)
            update: Whether to perform parameter updates (training mode)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        # Pass through all MLPs sequentially (nested structure)
        h = x
        for i, mlp in enumerate(self.mlps):
            h = mlp(h)

        # During training, accumulate gradients for multi-frequency update
        if update and self.training:
            self.step_count += 1

        return h

    def accumulate_gradients(self, level: int):
        """
        Accumulate gradients for a specific level.

        This should be called after backward pass but before optimizer step.
        """
        mlp = self.mlps[level]
        if self.grad_accumulators[level] is None:
            # Initialize accumulator
            self.grad_accumulators[level] = [
                p.grad.clone() if p.grad is not None else None
                for p in mlp.parameters()
            ]
        else:
            # Add to accumulator
            for i, p in enumerate(mlp.parameters()):
                if p.grad is not None:
                    if self.grad_accumulators[level][i] is None:
                        self.grad_accumulators[level][i] = p.grad.clone()
                    else:
                        self.grad_accumulators[level][i] += p.grad

    def apply_accumulated_gradients(self, level: int):
        """
        Apply accumulated gradients to a specific level.

        This should be called when it's time to update that level.
        """
        if self.grad_accumulators[level] is None:
            return

        mlp = self.mlps[level]
        for i, p in enumerate(mlp.parameters()):
            if self.grad_accumulators[level][i] is not None:
                p.grad = self.grad_accumulators[level][i] / self.chunk_sizes[level]

        # Clear accumulator
        self.grad_accumulators[level] = None


class MLP(nn.Module):
    """
    Simple MLP module used in CMS.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden layer dimension
        activation: Activation function name
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish' or activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        h = self.fc1(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class FullyNestedCMS(nn.Module):
    """
    Fully nested variant of CMS where each level wraps the previous.

    This is an alternative formulation mentioned in Appendix B.1:
        y_t = MLP^(f_k)(MLP^(f_{k-1})(...MLP^(f_1)(x_t)))

    Each outer MLP processes the output of inner MLPs.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        num_levels: int = 3,
        chunk_sizes: Optional[List[int]] = None,
        activation: str = 'relu',
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim
        self.num_levels = num_levels

        if chunk_sizes is None:
            chunk_sizes = [2 ** (i + 3) for i in range(num_levels)]
        self.chunk_sizes = chunk_sizes

        # Create nested MLPs
        # Each level takes input from previous level
        self.mlps = nn.ModuleList()
        for i in range(num_levels):
            mlp = MLP(
                dim=dim,  # All operate on same dimension
                hidden_dim=self.hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            self.mlps.append(mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through nested MLPs.

        Each MLP processes the output of the previous one.
        """
        h = x
        for mlp in self.mlps:
            h = h + mlp(h)  # Residual connection
        return h