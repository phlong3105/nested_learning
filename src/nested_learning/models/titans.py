"""
Self-Modifying Titans: Sequence models that learn their own update rules.

Based on Section 2.3, Equation 28-29:
    W_{t+1} = W_t * (I - x_t * x_t^T) - lr * grad_L

This module actually implements parameter self-modification during the forward pass.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SelfModifyingLinear(nn.Module):
    """
    Linear layer that modifies its own weights online using delta rule.

    Based on Equations 28-29:
        W_{t+1} = W_t (I - x_t x_t^T) - η ∇L_t
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        self_mod_lr: float = 0.01,
        use_bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.self_mod_lr = self_mod_lr

        # Main weight matrix (this will be modified online)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Running statistics for stability
        self.register_buffer('update_count', torch.zeros(1))

    def forward(self, x: torch.Tensor, update_weights: bool = True) -> torch.Tensor:
        """
        Forward pass with optional online weight modification.

        Args:
            x: Input tensor (batch, seq, in_features) or (batch, in_features)
            update_weights: Whether to apply delta-rule weight updates

        Returns:
            Output tensor
        """
        # Standard linear transform
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # Online weight modification using delta rule
        if update_weights and self.training:
            with torch.no_grad():
                # Flatten to (batch * seq, features) if needed
                if x.dim() == 3:
                    batch, seq, feat = x.shape
                    x_flat = x.reshape(-1, feat)
                else:
                    x_flat = x

                # Delta rule: W -= lr * x * x^T * W
                # This is the (I - xx^T) term from the paper
                # Simplified for efficiency: update = -lr * outer(x_avg, (x_avg @ W))

                # Average over batch for stability
                x_avg = x_flat.mean(dim=0)  # (in_features,)

                # Compute outer product component: x^T x (scalar normalization)
                x_norm_sq = (x_avg ** 2).sum()

                # Delta rule update (simplified)
                # W_new = W - lr * (W @ x) @ x^T / batch_size
                if x_norm_sq > 1e-6:  # Avoid division by zero
                    grad_approx = torch.outer(
                        self.weight @ x_avg,  # (out_features,)
                        x_avg,  # (in_features,)
                    ) / (x_norm_sq + 1e-6)

                    # Apply update with small learning rate
                    self.weight.sub_(grad_approx * self.self_mod_lr)

                    # Clip weights for stability
                    self.weight.clamp_(-10.0, 10.0)

                    self.update_count += 1

        return output


class SelfModifyingTitan(nn.Module):
    """
    A sequence model that learns to update its own parameters online.

    This implements the full self-modifying architecture:
    - RNN/GRU backbone for sequence processing
    - Self-modifying linear layers that update via delta rule
    - Online parameter modification during forward pass
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
        self_mod_lr: float = 0.001,
        dropout: float = 0.1,
        use_gru: bool = True,
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of recurrent layers
            self_mod_lr: Learning rate for online weight modification
            dropout: Dropout probability
            use_gru: Use GRU if True, LSTM if False
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Input projection with self-modification
        self.input_proj = SelfModifyingLinear(
            input_dim,
            hidden_dim,
            self_mod_lr=self_mod_lr,
        )

        # Recurrent backbone (standard - not self-modifying for stability)
        if use_gru:
            self.rnn = nn.GRU(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )

        # Output projection with self-modification
        self.output_proj = SelfModifyingLinear(
            hidden_dim,
            output_dim,
            self_mod_lr=self_mod_lr,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        enable_self_modification: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with online parameter modification.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            hidden: Initial hidden state (optional)
            enable_self_modification: Whether to apply delta-rule updates

        Returns:
            (output, hidden): Output tensor and final hidden state
        """
        batch_size, seq_len, _ = x.shape

        # Project input with self-modification
        h = self.input_proj(x, update_weights=enable_self_modification)
        h = torch.relu(h)
        h = self.dropout(h)

        # Recurrent processing
        h, hidden = self.rnn(h, hidden)

        # Layer norm for stability
        h = self.layer_norm(h)

        # Project output with self-modification
        output = self.output_proj(h, update_weights=enable_self_modification)

        return output, hidden

    def get_update_stats(self) -> dict:
        """Get statistics about self-modification updates."""
        return {
            'input_proj_updates': self.input_proj.update_count.item(),
            'output_proj_updates': self.output_proj.update_count.item(),
        }


class SelfModifyingAttention(nn.Module):
    """
    Attention layer with actual self-modifying parameters.

    This is a corrected version that actually modifies its weights online,
    not just maintains an external memory buffer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        self_mod_lr: float = 0.001,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Query, Key, Value projections with self-modification
        self.W_q = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False)
        self.W_k = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False)
        self.W_v = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False)

        # Output projection with self-modification
        self.W_o = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        enable_self_modification: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with self-modifying attention weights.

        Args:
            x: Input (batch, seq_len, dim)
            enable_self_modification: Whether to update weights online

        Returns:
            Output (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape

        # Project to Q, K, V with online weight updates
        Q = self.W_q(x, update_weights=enable_self_modification)
        K = self.W_k(x, update_weights=enable_self_modification)
        V = self.W_v(x, update_weights=enable_self_modification)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        output = torch.matmul(attn, V)

        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)

        # Output projection with self-modification
        output = self.W_o(output, update_weights=enable_self_modification)

        return output
