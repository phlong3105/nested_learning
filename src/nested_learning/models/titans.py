"""
Self-Modifying Titans: Sequence models that learn their own update rules.

Based on Section 2.3, Equation 28-29:
    W_{t+1} = W_t * (I - x_t * x_t^T) - lr * grad_L

This module implements parameter self-modification during the forward pass.

IMPORTANT: Self-modification is deferred to avoid breaking gradient computation.
The update is stored and applied after backward() completes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class SelfModifyingLinear(nn.Module):
    """
    Linear layer that modifies its own weights online using delta rule.

    Based on Equations 28-29:
        W_{t+1} = W_t (I - x_t x_t^T) - η ∇L_t

    Two modes available:
    - normalized=True (default): W -= lr * (W @ x @ x^T) / (x^T @ x)
      This is a normalized projection that's more stable but deviates from paper.
    - normalized=False: W -= lr * (W @ x @ x^T)
      This matches the paper exactly but may be less stable.

    The self-modification is stored and applied after the backward pass
    to avoid breaking gradient computation. Call apply_pending_updates()
    after optimizer.step() to apply the accumulated updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        self_mod_lr: float = 0.01,
        use_bias: bool = True,
        immediate_update: bool = False,  # If True, update immediately (only for inference)
        normalized: bool = True,  # If True, normalize by x^T @ x (more stable but deviates from paper)
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.self_mod_lr = self_mod_lr
        self.immediate_update = immediate_update
        self.normalized = normalized

        # Main weight matrix (this will be modified online)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Running statistics for stability
        self.register_buffer('update_count', torch.zeros(1))

        # Pending updates (accumulated, applied after backward)
        self._pending_updates: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor, update_weights: bool = True) -> torch.Tensor:
        """
        Forward pass with optional online weight modification.

        Args:
            x: Input tensor (batch, seq, in_features) or (batch, in_features)
            update_weights: Whether to compute delta-rule weight updates.
                           Works during both training AND inference for online adaptation.

        Returns:
            Output tensor
        """
        # Standard linear transform
        output = torch.nn.functional.linear(x, self.weight, self.bias)

        # Compute weight modification using delta rule
        # NOTE: Self-modification works during both training and inference
        # for online adaptation (per paper's intent for deployment-time learning)
        if update_weights:
            with torch.no_grad():
                # Flatten to (batch * seq, features) if needed
                if x.dim() == 3:
                    batch, seq, feat = x.shape
                    x_flat = x.reshape(-1, feat)
                else:
                    x_flat = x

                # Delta rule: W_{t+1} = W_t (I - x x^T) = W - W @ x @ x^T
                # Average over batch for stability
                x_avg = x_flat.mean(dim=0)  # (in_features,)

                # Compute outer product component: x^T @ x (scalar)
                x_norm_sq = (x_avg ** 2).sum()

                if x_norm_sq > 1e-8:  # Avoid division by zero
                    # W @ x @ x^T = outer(W @ x, x)
                    Wx = self.weight.data @ x_avg  # (out_features,)
                    outer_product = torch.outer(Wx, x_avg)  # (out_features, in_features)

                    if self.normalized:
                        # Normalized version (more stable, but deviates from paper):
                        # W -= lr * (W @ x @ x^T) / (x^T @ x)
                        # This makes the update a projection with bounded magnitude
                        update = outer_product / (x_norm_sq + 1e-8) * self.self_mod_lr
                    else:
                        # Paper-exact version (Equation 28-29):
                        # W -= lr * (W @ x @ x^T)
                        # This is the true (I - xx^T) operation scaled by lr
                        update = outer_product * self.self_mod_lr

                    if self.immediate_update:
                        # Apply immediately (for inference only)
                        self.weight.data.sub_(update)
                        self.weight.data.clamp_(-10.0, 10.0)
                    else:
                        # Store for later application
                        self._pending_updates.append(update.clone())

                    self.update_count += 1

        return output

    def apply_pending_updates(self):
        """
        Apply all pending self-modification updates.

        Call this AFTER optimizer.step() to apply the accumulated
        delta-rule updates without breaking gradient computation.
        """
        if not self._pending_updates:
            return

        with torch.no_grad():
            for update in self._pending_updates:
                self.weight.data.sub_(update)

            # Clip weights for stability
            self.weight.data.clamp_(-10.0, 10.0)

        self._pending_updates.clear()

    def clear_pending_updates(self):
        """Clear pending updates without applying them."""
        self._pending_updates.clear()


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

    def apply_pending_updates(self):
        """Apply pending self-modification updates."""
        self.input_proj.apply_pending_updates()
        self.output_proj.apply_pending_updates()

    def get_update_stats(self) -> dict:
        """Get statistics about self-modification updates."""
        return {
            'input_proj_updates': self.input_proj.update_count.item(),
            'output_proj_updates': self.output_proj.update_count.item(),
        }


class SelfModifyingAttention(nn.Module):
    """
    Attention layer with actual self-modifying parameters.

    This correctly implements online weight modification without
    breaking gradient computation by deferring updates.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        self_mod_lr: Learning rate for self-modification
        dropout: Dropout probability
        normalized: If True (default), use normalized self-modification (more stable).
                   If False, use paper-exact formulation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        self_mod_lr: float = 0.001,
        dropout: float = 0.1,
        normalized: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        # Query, Key, Value projections with self-modification
        self.W_q = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)
        self.W_k = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)
        self.W_v = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)

        # Output projection with self-modification
        self.W_o = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)

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
            enable_self_modification: Whether to compute weight updates

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

    def apply_pending_updates(self):
        """Apply pending self-modification updates."""
        self.W_q.apply_pending_updates()
        self.W_k.apply_pending_updates()
        self.W_v.apply_pending_updates()
        self.W_o.apply_pending_updates()


class L2RegressionAttention(nn.Module):
    """
    L2 Regression Attention with self-modifying weights.

    Implements the paper's L2 regression variant (Equations 27-29):
        W_{t+1} = W_t (I - x_t x_t^T) - η ∇_W L

    Unlike standard attention (dot-product similarity), this uses L2 regression
    for memory updates, which provides better capacity management.

    The memory update minimizes ||M(k) - v||^2 via gradient descent:
        M_{t+1} = M_t + η * (v_t - M_t @ k_t) @ k_t^T

    This is the delta-rule update for associative memory.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        memory_lr: Learning rate for memory updates
        self_mod_lr: Learning rate for self-modifying projections
        dropout: Dropout probability
        normalized: If True, use normalized self-modification (more stable)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        memory_lr: float = 0.1,
        self_mod_lr: float = 0.001,
        dropout: float = 0.1,
        normalized: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.memory_lr = memory_lr
        assert dim % num_heads == 0

        # Self-modifying projections (Eq 28-29)
        self.W_q = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)
        self.W_k = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)
        self.W_v = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)
        self.W_o = SelfModifyingLinear(dim, dim, self_mod_lr=self_mod_lr, use_bias=False, normalized=normalized)

        self.dropout = nn.Dropout(dropout)

        # Per-head memory matrices (fast weights)
        # Shape: (num_heads, head_dim, head_dim) - shared across batch
        self.register_buffer(
            'memory',
            torch.zeros(num_heads, self.head_dim, self.head_dim)
        )

    def reset_memory(self):
        """Reset the memory state."""
        self.memory.zero_()

    def forward(
        self,
        x: torch.Tensor,
        enable_self_modification: bool = True,
        reset_memory: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with L2 regression memory and self-modifying weights.

        Args:
            x: Input (batch, seq_len, dim)
            enable_self_modification: Whether to update projection weights
            reset_memory: Whether to reset memory before processing

        Returns:
            Output (batch, seq_len, dim)
        """
        if reset_memory:
            self.reset_memory()

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V with self-modification
        Q = self.W_q(x, update_weights=enable_self_modification)
        K = self.W_k(x, update_weights=enable_self_modification)
        V = self.W_v(x, update_weights=enable_self_modification)

        # Reshape for multi-head: (batch, seq, heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Process sequentially with L2 regression memory updates
        outputs = []

        for t in range(seq_len):
            q_t = Q[:, t]  # (batch, heads, head_dim)
            k_t = K[:, t]  # (batch, heads, head_dim)
            v_t = V[:, t]  # (batch, heads, head_dim)

            # L2 regression memory update (delta rule):
            # M_{t+1} = M_t + η * (v - M @ k) @ k^T
            # This minimizes ||M @ k - v||^2

            # Predict using current memory
            # memory: (heads, head_dim, head_dim)
            # k_t: (batch, heads, head_dim)
            # v_pred: (batch, heads, head_dim)
            v_pred = torch.einsum('hdk,bhk->bhd', self.memory, k_t)

            # Compute error
            error = v_t - v_pred  # (batch, heads, head_dim)

            # Update memory with delta rule (average across batch)
            # update: (heads, head_dim, head_dim)
            with torch.no_grad():
                update = torch.einsum('bhd,bhk->hdk', error, k_t) / batch_size
                self.memory = self.memory + self.memory_lr * update

            # Retrieve using query
            # output_t: (batch, heads, head_dim)
            output_t = torch.einsum('hdk,bhk->bhd', self.memory, q_t)
            outputs.append(output_t)

        # Stack outputs: (batch, seq, heads, head_dim)
        output = torch.stack(outputs, dim=1)

        # Concatenate heads: (batch, seq, dim)
        output = output.contiguous().view(batch_size, seq_len, self.dim)

        # Output projection with self-modification
        output = self.W_o(output, update_weights=enable_self_modification)
        output = self.dropout(output)

        return output

    def apply_pending_updates(self):
        """Apply pending self-modification updates."""
        self.W_q.apply_pending_updates()
        self.W_k.apply_pending_updates()
        self.W_v.apply_pending_updates()
        self.W_o.apply_pending_updates()
