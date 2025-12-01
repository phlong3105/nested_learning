"""
HOPE: Hierarchical Optimization with Persistent Evolution

A transformer-like model combining:
1. Linear attention with external memory (fast weights)
2. Continuum Memory System (multi-frequency MLPs)

IMPORTANT LIMITATIONS:
- This does NOT implement true self-modification (parameters don't update online)
- The "self-modifying" name was misleading - renamed to LinearAttentionWithMemory
- CMS multi-frequency updates are NOT integrated into training loop
- This is a working transformer playground, not the full paper implementation

For actual self-modifying parameters, see titans.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math

from ..memory import ContinuumMemorySystem, LinearAttention
from .titans import SelfModifyingAttention as TrueSelfModifyingAttention


class HOPE(nn.Module):
    """
    HOPE: Hierarchical Optimization with Persistent Evolution

    A self-referential learning module that combines self-modifying
    sequence modeling with continuum memory system.

    Args:
        dim: Model dimension
        n_layers: Number of HOPE blocks
        n_heads: Number of attention heads
        vocab_size: Vocabulary size
        chunk_sizes: Update frequencies for CMS (default: [16, 64, 256])
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        vocab_size: int = 50257,
        chunk_sizes: Optional[List[int]] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # HOPE blocks
        self.blocks = nn.ModuleList([
            HOPEBlock(
                dim=dim,
                n_heads=n_heads,
                chunk_sizes=chunk_sizes,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output layers
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights between input and output embeddings
        self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.

        Args:
            input_ids: Token indices (batch, seq_len)
            labels: Target token indices for loss computation

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)  # (batch, seq_len, dim)

        # Positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.pos_embedding(positions)  # (seq_len, dim)

        # Combine embeddings
        x = tok_emb + pos_emb

        # Apply HOPE blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.head(x)  # (batch, seq_len, vocab_size)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if labels is not None:
            return loss, logits
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)

        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self(input_ids_cond)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class HOPEBlock(nn.Module):
    """
    Single HOPE block combining self-modification and continuum memory.

    Architecture:
        x → Self-Modifying Layer → Continuum Memory → x'

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        chunk_sizes: Update frequencies for CMS
        dropout: Dropout probability
        use_true_self_modification: If True, use SelfModifyingAttention from titans.py
            which actually modifies weights online. If False (default), use
            LinearAttentionWithMemory which only updates external memory buffer.
        self_mod_lr: Learning rate for self-modification (only if use_true_self_modification=True)
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        chunk_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_true_self_modification: bool = False,
        self_mod_lr: float = 0.001,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.use_true_self_modification = use_true_self_modification

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Attention layer - choose based on flag
        if use_true_self_modification:
            # Use the actual self-modifying attention from titans.py
            # This modifies W_q, W_k, W_v, W_o weights online
            self.attention = TrueSelfModifyingAttention(
                dim=dim,
                num_heads=n_heads,
                self_mod_lr=self_mod_lr,
                dropout=dropout,
            )
        else:
            # Use linear attention with external memory buffer
            # NOTE: This is NOT self-modifying - see docstring
            self.attention = LinearAttentionWithMemory(
                dim=dim,
                n_heads=n_heads,
                dropout=dropout,
            )

        # Continuum Memory System
        if chunk_sizes is None:
            chunk_sizes = [16, 64, 256]  # Default multi-frequency

        self.cms = ContinuumMemorySystem(
            dim=dim,
            hidden_dim=4 * dim,
            num_levels=len(chunk_sizes),
            chunk_sizes=chunk_sizes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, enable_self_modification: bool = True) -> torch.Tensor:
        """
        Forward pass through HOPE block.

        Args:
            x: Input tensor (batch, seq_len, dim)
            enable_self_modification: If True and using true self-modification,
                apply online weight updates during forward pass.
        """
        # Attention + residual
        if self.use_true_self_modification:
            x = x + self.attention(self.ln1(x), enable_self_modification=enable_self_modification)
        else:
            x = x + self.attention(self.ln1(x))

        # Continuum memory with residual
        x = x + self.cms(self.ln2(x))

        return x


class LinearAttentionWithMemory(nn.Module):
    """
    Linear attention with external memory buffer (Fast Weights).

    This implements the linear attention mechanism from the paper with
    an accumulating memory state M = M + v_t * k_t^T.

    NOTE: Despite being in a file about "self-modifying" models, this does NOT
    modify its own parameters (W_q, W_k, W_v, W_o). It updates an external
    memory buffer. True self-modification would require online parameter updates.

    For actual self-modifying parameters, see SelfModifyingAttention in titans.py.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        # Query, Key, Value projections (these can self-modify)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Memory state for linear attention
        self.register_buffer(
            'memory',
            torch.zeros(self.n_heads, self.head_dim, self.head_dim)
        )

    def reset_memory(self):
        """Reset memory state."""
        self.memory.zero_()

    def forward(self, x: torch.Tensor, reset_memory: bool = True) -> torch.Tensor:
        """
        Forward pass with linear attention and memory.

        Args:
            x: Input (batch, seq_len, dim)
            reset_memory: Whether to reset memory before processing each sequence.
                          Default True to prevent memory bleeding between sequences.

        Note: Memory accumulates within a sequence (fast weights), but resets
        between different forward passes by default. Set reset_memory=False
        for truly persistent memory across sequences (advanced use case).
        """
        # Reset memory at start of each sequence by default
        # This prevents memory from previous sequences bleeding into current one
        if reset_memory:
            self.reset_memory()

        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        # (batch, n_heads, seq_len, head_dim)

        # Use per-sequence memory instead of global buffer for proper batching
        # Initialize memory for this forward pass
        memory = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            device=x.device, dtype=x.dtype
        )

        # Linear attention with memory accumulation within sequence
        outputs = []
        for t in range(seq_len):
            k_t = K[:, :, t, :]  # (batch, n_heads, head_dim)
            v_t = V[:, :, t, :]  # (batch, n_heads, head_dim)
            q_t = Q[:, :, t, :]  # (batch, n_heads, head_dim)

            # Update memory: M = M + v * k^T
            # Shape: (batch, n_heads, head_dim, head_dim)
            memory_update = torch.einsum('bhd,bhk->bhdk', v_t, k_t)
            memory = memory + memory_update

            # Retrieve: y = M * q
            y_t = torch.einsum('bhdk,bhk->bhd', memory, q_t)
            outputs.append(y_t)

        # Stack and reshape
        output = torch.stack(outputs, dim=2)  # (batch, n_heads, seq_len, head_dim)
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
        output = output.view(batch_size, seq_len, self.dim)  # Combine heads

        # Output projection
        output = self.W_o(output)
        output = self.dropout(output)

        return output