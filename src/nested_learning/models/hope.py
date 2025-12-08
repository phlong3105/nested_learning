"""
HOPE: Hierarchical Optimization with Persistent Evolution

A transformer-like model implementing the full nested learning framework:
1. Self-modifying attention (weights update online via delta rule)
2. Continuum Memory System (multi-frequency MLPs for temporal hierarchy)
3. Integration with nested optimizers for end-to-end nested learning

The key insight from the paper: different timescales require different
treatment. Fast-changing patterns are handled by self-modifying attention,
while slower patterns are captured by CMS at different frequencies.

Architecture per block:
    x -> Self-Modifying Attention -> CMS -> x'

Where:
- Self-modifying attention: Weights change during forward pass (fast adaptation)
- CMS: Multi-frequency MLP stack (temporal memory hierarchy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from contextlib import contextmanager
import copy
import math

from ..memory import ContinuumMemorySystem, LinearAttention
from .titans import SelfModifyingAttention, SelfModifyingLinear


class HOPE(nn.Module):
    """
    HOPE: Hierarchical Optimization with Persistent Evolution

    A self-referential transformer that combines:
    1. Self-modifying attention (parameters change online)
    2. Continuum Memory System (multi-frequency temporal hierarchy)

    Designed to work with NestedLearningTrainer for full nested optimization:
    - Model parameters updated via learned optimizer (DeepMomentumGD)
    - CMS levels updated at different frequencies
    - Attention weights self-modify during forward pass

    Args:
        dim: Model dimension
        n_layers: Number of HOPE blocks
        n_heads: Number of attention heads
        vocab_size: Vocabulary size for language modeling
        chunk_sizes: Update frequencies for CMS levels (default: [8, 32, 128])
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        use_self_modification: Enable true self-modifying attention (default: True)
        self_mod_lr: Learning rate for self-modification
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
        use_self_modification: bool = True,  # Default to TRUE for nested learning
        self_mod_lr: float = 0.001,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.use_self_modification = use_self_modification

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # Positional embedding
        self.pos_embedding = nn.Embedding(max_seq_len, dim)

        # HOPE blocks
        if chunk_sizes is None:
            chunk_sizes = [8, 32, 128]  # Fast, medium, slow memory

        self.blocks = nn.ModuleList([
            HOPEBlock(
                dim=dim,
                n_heads=n_heads,
                chunk_sizes=chunk_sizes,
                dropout=dropout,
                use_self_modification=use_self_modification,
                self_mod_lr=self_mod_lr,
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
        enable_self_modification: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            input_ids: Token indices (batch, seq_len)
            labels: Target token indices for loss computation
            enable_self_modification: Whether to allow online weight updates

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)

        # Positional embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.pos_embedding(positions)

        # Combine embeddings
        x = tok_emb + pos_emb

        # Apply HOPE blocks with self-modification
        for block in self.blocks:
            x = block(x, enable_self_modification=enable_self_modification)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.head(x)

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
        enable_self_modification: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Note: Self-modification is active during generation, allowing
        the model to adapt its weights to the generated context.
        This works in both training and eval modes.

        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            enable_self_modification: Whether to allow online weight updates

        Returns:
            Generated token indices (batch, seq_len + max_new_tokens)
        """
        # No need to switch to training mode - self-modification now works
        # during inference for online adaptation (per paper's intent)

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass with self-modification
            logits = self(input_ids_cond, enable_self_modification=enable_self_modification)

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

    def get_cms_modules(self) -> List[ContinuumMemorySystem]:
        """Get all CMS modules for multi-frequency training."""
        cms_modules = []
        for block in self.blocks:
            cms_modules.append(block.cms)
        return cms_modules

    def get_self_mod_stats(self) -> dict:
        """Get statistics about self-modification updates."""
        stats = {'blocks': []}
        for i, block in enumerate(self.blocks):
            block_stats = block.get_self_mod_stats()
            block_stats['block_idx'] = i
            stats['blocks'].append(block_stats)
        return stats

    def apply_pending_updates(self):
        """
        Apply pending self-modification updates to all blocks.

        Call this AFTER optimizer.step() to apply the accumulated
        delta-rule updates without breaking gradient computation.
        """
        for block in self.blocks:
            block.apply_pending_updates()

    def clear_pending_updates(self):
        """Clear all pending updates without applying them."""
        for block in self.blocks:
            block.clear_pending_updates()

    def get_self_mod_state(self) -> Dict[str, torch.Tensor]:
        """
        Snapshot the current state of all self-modifying layers.

        Returns a dict of weight tensors keyed by their path.
        Used by adaptation_scope() for save/restore.
        """
        state = {}
        for name, module in self.named_modules():
            if isinstance(module, SelfModifyingLinear):
                state[f"{name}.weight"] = module.weight.data.clone()
                if module.bias is not None:
                    state[f"{name}.bias"] = module.bias.data.clone()
        return state

    def set_self_mod_state(self, state: Dict[str, torch.Tensor]):
        """
        Restore self-modifying layers to a previous state.

        Args:
            state: Dict from get_self_mod_state()
        """
        for name, module in self.named_modules():
            if isinstance(module, SelfModifyingLinear):
                weight_key = f"{name}.weight"
                if weight_key in state:
                    module.weight.data.copy_(state[weight_key])
                bias_key = f"{name}.bias"
                if bias_key in state and module.bias is not None:
                    module.bias.data.copy_(state[bias_key])

    @contextmanager
    def adaptation_scope(self, enable_self_modification: bool = True):
        """
        Context manager for temporary online adaptation.

        Within this scope, self-modification is enabled and weights may change.
        When the scope exits, weights are restored to their original values.

        This is useful for:
        - Safe inference-time adaptation that doesn't permanently modify the model
        - Experimenting with different prompts without accumulating weight drift
        - Testing adaptation behavior without affecting subsequent inferences

        Example:
            # Weights adapt within scope, then revert
            with hope.adaptation_scope():
                output = hope.generate(prompt, max_new_tokens=100)
            # Model weights are now restored to pre-scope values

        Args:
            enable_self_modification: Whether to enable self-mod within scope.
                                     Default True (otherwise why use this scope?)
        """
        # Save current state
        saved_state = self.get_self_mod_state()

        # Clear any pending updates from before the scope
        self.clear_pending_updates()

        try:
            yield
        finally:
            # Clear any pending updates accumulated during scope
            self.clear_pending_updates()

            # Restore original weights
            self.set_self_mod_state(saved_state)


class HOPEBlock(nn.Module):
    """
    Single HOPE block combining self-modification and continuum memory.

    Architecture:
        x -> LayerNorm -> Self-Modifying Attention -> + -> LayerNorm -> CMS -> + -> x'
                              |__________________________|                |______|

    The residual connections allow gradients to flow while the self-modifying
    attention and CMS components add their learned transformations.

    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        chunk_sizes: Update frequencies for CMS levels
        dropout: Dropout probability
        use_self_modification: If True, attention weights update online
        self_mod_lr: Learning rate for self-modification
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        chunk_sizes: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_self_modification: bool = True,
        self_mod_lr: float = 0.001,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.use_self_modification = use_self_modification

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Attention layer
        if use_self_modification:
            # True self-modifying attention (weights change during forward)
            self.attention = SelfModifyingAttention(
                dim=dim,
                num_heads=n_heads,
                self_mod_lr=self_mod_lr,
                dropout=dropout,
            )
        else:
            # Standard linear attention with external memory
            self.attention = LinearAttentionWithMemory(
                dim=dim,
                n_heads=n_heads,
                dropout=dropout,
            )

        # Continuum Memory System
        if chunk_sizes is None:
            chunk_sizes = [8, 32, 128]

        self.cms = ContinuumMemorySystem(
            dim=dim,
            hidden_dim=4 * dim,
            num_levels=len(chunk_sizes),
            chunk_sizes=chunk_sizes,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        enable_self_modification: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through HOPE block.

        Args:
            x: Input tensor (batch, seq_len, dim)
            enable_self_modification: If True and using self-modification,
                apply online weight updates during forward pass.
                NOTE: Self-modification now works during both training AND
                inference for online adaptation (per paper's intent).
        """
        # Self-modifying attention + residual
        # NOTE: Self-modification is allowed during inference for online adaptation
        # The paper describes self-modification as an online learning mechanism
        # that should work during deployment, not just training.
        if self.use_self_modification:
            x = x + self.attention(
                self.ln1(x),
                enable_self_modification=enable_self_modification
            )
        else:
            x = x + self.attention(self.ln1(x))

        # Continuum memory with residual
        x = x + self.cms(self.ln2(x))

        return x

    def get_self_mod_stats(self) -> dict:
        """Get self-modification statistics for this block."""
        stats = {'use_self_modification': self.use_self_modification}

        if self.use_self_modification and hasattr(self.attention, 'W_q'):
            stats['attention_updates'] = {
                'W_q': self.attention.W_q.update_count.item(),
                'W_k': self.attention.W_k.update_count.item(),
                'W_v': self.attention.W_v.update_count.item(),
                'W_o': self.attention.W_o.update_count.item(),
            }

        stats['cms_stats'] = self.cms.get_level_stats()

        return stats

    def apply_pending_updates(self):
        """Apply pending self-modification updates."""
        if self.use_self_modification and hasattr(self.attention, 'apply_pending_updates'):
            self.attention.apply_pending_updates()

    def clear_pending_updates(self):
        """Clear pending updates without applying them."""
        if self.use_self_modification and hasattr(self.attention, 'W_q'):
            self.attention.W_q.clear_pending_updates()
            self.attention.W_k.clear_pending_updates()
            self.attention.W_v.clear_pending_updates()
            self.attention.W_o.clear_pending_updates()


class LinearAttentionWithMemory(nn.Module):
    """
    Linear attention with external memory buffer (Fast Weights).

    This implements efficient linear attention with O(n) complexity
    using an accumulating memory state: M = M + v_t * k_t^T

    Note: This does NOT modify its own parameters. For true self-modification,
    use SelfModifyingAttention from titans.py.
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

        # Query, Key, Value projections
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)

        # Output projection
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # Feature map for linear attention (ELU + 1)
        self.feature_map = lambda x: F.elu(x) + 1

    def forward(self, x: torch.Tensor, reset_memory: bool = True) -> torch.Tensor:
        """
        Forward pass with linear attention.

        Args:
            x: Input (batch, seq_len, dim)
            reset_memory: Whether to reset memory (always True for this impl)

        Returns:
            Output (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Apply feature map for linear attention
        Q = self.feature_map(Q)
        K = self.feature_map(K)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Linear attention with causal masking via cumsum
        # Memory accumulates: M_t = M_{t-1} + k_t * v_t^T
        # Output: y_t = M_t * q_t / (sum(k) * q_t)

        # Efficient computation using cumulative sum
        KV = torch.einsum('bhnd,bhnm->bhdm', K, V)  # (batch, heads, head_dim, head_dim)

        # For causal, we need sequential accumulation
        outputs = []
        memory = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            device=x.device, dtype=x.dtype
        )
        key_sum = torch.zeros(
            batch_size, self.n_heads, self.head_dim,
            device=x.device, dtype=x.dtype
        )

        for t in range(seq_len):
            k_t = K[:, :, t, :]  # (batch, heads, head_dim)
            v_t = V[:, :, t, :]
            q_t = Q[:, :, t, :]

            # Update memory: M += k * v^T
            memory = memory + torch.einsum('bhd,bhe->bhde', k_t, v_t)
            key_sum = key_sum + k_t

            # Retrieve: y = M @ q / (k_sum @ q)
            numerator = torch.einsum('bhde,bhd->bhe', memory, q_t)
            denominator = torch.einsum('bhd,bhd->bh', key_sum, q_t).unsqueeze(-1) + 1e-6

            y_t = numerator / denominator
            outputs.append(y_t)

        # Stack outputs
        output = torch.stack(outputs, dim=2)  # (batch, heads, seq, head_dim)
        output = output.transpose(1, 2).contiguous()  # (batch, seq, heads, head_dim)
        output = output.view(batch_size, seq_len, self.dim)

        # Output projection
        output = self.W_o(output)
        output = self.dropout(output)

        return output


class HOPEForSequenceClassification(nn.Module):
    """
    HOPE model with a classification head.

    Useful for downstream tasks like sentiment analysis.
    """

    def __init__(
        self,
        num_classes: int,
        dim: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        vocab_size: int = 50257,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_self_modification: bool = True,
    ):
        super().__init__()

        self.hope = HOPE(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            use_self_modification=use_self_modification,
        )

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        enable_self_modification: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for classification.

        Args:
            input_ids: Token indices (batch, seq_len)
            labels: Class labels for loss computation
            enable_self_modification: Whether to allow online weight updates

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        # Get HOPE representations
        hidden_states = self.hope(
            input_ids,
            enable_self_modification=enable_self_modification,
        )

        # Pool: use [CLS] token or mean pooling
        pooled = hidden_states[:, 0, :]  # First token

        # Classify
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if labels is not None:
            return loss, logits
        return logits
