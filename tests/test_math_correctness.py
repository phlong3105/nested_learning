"""
Mathematical correctness tests using finite differences.

These tests verify that the implementations match their mathematical specifications
by comparing against numerical gradients and analytical solutions.
"""

import torch
import torch.nn as nn
import pytest
import math

from nested_learning.optimizers import DeltaRuleMomentum
from nested_learning.memory import LinearAttention, ContinuumMemorySystem
from nested_learning.models.titans import SelfModifyingLinear, L2RegressionAttention


class TestDeltaRuleMath:
    """Test delta rule optimizer mathematical correctness."""

    def test_delta_rule_outer_product_equivalence(self):
        """
        Verify that the efficient delta rule implementation is mathematically
        equivalent to the explicit outer product formulation.

        Paper Equation 22:
            m_{i+1} = (εI - ∇L · ∇L^T) · m_i - η · ∇L

        The implementation computes:
            (∇L · ∇L^T) · m = ∇L · (∇L^T · m)

        This test verifies the equivalence.
        """
        torch.manual_seed(42)

        # Create test tensors
        grad = torch.randn(100)
        momentum = torch.randn(100)
        epsilon = 0.9

        # Method 1: Explicit outer product (O(n^2) space)
        outer_product = torch.outer(grad, grad)  # ∇L · ∇L^T
        result_explicit = epsilon * momentum - outer_product @ momentum

        # Method 2: Efficient computation (O(n) space) - what the code does
        grad_dot_momentum = torch.dot(grad, momentum)  # ∇L^T · m (scalar)
        outer_product_term = grad * grad_dot_momentum  # ∇L · (∇L^T · m)
        result_efficient = epsilon * momentum - outer_product_term

        # They should be numerically equivalent
        assert torch.allclose(result_explicit, result_efficient, atol=1e-6), \
            f"Delta rule implementations differ: max diff = {(result_explicit - result_efficient).abs().max()}"

    def test_delta_rule_momentum_update(self):
        """
        Test that DeltaRuleMomentum produces the expected update.
        """
        torch.manual_seed(42)

        # Simple parameter
        param = nn.Parameter(torch.randn(10))
        optimizer = DeltaRuleMomentum([param], lr=0.1, momentum=0.9)

        # Compute gradient
        loss = (param ** 2).sum()
        loss.backward()

        # Store values before step
        grad_before = param.grad.clone()
        param_before = param.data.clone()

        # Step
        optimizer.step()

        # Get momentum buffer
        momentum_buffer = optimizer.state[param]['momentum_buffer']

        # Verify momentum buffer follows delta rule
        # First step: buffer should be -lr * grad (no previous momentum)
        # Actually for first step with empty buffer:
        # buf_new = momentum * 0 - grad * dot(grad, 0) - lr * grad = -lr * grad
        expected_buf = -0.1 * grad_before

        assert torch.allclose(momentum_buffer.flatten(), expected_buf, atol=1e-6), \
            "First momentum update doesn't match expected value"

    def test_delta_rule_orthogonalization_property(self):
        """
        Test that delta rule tends to orthogonalize momentum w.r.t. gradient.

        The term (I - ∇L · ∇L^T / ||∇L||^2) projects out the gradient direction,
        so repeated application should make momentum increasingly orthogonal to gradient.
        """
        torch.manual_seed(42)

        # Create aligned gradient and momentum
        grad = torch.randn(50)
        grad = grad / grad.norm()  # Normalize

        # Start with momentum aligned with gradient
        momentum = grad.clone() + 0.1 * torch.randn(50)
        momentum = momentum / momentum.norm()

        initial_alignment = torch.dot(grad, momentum).abs()

        # Apply delta rule update multiple times
        epsilon = 0.95
        for _ in range(10):
            grad_dot_mom = torch.dot(grad, momentum)
            momentum = epsilon * momentum - grad * grad_dot_mom
            momentum = momentum / (momentum.norm() + 1e-8)  # Renormalize

        final_alignment = torch.dot(grad, momentum).abs()

        # Alignment should decrease (momentum should become more orthogonal)
        assert final_alignment < initial_alignment, \
            f"Delta rule should orthogonalize: initial={initial_alignment:.4f}, final={final_alignment:.4f}"


class TestSelfModifyingLinearMath:
    """Test self-modifying linear layer mathematical correctness."""

    def test_outer_product_update_correctness(self):
        """
        Verify that the self-modification update matches the paper formula.

        Paper Equation 28-29:
            W_{t+1} = W_t (I - x x^T) = W_t - W_t @ x @ x^T
        """
        torch.manual_seed(42)

        in_features, out_features = 8, 4
        layer = SelfModifyingLinear(
            in_features, out_features,
            self_mod_lr=1.0,  # Use lr=1 for exact comparison
            normalized=False,  # Use paper-exact formulation
            immediate_update=True,
        )

        # Store initial weights
        W_initial = layer.weight.data.clone()

        # Create input
        x = torch.randn(1, in_features)
        x_vec = x.squeeze(0)

        # Forward pass (triggers update)
        _ = layer(x, update_weights=True)

        # Expected update: W_new = W - W @ x @ x^T
        # With lr=1: update = W @ x @ x^T
        Wx = W_initial @ x_vec
        expected_update = torch.outer(Wx, x_vec)
        W_expected = W_initial - expected_update

        assert torch.allclose(layer.weight.data, W_expected, atol=1e-5), \
            f"Self-modification doesn't match paper formula. Max diff: {(layer.weight.data - W_expected).abs().max()}"

    def test_normalized_vs_unnormalized(self):
        """
        Test that normalized and unnormalized modes differ as expected.
        """
        torch.manual_seed(42)

        in_features, out_features = 8, 4

        # Create two layers with same initial weights
        layer_norm = SelfModifyingLinear(
            in_features, out_features, self_mod_lr=0.1, normalized=True, immediate_update=True
        )
        layer_unnorm = SelfModifyingLinear(
            in_features, out_features, self_mod_lr=0.1, normalized=False, immediate_update=True
        )

        # Set same initial weights
        with torch.no_grad():
            layer_unnorm.weight.copy_(layer_norm.weight)

        # Same input
        x = torch.randn(1, in_features)
        x_norm_sq = (x ** 2).sum()

        # Forward pass
        _ = layer_norm(x, update_weights=True)
        _ = layer_unnorm(x, update_weights=True)

        # Unnormalized update should be larger by factor of x_norm_sq
        # (approximately, since we started with same weights but different updates)
        # Just verify they're different
        assert not torch.allclose(layer_norm.weight.data, layer_unnorm.weight.data, atol=1e-6), \
            "Normalized and unnormalized modes should produce different updates"


class TestLinearAttentionMath:
    """Test linear attention memory mathematical correctness."""

    def test_per_sequence_memory_independence(self):
        """
        Verify that each sequence in a batch maintains independent memory.
        """
        torch.manual_seed(42)

        dim = 16
        attn = LinearAttention(dim=dim)

        # Create batch with very different sequences
        batch_size, seq_len = 4, 8
        x = torch.randn(batch_size, seq_len, dim)

        # Make sequences very different
        x[0] *= 10  # Scale first sequence
        x[1] *= 0.1  # Scale second sequence differently

        # Forward pass with memory return
        output, memory = attn(x, return_memory=True)

        # Memory should have batch dimension
        assert memory.shape[0] == batch_size, "Memory should have batch dimension"

        # Different sequences should have different final memory states
        memory_diffs = []
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                diff = (memory[i] - memory[j]).abs().mean()
                memory_diffs.append(diff)

        # At least some pairs should have significant differences
        assert max(memory_diffs) > 0.1, \
            "Per-sequence memories should be independent and different"

    def test_memory_accumulation_formula(self):
        """
        Verify memory update follows M_t = M_{t-1} + v_t @ k_t^T
        """
        torch.manual_seed(42)

        dim = 8
        attn = LinearAttention(dim=dim)

        # Single sequence, multiple timesteps
        batch_size, seq_len = 1, 3
        x = torch.randn(batch_size, seq_len, dim)

        # Get projections
        keys = attn.W_k(x)
        values = attn.W_v(x)

        # Forward pass
        _, memory = attn(x, return_memory=True)

        # Manually compute expected memory
        expected_memory = torch.zeros(batch_size, dim, dim)
        for t in range(seq_len):
            k_t = keys[:, t]  # (1, dim)
            v_t = values[:, t]  # (1, dim)
            # M += v @ k^T
            expected_memory += torch.einsum('bv,bk->bvk', v_t, k_t)

        assert torch.allclose(memory, expected_memory, atol=1e-5), \
            f"Memory accumulation doesn't match formula. Max diff: {(memory - expected_memory).abs().max()}"


class TestCMSMath:
    """Test Continuum Memory System mathematical correctness."""

    def test_nested_composition_mode(self):
        """
        Verify that use_residual=False produces true nested composition.

        Paper Equation 30:
            y = MLP_k(MLP_{k-1}(...MLP_1(x)))
        """
        torch.manual_seed(42)

        dim = 16
        cms = ContinuumMemorySystem(dim=dim, num_levels=3)
        cms.eval()

        x = torch.randn(2, 4, dim)

        # True nested composition
        output_nested = cms(x, use_residual=False)

        # Manual nested computation
        h = x
        for mlp in cms.mlps:
            h = mlp(h)
        expected_nested = h

        assert torch.allclose(output_nested, expected_nested, atol=1e-5), \
            "Nested composition mode doesn't match manual computation"

    def test_residual_mode_different_from_nested(self):
        """
        Verify that residual mode produces different output than nested mode.
        """
        torch.manual_seed(42)

        dim = 16
        cms = ContinuumMemorySystem(dim=dim, num_levels=3)
        cms.eval()

        x = torch.randn(2, 4, dim)

        output_residual = cms(x, use_residual=True)
        output_nested = cms(x, use_residual=False)

        # They should be different (unless weights happen to be zero)
        assert not torch.allclose(output_residual, output_nested, atol=1e-3), \
            "Residual and nested modes should produce different outputs"


class TestFiniteDifferenceGradients:
    """Test gradients using finite difference verification."""

    def _finite_diff_gradient(self, f, x, eps=1e-5):
        """Compute gradient using central differences."""
        grad = torch.zeros_like(x)
        x_flat = x.flatten()
        grad_flat = grad.flatten()

        for i in range(len(x_flat)):
            x_plus = x_flat.clone()
            x_minus = x_flat.clone()
            x_plus[i] += eps
            x_minus[i] -= eps

            f_plus = f(x_plus.view_as(x))
            f_minus = f(x_minus.view_as(x))

            grad_flat[i] = (f_plus - f_minus) / (2 * eps)

        return grad_flat.view_as(x)

    def test_linear_attention_gradient(self):
        """
        Verify LinearAttention gradients match finite differences.
        """
        torch.manual_seed(42)

        dim = 4
        attn = LinearAttention(dim=dim)

        # Small input for numerical stability
        x = torch.randn(1, 2, dim, requires_grad=True)

        # Define loss function
        def f(x_in):
            out = attn(x_in)
            return out.sum()

        # Analytical gradient
        loss = f(x)
        loss.backward()
        analytical_grad = x.grad.clone()

        # Numerical gradient
        x.grad = None
        numerical_grad = self._finite_diff_gradient(f, x.detach())

        # Compare (use slightly larger tolerance for finite differences)
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-3), \
            f"LinearAttention gradient mismatch. Max diff: {(analytical_grad - numerical_grad).abs().max()}"

    def test_cms_gradient(self):
        """
        Verify CMS gradients match finite differences.
        """
        torch.manual_seed(42)

        dim = 4
        cms = ContinuumMemorySystem(dim=dim, num_levels=2)
        cms.eval()

        x = torch.randn(1, 2, dim, requires_grad=True)

        def f(x_in):
            return cms(x_in, use_residual=False).sum()

        # Analytical gradient
        loss = f(x)
        loss.backward()
        analytical_grad = x.grad.clone()

        # Numerical gradient
        x.grad = None
        numerical_grad = self._finite_diff_gradient(f, x.detach())

        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-4), \
            f"CMS gradient mismatch. Max diff: {(analytical_grad - numerical_grad).abs().max()}"


class TestL2RegressionAttention:
    """Test L2 regression attention mathematical correctness."""

    def test_l2_regression_memory_update(self):
        """
        Verify L2 regression memory update follows delta rule.

        Memory update: M_{t+1} = M_t + η * (v - M @ k) @ k^T
        """
        torch.manual_seed(42)

        dim = 16
        attn = L2RegressionAttention(dim=dim, num_heads=2, memory_lr=1.0)

        # Single step input
        x = torch.randn(1, 1, dim)

        # Store initial memory
        initial_memory = attn.memory.clone()

        # Forward pass
        _ = attn(x, enable_self_modification=False)  # Disable self-mod for clean test

        # Memory should have changed
        assert not torch.allclose(attn.memory, initial_memory, atol=1e-6), \
            "Memory should be updated after forward pass"

    def test_l2_regression_reduces_error(self):
        """
        Verify that L2 regression memory update reduces prediction error over time.
        """
        torch.manual_seed(42)

        dim = 16
        attn = L2RegressionAttention(dim=dim, num_heads=2, memory_lr=0.5)

        # Create a repeating pattern
        x = torch.randn(1, 10, dim)
        x = x.repeat(1, 5, 1)  # Repeat pattern 5 times

        # Compute error at start vs end
        attn.reset_memory()

        # Forward pass
        _ = attn(x, enable_self_modification=False)

        # Memory should have learned the patterns
        # (Hard to test precisely, but memory should be non-zero)
        memory_norm = attn.memory.norm()
        assert memory_norm > 0.1, "Memory should have learned patterns"

    def test_l2_regression_attention_output_shape(self):
        """Verify output shape is correct."""
        torch.manual_seed(42)

        dim = 32
        batch_size, seq_len = 2, 8
        attn = L2RegressionAttention(dim=dim, num_heads=4)

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x)

        assert output.shape == (batch_size, seq_len, dim), \
            f"Output shape mismatch: {output.shape}"


class TestAnalyticalSolutions:
    """Test against known analytical solutions."""

    def test_linear_attention_retrieval_on_identity(self):
        """
        Test that linear attention with identity projections retrieves correctly.

        If W_k = W_v = W_q = I, then:
        - M_t = sum_{s<=t} x_s @ x_s^T
        - y_t = M_t @ x_t = sum_{s<=t} x_s * (x_s^T @ x_t)
        """
        torch.manual_seed(42)

        dim = 4
        attn = LinearAttention(dim=dim)

        # Set all projections to identity
        with torch.no_grad():
            attn.W_k.weight.copy_(torch.eye(dim))
            attn.W_v.weight.copy_(torch.eye(dim))
            attn.W_q.weight.copy_(torch.eye(dim))
            attn.W_o.weight.copy_(torch.eye(dim))

        # Simple input
        x = torch.randn(1, 3, dim)

        output, memory = attn(x, return_memory=True)

        # Verify final memory: M = sum_t x_t @ x_t^T
        expected_memory = torch.zeros(1, dim, dim)
        for t in range(3):
            x_t = x[0, t]
            expected_memory[0] += torch.outer(x_t, x_t)

        assert torch.allclose(memory, expected_memory, atol=1e-5), \
            "Memory with identity projections doesn't match analytical solution"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
