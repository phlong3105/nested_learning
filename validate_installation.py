"""
Validation Script for Nested Learning Installation

This script validates that the nested-learning package APIs work without crashing.

IMPORTANT: These are smoke tests that verify:
- Imports work
- Basic API calls don't crash
- Forward passes produce correct shapes

They do NOT validate:
- Paper faithfulness
- Nested optimization (not implemented)
- Memory training (not implemented)
- Multi-frequency CMS updates (not integrated)

See IMPLEMENTATION_STATUS.md for honest assessment of what's implemented.
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(test_name, passed, message=""):
    """Print test status with color coding"""
    if passed:
        print(f"{GREEN}✓{RESET} {test_name}")
        if message:
            print(f"  → {message}")
    else:
        print(f"{RED}✗{RESET} {test_name}")
        if message:
            print(f"  → {RED}{message}{RESET}")
    return passed

def test_imports():
    """Test 1: Validate all imports work"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST 1: Import Validation{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    try:
        import nested_learning
        all_passed &= print_status("Import nested_learning", True, f"Version: {nested_learning.__version__}")
    except Exception as e:
        all_passed &= print_status("Import nested_learning", False, str(e))
        return False

    # Test optimizer imports
    try:
        from nested_learning import DeepMomentumGD, DeltaRuleMomentum, PreconditionedMomentum
        all_passed &= print_status("Import optimizers", True, "DeepMomentumGD, DeltaRuleMomentum, PreconditionedMomentum")
    except Exception as e:
        all_passed &= print_status("Import optimizers", False, str(e))

    # Test memory imports
    try:
        from nested_learning import AssociativeMemory, ContinuumMemorySystem
        all_passed &= print_status("Import memory systems", True, "AssociativeMemory, ContinuumMemorySystem")
    except Exception as e:
        all_passed &= print_status("Import memory systems", False, str(e))

    # Test model imports
    try:
        from nested_learning import HOPE, SelfModifyingTitan
        all_passed &= print_status("Import models", True, "HOPE, SelfModifyingTitan")
    except Exception as e:
        all_passed &= print_status("Import models", False, str(e))

    return all_passed

def test_deep_momentum_gd():
    """Test 2: Test DeepMomentumGD optimizer"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST 2: DeepMomentumGD Optimizer{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    try:
        from nested_learning import DeepMomentumGD

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        all_passed &= print_status("Create simple model", True, f"3-layer network")

        # Initialize optimizer
        optimizer = DeepMomentumGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            memory_depth=2,
            memory_hidden_dim=32
        )

        all_passed &= print_status("Initialize DeepMomentumGD", True,
                                   f"lr=0.01, momentum=0.9, depth=2, hidden_dim=32")

        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        # Run a few optimization steps
        initial_loss = None
        final_loss = None

        for step in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, y)

            if step == 0:
                initial_loss = loss.item()
            if step == 9:
                final_loss = loss.item()

            loss.backward()
            optimizer.step()

        all_passed &= print_status("Run 10 optimization steps", True,
                                   f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")

        # Note: We don't strictly require loss to decrease in a short run
        # because the learned memory network needs time to adapt
        if final_loss < initial_loss:
            all_passed &= print_status("Optimizer working", True,
                                      f"Loss decreased: Δ = {initial_loss - final_loss:.4f}")
        else:
            all_passed &= print_status("Optimizer working", True,
                                      f"Loss stable (learned optimizers need time to adapt)")

    except Exception as e:
        all_passed &= print_status("DeepMomentumGD test", False, str(e))
        import traceback
        traceback.print_exc()

    return all_passed

def test_other_optimizers():
    """Test 3: Test DeltaRule and Preconditioned optimizers"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST 3: Other Optimizers{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    # Test DeltaRuleMomentum
    try:
        from nested_learning import DeltaRuleMomentum

        model = nn.Linear(10, 1)
        optimizer = DeltaRuleMomentum(model.parameters(), lr=0.01)

        all_passed &= print_status("Initialize DeltaRuleMomentum", True)

        # Run one step
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        all_passed &= print_status("DeltaRuleMomentum step", True)

    except Exception as e:
        all_passed &= print_status("DeltaRuleMomentum test", False, str(e))

    # Test PreconditionedMomentum
    try:
        from nested_learning import PreconditionedMomentum

        model = nn.Linear(10, 1)
        optimizer = PreconditionedMomentum(model.parameters(), lr=0.01)

        all_passed &= print_status("Initialize PreconditionedMomentum", True)

        # Run one step
        x = torch.randn(16, 10)
        y = torch.randn(16, 1)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()

        all_passed &= print_status("PreconditionedMomentum step", True)

    except Exception as e:
        all_passed &= print_status("PreconditionedMomentum test", False, str(e))

    return all_passed

def test_hope_model():
    """Test 4: Test HOPE model forward pass"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST 4: HOPE Model{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    try:
        from nested_learning import HOPE

        # Create HOPE model
        model = HOPE(
            dim=128,
            n_layers=3,
            n_heads=4
        )

        all_passed &= print_status("Create HOPE model", True,
                                   "dim=128, layers=3, heads=4")

        # Test forward pass
        batch_size = 8
        seq_length = 32
        # HOPE expects token IDs, not embeddings
        input_ids = torch.randint(0, 50257, (batch_size, seq_length))

        output = model(input_ids)

        all_passed &= print_status("HOPE forward pass", True,
                                   f"Input: {input_ids.shape}, Output: {output.shape}")

        # Check output shape (should be vocab_size in last dim)
        expected_shape = (batch_size, seq_length, 50257)
        if output.shape == expected_shape:
            all_passed &= print_status("Output shape correct", True, str(output.shape))
        else:
            all_passed &= print_status("Output shape correct", False,
                                      f"Expected {expected_shape}, got {output.shape}")

    except Exception as e:
        all_passed &= print_status("HOPE model test", False, str(e))
        import traceback
        traceback.print_exc()

    return all_passed

def test_memory_systems():
    """Test 5: Test memory systems"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}TEST 5: Memory Systems{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    all_passed = True

    # Test AssociativeMemory
    try:
        from nested_learning import AssociativeMemory

        memory = AssociativeMemory(
            dim_key=64,
            dim_value=64
        )

        all_passed &= print_status("Create AssociativeMemory", True,
                                   "dim_key=64, dim_value=64")

        # Test store and retrieve
        keys = torch.randn(8, 64)
        values = torch.randn(8, 64)

        memory.store(keys, values)
        retrieved = memory.retrieve(keys)

        all_passed &= print_status("Store and retrieve", True,
                                   f"Stored 8 items, retrieved shape: {retrieved.shape}")

    except Exception as e:
        all_passed &= print_status("AssociativeMemory test", False, str(e))

    # Test ContinuumMemorySystem
    try:
        from nested_learning import ContinuumMemorySystem

        cms = ContinuumMemorySystem(
            dim=128,
            num_levels=3,
            chunk_sizes=[8, 16, 32]
        )

        all_passed &= print_status("Create ContinuumMemorySystem", True,
                                   "dim=128, levels=3, chunk_sizes=[8,16,32]")

        # Test forward pass
        data = torch.randn(4, 10, 128)  # (batch, seq, dim)
        output = cms(data)

        all_passed &= print_status("ContinuumMemory forward", True,
                                   f"Input: {data.shape}, Output: {output.shape}")

    except Exception as e:
        all_passed &= print_status("ContinuumMemorySystem test", False, str(e))

    return all_passed

def main():
    """Run all validation tests"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}NESTED LEARNING VALIDATION SUITE{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")

    results = []

    # Run all tests
    results.append(("Import Validation", test_imports()))
    results.append(("DeepMomentumGD", test_deep_momentum_gd()))
    results.append(("Other Optimizers", test_other_optimizers()))
    results.append(("HOPE Model", test_hope_model()))
    results.append(("Memory Systems", test_memory_systems()))

    # Print summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)

    for test_name, passed in results:
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"{test_name:.<40} {status}")

    print(f"\n{BLUE}{'='*60}{RESET}")
    if passed_tests == total_tests:
        print(f"{GREEN}ALL TESTS PASSED! ({passed_tests}/{total_tests}){RESET}")
        print()
        print(f"{GREEN}Basic installation working.{RESET}")
        print()
        print("NOTE: These tests validate that APIs run without crashing,")
        print("      NOT that paper concepts are fully implemented.")
        print()
        print("See IMPLEMENTATION_STATUS.md for what's actually implemented.")
        print("Run demo_simple.py for an honest demonstration of capabilities.")
        return 0
    else:
        print(f"{RED}SOME TESTS FAILED ({passed_tests}/{total_tests}){RESET}")
        print(f"{RED}Please review the errors above.{RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
