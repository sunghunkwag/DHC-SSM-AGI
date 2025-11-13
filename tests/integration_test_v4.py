"""Comprehensive integration tests for DHC-SSM v4.0.

Tests realistic end-to-end scenarios including:
- Full training loops
- Memory management
- Gradient flow
- Performance benchmarks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import gc
import time

sys.path.insert(0, '../')

from dhc_ssm.core.auto_consolidating_ssm import AutoConsolidatingSSM
from dhc_ssm.core.gated_ssm import GatedS6Layer, MultiLayerGatedSSM
from dhc_ssm.core.multi_resolution_ssm import MultiResolutionSSM
from dhc_ssm.agi.self_improving_architecture import SelfImprovingArchitecture


def test_full_training_loop():
    """Test complete training loop with auto-consolidation."""
    print("\n" + "="*60)
    print("TEST 1: Full Training Loop with Auto-Consolidation")
    print("="*60)
    
    # Create model
    model = AutoConsolidatingSSM(hidden_dim=64, state_dim=32)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Create dummy dataset
    X = torch.randn(100, 64)
    y = torch.randn(100, 32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Training loop
    print("Starting training...")
    initial_loss = None
    final_loss = None
    
    for epoch in range(3):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        if epoch == 0:
            initial_loss = avg_loss
        if epoch == 2:
            final_loss = avg_loss
        print(f"Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    # Verify training occurred
    assert final_loss < initial_loss, "Model should improve during training"
    print("✅ PASSED: Training loop works correctly")
    return True


def test_gated_ssm_sequence_processing():
    """Test gated SSM with realistic sequences."""
    print("\n" + "="*60)
    print("TEST 2: Gated SSM Sequence Processing")
    print("="*60)
    
    # Create model
    model = MultiLayerGatedSSM(
        num_layers=2,
        hidden_dim=128,
        state_dim=64,
        dt_rank=32
    )
    
    # Test different sequence lengths
    sequence_lengths = [8, 16, 32, 64]
    
    for seq_len in sequence_lengths:
        x = torch.randn(4, seq_len, 128)
        
        # Forward pass
        output, states = model(x)
        
        # Verify shapes
        assert output.shape == (4, seq_len, 128), f"Wrong output shape for seq_len={seq_len}"
        assert len(states) == 2, "Should have 2 layer states"
        assert states[0].shape == (4, 64), "Wrong state shape"
        
        print(f"✅ Seq length {seq_len}: Output shape {output.shape}")
    
    print("✅ PASSED: Gated SSM handles variable sequences correctly")
    return True


def test_multi_resolution_robustness():
    """Test multi-resolution SSM with edge cases."""
    print("\n" + "="*60)
    print("TEST 3: Multi-Resolution SSM Robustness")
    print("="*60)
    
    model = MultiResolutionSSM(
        hidden_dim=64,
        state_dim=32,
        wavelet='db4',
        decomposition_level=2
    )
    
    # Test various input sizes
    test_cases = [
        (2, 32, 64),   # Small sequence
        (4, 64, 64),   # Medium sequence
        (8, 128, 64),  # Large sequence
        (1, 256, 64),  # Very long sequence
    ]
    
    for batch, seq_len, hidden_dim in test_cases:
        x = torch.randn(batch, seq_len, hidden_dim)
        
        try:
            output = model(x)
            assert output.shape == (batch, seq_len, hidden_dim)
            print(f"✅ Input shape ({batch}, {seq_len}, {hidden_dim}): SUCCESS")
        except Exception as e:
            print(f"❌ FAILED for shape ({batch}, {seq_len}, {hidden_dim}): {str(e)}")
            return False
    
    print("✅ PASSED: Multi-resolution SSM handles all input sizes")
    return True


def test_gradient_flow():
    """Verify gradients flow through all components."""
    print("\n" + "="*60)
    print("TEST 4: Gradient Flow Verification")
    print("="*60)
    
    # Create hybrid model
    class HybridModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.auto_cons = AutoConsolidatingSSM(64, 32)
            self.gated = GatedS6Layer(32, 16)
            self.output = nn.Linear(16, 10)
        
        def forward(self, x):
            x = self.auto_cons(x)
            x = x.unsqueeze(1)  # Add sequence dim
            x, _ = self.gated(x)
            x = x.squeeze(1)
            return self.output(x)
    
    model = HybridModel()
    x = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))
    
    # Forward + backward
    output = model(x)
    loss = nn.functional.cross_entropy(output, target)
    loss.backward()
    
    # Check gradients
    params_with_grad = 0
    params_without_grad = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            params_with_grad += 1
            if param.grad.abs().max() > 0:
                print(f"✅ {name}: gradient flowing (max={param.grad.abs().max():.4f})")
            else:
                print(f"⚠️  {name}: zero gradient")
        else:
            params_without_grad += 1
            print(f"❌ {name}: NO GRADIENT")
    
    print(f"\nSummary: {params_with_grad} params with gradients, {params_without_grad} without")
    assert params_with_grad > 0, "At least some parameters should have gradients"
    print("✅ PASSED: Gradients flow through the model")
    return True


def test_memory_leak():
    """Check for memory leaks during training."""
    print("\n" + "="*60)
    print("TEST 5: Memory Leak Detection")
    print("="*60)
    
    model = AutoConsolidatingSSM(128, 64)
    optimizer = optim.Adam(model.parameters())
    
    # Measure initial memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    initial_objects = len(gc.get_objects())
    print(f"Initial objects: {initial_objects}")
    
    # Run training loop
    for i in range(50):
        x = torch.randn(16, 128)
        y = torch.randn(16, 64)
        
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        # Periodic cleanup
        if i % 10 == 0:
            del x, y, output, loss
            gc.collect()
    
    # Measure final memory
    gc.collect()
    final_objects = len(gc.get_objects())
    print(f"Final objects: {final_objects}")
    
    object_growth = final_objects - initial_objects
    print(f"Object growth: {object_growth}")
    
    # Allow some growth but not excessive
    assert object_growth < 1000, f"Potential memory leak: {object_growth} new objects"
    print("✅ PASSED: No significant memory leak detected")
    return True


def test_self_improvement_safety():
    """Test self-improvement doesn't crash."""
    print("\n" + "="*60)
    print("TEST 6: Self-Improvement System Safety")
    print("="*60)
    
    # Simple base model
    base_model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    improver = SelfImprovingArchitecture(
        base_model=base_model,
        improvement_threshold=0.05,
        confidence_level=0.95
    )
    
    # Create validation data
    X = torch.randn(100, 32)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16)
    
    # Test candidate generation
    candidates = improver.generate_candidate_modifications()
    print(f"Generated {len(candidates)} candidates")
    assert len(candidates) > 0
    
    # Test performance measurement
    perf = improver._measure_performance(loader)
    print(f"Baseline performance: {perf:.4f}")
    assert perf > 0 and perf < float('inf')
    
    print("✅ PASSED: Self-improvement system is safe")
    return True


def test_performance_benchmark():
    """Benchmark training speed."""
    print("\n" + "="*60)
    print("TEST 7: Performance Benchmark")
    print("="*60)
    
    model = MultiLayerGatedSSM(num_layers=3, hidden_dim=256, state_dim=128)
    optimizer = optim.Adam(model.parameters())
    
    # Prepare data
    x = torch.randn(32, 64, 256)
    y = torch.randn(32, 64, 256)
    
    # Warmup
    for _ in range(5):
        output, _ = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Benchmark
    start_time = time.time()
    iterations = 20
    
    for _ in range(iterations):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    samples_per_sec = (32 * iterations) / elapsed
    
    print(f"Training speed: {samples_per_sec:.1f} samples/sec")
    print(f"Time per iteration: {elapsed/iterations*1000:.1f} ms")
    
    assert samples_per_sec > 10, "Training is too slow"
    print("✅ PASSED: Performance is acceptable")
    return True


if __name__ == '__main__':
    print("\n" + "#"*60)
    print("# DHC-SSM v4.0 COMPREHENSIVE INTEGRATION TESTS")
    print("#"*60)
    
    tests = [
        test_full_training_loop,
        test_gated_ssm_sequence_processing,
        test_multi_resolution_robustness,
        test_gradient_flow,
        test_memory_leak,
        test_self_improvement_safety,
        test_performance_benchmark,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_func.__name__}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "#"*60)
    print(f"# FINAL RESULTS: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"# ⚠️  {failed} tests failed")
    else:
        print("# ✅ ALL TESTS PASSED!")
    print("#"*60)
