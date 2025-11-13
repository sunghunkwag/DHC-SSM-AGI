# DHC-SSM v4.0 Testing Guide

## Quick Start

### Installation

```bash
# Clone repository
git clone -b feature/v4.0-auto-consolidation https://github.com/sunghunkwag/DHC-SSM-AGI.git
cd DHC-SSM-AGI

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Running Tests

### 1. Unit Tests (Fast)

Test individual v4.0 components:

```bash
# All unit tests
pytest tests/test_v4_features.py -v

# Specific component tests
pytest tests/test_v4_features.py::TestAutoConsolidatingSSM -v
pytest tests/test_v4_features.py::TestGatedSSM -v
pytest tests/test_v4_features.py::TestMultiResolutionSSM -v
pytest tests/test_v4_features.py::TestSelfImprovingArchitecture -v
```

**Expected output:**
```
======================== 17 passed in 3.2s ========================
```

---

### 2. Integration Tests (Comprehensive)

Test end-to-end scenarios:

```bash
# Run all integration tests
python tests/integration_test_v4.py
```

**Expected output:**
```
#############################################################
# DHC-SSM v4.0 COMPREHENSIVE INTEGRATION TESTS
#############################################################

[TEST 1] Full Training Loop with Auto-Consolidation
✅ PASSED: Training loop works correctly

[TEST 2] Gated SSM Sequence Processing
✅ PASSED: Gated SSM handles variable sequences correctly

[TEST 3] Multi-Resolution SSM Robustness
✅ PASSED: Multi-resolution SSM handles all input sizes

[TEST 4] Gradient Flow Verification
✅ PASSED: Gradients flow through the model

[TEST 5] Memory Leak Detection
✅ PASSED: No significant memory leak detected

[TEST 6] Self-Improvement System Safety
✅ PASSED: Self-improvement system is safe

[TEST 7] Performance Benchmark
✅ PASSED: Performance is acceptable

#############################################################
# FINAL RESULTS: 7/7 tests passed
# ✅ ALL TESTS PASSED!
#############################################################
```

---

### 3. Quick Smoke Test

Minimal test to verify installation:

```python
python -c "
import torch
from dhc_ssm import AutoConsolidatingSSM, GatedS6Layer, MultiResolutionSSM

# Test 1: Auto-consolidating SSM
model1 = AutoConsolidatingSSM(64, 32)
x1 = torch.randn(8, 64)
out1 = model1(x1)
print(f'✅ AutoConsolidatingSSM: {out1.shape}')

# Test 2: Gated SSM
model2 = GatedS6Layer(128, 64)
x2 = torch.randn(4, 16, 128)
out2, _ = model2(x2)
print(f'✅ GatedS6Layer: {out2.shape}')

# Test 3: Multi-resolution SSM
model3 = MultiResolutionSSM(64, 32)
x3 = torch.randn(4, 128, 64)
out3 = model3(x3)
print(f'✅ MultiResolutionSSM: {out3.shape}')

print('\n✅ All components working!')
"
```

---

## Troubleshooting

### Issue 1: ModuleNotFoundError for PyWavelets

**Error:**
```
ModuleNotFoundError: No module named 'pywt'
```

**Solution:**
```bash
pip install PyWavelets>=1.4.0
```

---

### Issue 2: Import Error for v4.0 Components

**Error:**
```
ImportError: cannot import name 'AutoConsolidatingSSM'
```

**Solution:**
```bash
# Reinstall in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="$PWD:$PYTHONPATH"
```

---

### Issue 3: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use smaller models for testing
model = AutoConsolidatingSSM(hidden_dim=64, state_dim=32)  # Instead of 256, 128

# Or run on CPU
device = 'cpu'
model = model.to(device)
```

---

### Issue 4: Gradient Hook Warnings

**Warning:**
```
UserWarning: Backward hook registered twice
```

**Solution:**
This is expected behavior when reloading models. Use `model.reset_buffers()` between training sessions.

---

## Performance Benchmarking

### Measure Training Speed

```python
import torch
import time
from dhc_ssm import MultiLayerGatedSSM

model = MultiLayerGatedSSM(num_layers=3, hidden_dim=256, state_dim=128)
optimizer = torch.optim.Adam(model.parameters())

x = torch.randn(32, 64, 256)
y = torch.randn(32, 64, 256)

# Warmup
for _ in range(5):
    out, _ = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benchmark
start = time.time()
for _ in range(20):
    optimizer.zero_grad()
    out, _ = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()

elapsed = time.time() - start
print(f"Training speed: {(32 * 20) / elapsed:.1f} samples/sec")
```

**Expected performance:**
- CPU: 50-200 samples/sec
- GPU (CUDA): 500-2000 samples/sec

---

### Memory Profiling

```python
import torch
import gc
from dhc_ssm import AutoConsolidatingSSM

model = AutoConsolidatingSSM(128, 64)
optimizer = torch.optim.Adam(model.parameters())

# Check memory before
gc.collect()
initial_objects = len(gc.get_objects())
print(f"Initial objects: {initial_objects}")

# Run training
for i in range(100):
    x = torch.randn(16, 128)
    y = torch.randn(16, 64)
    
    optimizer.zero_grad()
    output = model(x)
    loss = torch.nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    
    if i % 20 == 0:
        del x, y, output, loss
        gc.collect()

# Check memory after
gc.collect()
final_objects = len(gc.get_objects())
print(f"Final objects: {final_objects}")
print(f"Object growth: {final_objects - initial_objects}")

# Growth < 1000 indicates no memory leak
assert final_objects - initial_objects < 1000, "Potential memory leak!"
print("✅ No memory leak detected")
```

---

## Continuous Integration

### GitHub Actions (Recommended)

Create `.github/workflows/test.yml`:

```yaml
name: DHC-SSM Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/test_v4_features.py -v --cov=dhc_ssm
    
    - name: Run integration tests
      run: |
        python tests/integration_test_v4.py
```

---

## Test Coverage

### Generate Coverage Report

```bash
# Install coverage tools
pip install pytest-cov

# Run with coverage
pytest tests/test_v4_features.py --cov=dhc_ssm --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

**Target coverage: >80%**

---

## Known Issues

### Issue: Wavelet Decomposition on Short Sequences

**Symptom:** `IndexError` with sequences < 32 tokens

**Status:** ✅ FIXED in commit `2f0c6bd`

**Workaround (if using older version):**
```python
# Pad short sequences
if x.shape[1] < 32:
    padding = torch.zeros(x.shape[0], 32 - x.shape[1], x.shape[2])
    x = torch.cat([x, padding], dim=1)
```

---

### Issue: Matrix Dimension Mismatch in Gated SSM

**Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**Status:** ✅ FIXED in commit `6eb77bb`

**Solution:** Update to latest version

---

## Getting Help

If tests fail:

1. Check you're on the correct branch: `feature/v4.0-auto-consolidation`
2. Ensure all dependencies are installed: `pip install -r requirements.txt`
3. Try running individual tests to isolate the issue
4. Check `BUGFIXES_v4.0.md` for known issues
5. Open an issue on GitHub with:
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - Error traceback
   - System info (OS, CUDA version if using GPU)

---

**Last Updated:** November 13, 2025  
**Version:** 4.0.0  
**Status:** ✅ All Tests Passing
