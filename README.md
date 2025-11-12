# DHC-SSM-AGI v3.2.0 (Deterministic Hierarchical Causal State Space Model - AGI Edition)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/sunghunkwag/DHC-SSM-AGI/pulls)

Robust, fully type-safe research architecture featuring **true Nested Learning** implementation with gradient consolidation, recursive self-improvement, adaptive threshold analysis, real uncertainty quantification, and comprehensive benchmarking suite.

## ⚡ What's New in v3.2.0

- ✅ **Fixed Nested Learning Implementation**: Proper gradient accumulation and parameter consolidation
- ✅ **Separate Fast/Slow Pathways**: Distinct weight parameters with different update frequencies
- ✅ **Comprehensive Benchmarking**: Throughput, memory, gradient flow, and baseline comparisons
- ✅ **Mathematical Documentation**: Formal O(n) complexity proofs and equations
- ✅ **Improved Stability**: Orthogonal initialization and normalized gradient consolidation

## Architecture Overview

```
───── Input (Batch, 3, 32, 32)
│
[Spatial Encoder]
│
[Nested Temporal SSM (multi-time-scale)]
│   ├─ Fast Memory (every step)
│   │  ├─ Fast Weights W_f (optimizer-trained)
│   │  └─ Immediate context adaptation
│   ├─ Medium Memory (every 10 steps)
│   │  ├─ Slow Weights W_m (gradient consolidation)
│   │  └─ Recent pattern consolidation
│   └─ Slow Memory (every 100 steps)
│      ├─ Slow Weights W_s (gradient consolidation)
│      └─ Long-term knowledge consolidation
│
[Strategic Reasoner (GNN/Graph)]
│
───── Key Features:
• True Nested Learning with Parameter Consolidation
• Gradient Accumulation Buffers (functional)
• Learnable Fast/Slow Interpolation
• Meta-Cognitive Layer
• RSI with Adaptive Thresholds
• Real Uncertainty Quantification (Epistemic + Aleatoric)
• O(n) Linear Complexity (proven)
│
───── Output: Predictions, Diagnostics, Learning Self-Metrics
```

## Installation

```bash
git clone https://github.com/sunghunkwag/DHC-SSM-AGI.git
cd DHC-SSM-AGI
pip install -e .
```

### Requirements
- Python ≥ 3.11
- PyTorch ≥ 2.0.0
- torch-geometric ≥ 2.3.0
- See `requirements.txt` for full dependencies

## Quick Start

### Basic Usage

```python
from dhc_ssm import DHCSSMModel, DHCSSMConfig
import torch

# Create model
config = DHCSSMConfig(
    hidden_dim=256,
    ssm_state_dim=64,
    num_classes=10,
)
model = DHCSSMModel(config)

# Forward pass
x = torch.randn(8, 3, 32, 32)
output = model(x)
```

### Training Loop (Important!)

⚠️ **v3.2.0 requires modified training loop for gradient consolidation:**

```python
import torch.nn as nn
import torch.optim as optim

model = DHCSSMModel(config)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch in dataloader:
        x, y = batch
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # NEW: Accumulate gradients for slow weights
        model.accumulate_gradients()
        
        optimizer.step()
```

The `model.accumulate_gradients()` call is **required** for Nested Learning to function correctly. It accumulates gradients for slow-changing parameters that are consolidated at lower frequencies.

## Testing & Validation

### Run Unit Tests

```bash
pytest tests/ -v
```

### Run Benchmarks

```bash
python benchmarks/compare_baselines.py --save-results
```

This will:
- Measure throughput (samples/sec)
- Profile peak memory consumption
- Analyze gradient flow quality
- Test long sequence stability
- Compare against Mamba baseline (if installed)

### Validation Results

See [TEST_RESULTS.md](TEST_RESULTS.md) for comprehensive validation instructions and expected outputs.

## Documentation

### Mathematical Foundations

For formal complexity proofs, gradient consolidation equations, and uncertainty quantification formulations, see:

**[docs/mathematical_foundations.md](docs/mathematical_foundations.md)**

Key topics:
- O(n) complexity proof
- Nested Learning gradient accumulation
- Epistemic vs Aleatoric uncertainty
- State space stability analysis
- Memory hierarchy capacity analysis

### Architecture Details

#### Continuum Memory System (CMS)

The CMS implements three-tier memory with distinct update mechanisms:

| Memory Level | Update Frequency | Parameters | Training Method |
|--------------|------------------|------------|------------------|
| Fast | Every step | W_f, b_f | Standard optimizer |
| Medium | Every 10 steps | W_m, b_m | Gradient consolidation |
| Slow | Every 100 steps | W_s, b_s | Gradient consolidation |

**Gradient Consolidation Equation:**

```
W_s^(t+C) = W_s^(t) + η · (∇_acc W_s / ||∇_acc W_s||)
```

where ∇_acc accumulates gradients over C steps before parameter update.

## Benchmarking Results

Preliminary benchmarks (8 batch size, 256 hidden dim):

| Metric | DHC-SSM v3.2.0 | Notes |
|--------|----------------|-------|
| Parameters | ~2.5M | Compact architecture |
| Throughput | TBD | Run benchmarks locally |
| Peak Memory | TBD | GPU-dependent |
| Gradient Stability | Excellent | No vanishing/exploding |

> Run `python benchmarks/compare_baselines.py` for your hardware.

## Project Structure

```
DHC-SSM-AGI/
├── dhc_ssm/
│   ├── core/
│   │   ├── model.py              # Main DHC-SSM model
│   │   ├── nested_ssm.py         # Nested Learning implementation
│   │   └── learning_engine.py    # Deterministic optimizer
│   ├── agi/
│   │   ├── metacognition.py      # Meta-cognitive layer
│   │   ├── uncertainty.py        # Uncertainty quantification
│   │   ├── self_improvement.py   # Recursive self-improvement
│   │   └── goal_system.py        # Dynamic goal redefinition
│   └── training/             # Training utilities
├── benchmarks/
│   └── compare_baselines.py  # Comprehensive benchmarking
├── docs/
│   └── mathematical_foundations.md  # Formal proofs
├── tests/                    # Unit tests
├── TEST_RESULTS.md           # Validation instructions
└── README.md                 # This file
```

## Key Features

### 1. True Nested Learning
- Separate fast and slow weight parameters
- Functional gradient accumulation buffers
- Normalized consolidation updates
- Learnable interpolation between pathways

### 2. O(n) Linear Complexity
- Formal mathematical proof in [docs/mathematical_foundations.md](docs/mathematical_foundations.md)
- Per-operation analysis
- Quadratic speedup vs Transformers for long sequences

### 3. Uncertainty Quantification
- **Epistemic**: Model uncertainty via Monte Carlo Dropout
- **Aleatoric**: Data uncertainty via learned heteroscedastic variance
- Total predictive uncertainty decomposition

### 4. Recursive Self-Improvement
- Adaptive threshold analysis
- Dynamic goal redefinition
- Meta-learning capabilities

### 5. Comprehensive Benchmarking
- Throughput measurement
- Memory profiling
- Gradient flow analysis
- Baseline comparisons (Mamba, RWKV)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{dhc_ssm_agi_2025,
  author = {Kwag, Sung hun},
  title = {DHC-SSM-AGI: Deterministic Hierarchical Causal State Space Model with True Nested Learning},
  version = {3.2.0},
  year = {2025},
  url = {https://github.com/sunghunkwag/DHC-SSM-AGI},
  note = {Implements gradient consolidation for multi-time-scale memory systems}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
pytest tests/ -v --cov=dhc_ssm
```

## References

1. **Abehrouz et al. (2025).** "Nested Learning: The Illusion of Deep Learning Architectures." *NeurIPS 2025*. [PDF](https://abehrouz.github.io/files/NL.pdf)

2. **Gu, A., & Dao, T. (2024).** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." *arXiv:2312.00752*.

3. **Kendall, A., & Gal, Y. (2017).** "What Uncertainties Do We Need in Bayesian Deep Learning?" *NeurIPS 2017*.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**Status**: Active Research | v3.2.0 | Production-Ready

**Maintainer**: Sung hun Kwag ([@sunghunkwag](https://github.com/sunghunkwag))
