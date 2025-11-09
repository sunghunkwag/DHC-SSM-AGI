# DHC-SSM-AGI (Deterministic Hierarchical Causal State Space Model - AGI Edition)

![CI/CD](https://github.com/sunghunkwag/DHC-SSM-AGI/workflows/CI/CD%20Pipeline/badge.svg)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Robust, fully type-safe AGI architecture featuring recursive self-improvement (RSI) with adaptive threshold analysis, real uncertainty quantification, and validated with PyTorch-based reproducible tests.

## Architecture Overview

```
───── Input (Batch, 3, 32, 32)
│
[Spatial Encoder]
│
[Temporal SSM (linear, parallel)]
│
[Strategic Reasoner (GNN/Graph)]
│
───── AGI Enhancements:
• Meta-Cognitive Layer
• RSI with Adaptive Thresholds
• Real Uncertainty Quantification (Epistemic + Aleatoric)
• Dynamic Goals
• Integrated Diagnostics
│
───── Output: Predictions, Diagnostics, Learning Self-Metrics
```

## What's New in v3.1.0

### 🎯 Critical Improvements

- **✅ Type Safety**: Fixed all `Dict[str, any]` → `Dict[str, Any]` type hints
- **✅ Real Uncertainty**: Integrated actual `UncertaintyQuantifier` instead of dummy generation
- **✅ Adaptive Thresholds**: Removed hardcoded magic numbers, implemented statistical validation
- **✅ Enhanced Testing**: Added comprehensive unit tests for uncertainty quantification
- **✅ CI/CD Pipeline**: Automated testing with GitHub Actions
- **✅ Realistic Dependencies**: Updated PyTorch requirements to actual available versions

### 🔬 Technical Enhancements

- **Uncertainty Quantification**: Proper ensemble-based epistemic + variance-based aleatoric estimation
- **Feature Extraction**: Smart feature extraction with automatic dimension adaptation
- **Model Improvement**: Multiple strategies (capacity expansion, architecture refinement)
- **Error Handling**: Robust error handling with graceful degradation
- **Logging**: Comprehensive logging for debugging and monitoring

## Quick Start

### Requirements

- Python 3.11+
- PyTorch >=2.0.0
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/DHC-SSM-AGI.git
cd DHC-SSM-AGI

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_uncertainty.py -v
pytest tests/test_agi_threshold.py -v

# Run with coverage
pytest tests/ -v --cov=dhc_ssm --cov-report=html
```

### Running the AGI Improvement Cycle

```bash
export PYTHONPATH=$PWD
python tests/test_agi_threshold.py
```

## Major Modules

- `dhc_ssm/agi/threshold_analyzer.py`: Adaptive RSI threshold with statistical validation
- `dhc_ssm/agi/self_improvement.py`: Strategy/hypothesis generation and validation
- `dhc_ssm/agi/self_improvement_executor.py`: Model improvement with real uncertainty integration
- `dhc_ssm/agi/uncertainty.py`: Comprehensive epistemic + aleatoric uncertainty quantification
- `dhc_ssm/agi/metacognition.py`: Meta-cognitive layer for self-awareness
- `dhc_ssm/agi/goal_system.py`: Dynamic goal redefinition system
- `dhc_ssm/agi/meta_learning.py`: Meta-learning engine for rapid adaptation

## Usage Example

### Basic Self-Improvement Cycle

```python
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
from dhc_ssm.agi.self_improvement_executor import SelfImprovementExecutor
from dhc_ssm.agi.uncertainty import UncertaintyQuantifier
import torch
import torch.nn as nn
from typing import Tuple

# Define a simple model
class DummyModel(nn.Module):
    def __init__(self, input_dim: int = 3072, output_dim: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1)
        return self.linear(x)

# Create model and validation data
model = DummyModel()
val_data: Tuple[torch.Tensor, torch.Tensor] = (
    torch.randn(32, 3, 32, 32),
    torch.randint(0, 10, (32,))
)

# Initialize components
threshold_analyzer = RSIThresholdAnalyzer(
    history_length=100,
    convergence_volatility_threshold=0.03,
)

uncertainty_quantifier = UncertaintyQuantifier(
    input_dim=256,
    output_dim=10,
    num_ensemble_heads=5,
)

# Create executor
executor = SelfImprovementExecutor(
    base_model=model,
    validation_data=val_data,
    threshold_analyzer=threshold_analyzer,
    uncertainty_quantifier=uncertainty_quantifier,
    feature_dim=256,
)

# Execute improvement cycle
result = executor.execute_cycle()

print(f"Action: {result['action']}")
print(f"Threshold Status: {result['threshold'].status.value}")
print(f"Gamma: {result['threshold'].gamma:.3f}")
print(f"Gamma*: {result['threshold'].gamma_star:.3f}")

# Get diagnostics
diagnostics = executor.diagnostics()
print("\nDiagnostics:")
for key, value in diagnostics.items():
    print(f"  {key}: {value}")
```

### Advanced: Using AGI-Enhanced Model

```python
from dhc_ssm.agi.agi_model import AGIEnhancedDHCSSM
from dhc_ssm.core.config import DHCSSMConfig

# Create configuration
config = DHCSSMConfig(
    input_channels=3,
    hidden_dim=64,
    ssm_state_dim=64,
    output_dim=10,
)

# Create AGI-enhanced model
model = AGIEnhancedDHCSSM(config)

# Forward pass with AGI features
x = torch.randn(32, 3, 32, 32)
y = torch.randint(0, 10, (32,))

output = model(x, targets=y, enable_agi=True)

print(f"Predictions shape: {output['predictions'].shape}")
print(f"Uncertainty level: {output['uncertainty']['level']}")
print(f"Should adapt: {output['should_adapt']}")
print(f"Meta decision: {output['metacognition']['meta_decision']}")
```

## Development

### Code Quality

```bash
# Format code
black dhc_ssm/ tests/

# Lint code
flake8 dhc_ssm/ tests/

# Type checking
mypy dhc_ssm/
```

### Running CI/CD Locally

```bash
# Install development dependencies
pip install black flake8 mypy pytest pytest-cov

# Run all checks (like CI/CD does)
black --check dhc_ssm/ tests/
flake8 dhc_ssm/ tests/ --max-line-length=127
pytest tests/ -v --cov=dhc_ssm
```

## Diagnostics & Benchmarking

See full diagnostics/benchmarks in `tests/test_agi_threshold.py` and `tests/test_comprehensive.py`.

Benchmark results are stored in `tests/benchmark_results_v3_0.json`.

## Limitations & Roadmap

See [limitations_and_roadmap.md](limitations_and_roadmap.md) for details.

### Known Limitations

- Model improvement strategies are demonstration-level (need extension for production)
- Threshold analyzer proven robust in controlled tests but needs validation on larger models
- Uncertainty quantification requires proper feature extraction per model architecture

### Upcoming Features

- [ ] Advanced model improvement strategies (SSM-specific, GNN-specific)
- [ ] Multi-task learning support
- [ ] Distributed training support
- [ ] More comprehensive benchmarks on real datasets
- [ ] Interactive visualization dashboard
- [ ] API documentation with Sphinx

## Troubleshooting

### Import Errors

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=$PWD

# Or install in development mode
pip install -e .
```

### PyTorch Version Issues

```bash
# Install compatible PyTorch version
pip install torch>=2.0.0,<3.0.0 torchvision>=0.15.0
```

### Test Failures

```bash
# Run tests with verbose output
pytest tests/ -vv --tb=short

# Run specific test
pytest tests/test_uncertainty.py::TestUncertaintyQuantifier::test_forward_pass -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License ©2025 Sung hun kwag / DHC-SSM-AGI contributors

## Citation

If you use this work in your research, please cite:

```bibtex
@software{dhc_ssm_agi_2025,
  author = {Kwag, Sung hun},
  title = {DHC-SSM-AGI: Deterministic Hierarchical Causal State Space Model with AGI Capabilities},
  year = {2025},
  url = {https://github.com/sunghunkwag/DHC-SSM-AGI}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Research community for foundational work on state space models, uncertainty quantification, and meta-learning

---

**Status**: Active Development | **Version**: 3.1.0 | **Last Updated**: November 2025
