# DHC-SSM-AGI (Deterministic Hierarchical Causal State Space Model - AGI Edition)

Robust, fully type-safe AGI architecture featuring recursive self-improvement (RSI) with threshold/convergence analysis, validated with PyTorch-based reproducible tests.

## Architecture Overview

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
    • RSI / Hypothesis-Validation Loop
    • Dynamic Goals
    • Uncertainty Quantification
    • Integrated Diagnostics
        │
───── Output: Predictions, Diagnostics, Learning Self-Metrics

## Quick Start

### Requirements
- Python 3.11+
- PyTorch >=2.0

### Installation
```bash
git clone https://github.com/sunghunkwag/DHC-SSM-AGI.git
cd DHC-SSM-AGI
pip install -e .
```

### Running the Test Suite
```bash
export PYTHONPATH=$PWD
python tests/test_agi_threshold.py
```

## Major Modules

- `dhc_ssm/agi/threshold_analyzer.py`: RSI threshold, convergence, diagnostics
- `dhc_ssm/agi/self_improvement.py`: Strategy/hypothesis, experiment validation, state
- `dhc_ssm/agi/self_improvement_executor.py`: Model improvement, rollback, threshold-validation
- `tests/test_agi_threshold.py`: End-to-end logic, robust AGI improvement cycle / failure detection

## Usage Example

```python
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
from dhc_ssm.agi.self_improvement_executor import SelfImprovementExecutor
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3072, 10)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)

model = DummyModel()
val_data = (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
executor = SelfImprovementExecutor(model, val_data)
result = executor.execute_cycle()
print(result['action'], result['threshold'].diagnostics() if hasattr(result['threshold'], 'diagnostics') else '')
```

## Diagnostics & Benchmarking

Run full diagnostics/benchmarks, see [tests/test_agi_threshold.py](tests/test_agi_threshold.py)

## Limitations & Roadmap
See [limitations_and_roadmap.md](limitations_and_roadmap.md).

- Model improvement is basic (demo Linear expansion); applying to real SSM/graph models needs extension
- Threshold/analyzer logic proven robust in controlled test but should be battle-tested on larger models
- Documentation and user-level examples being improved iteratively

----

## License
MIT License ©2025 Sung hun kwag / DHC-SSM-AGI contributors
