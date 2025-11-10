# DHC-SSM-AGI (Deterministic Hierarchical Causal State Space Model - AGI Edition)

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Robust, fully type-safe research architecture featuring recursive self-improvement, Nested Learning (Google Research, NeurIPS 2025), adaptive threshold analysis, real uncertainty quantification, and extensive modular test suite.

## Architecture Overview

```
───── Input (Batch, 3, 32, 32)
│
[Spatial Encoder]
│
[Nested Temporal SSM (multi-time-scale)]
│   ├─ Fast Memory (every step)
│   ├─ Medium Memory (every 10 steps)
│   └─ Slow Memory (every 100 steps)
│
[Strategic Reasoner (GNN/Graph)]
│
───── Key Features:
• Nested Learning with CMS
• Meta-Cognitive Layer
• RSI with Adaptive Thresholds
• Real Uncertainty Quantification (Epistemic + Aleatoric)
• Dynamic Goals
• Deep Momentum Optimizers
• Integrated Diagnostics
│
───── Output: Predictions, Diagnostics, Learning Self-Metrics
```

## Manual Testing

To run all tests locally:

```bash
pytest tests/ -v
```

Coverage analysis and all unit tests are available, but results are not published online or automated.

## Documentation & Contribution
- All documentation for Nested Learning, design, and testing is included in the repo.
- Experimental cycles and API/usage remain unchanged; see documentation files for module usage.

## Citation
If you use this work in your research, please cite:

```bibtex
@software{dhc_ssm_agi_2025,
  author = {Kwag, Sung hun},
  title = {DHC-SSM-AGI: Deterministic Hierarchical Causal State Space Model with Nested Learning},
  year = {2025},
  url = {https://github.com/sunghunkwag/DHC-SSM-AGI}
}
```

**Status**: Active Research | CI/CD: Disabled | Manual Testing Only
