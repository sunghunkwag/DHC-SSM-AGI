"""DHC-SSM: Deterministic Hierarchical Causal State Space Model
Enhanced Architecture v4.0

A production-ready deep learning architecture combining:
- Auto-consolidating multi-timescale gradient learning
- Gated SSMs with in-context learning capabilities
- Multi-resolution wavelet processing
- Self-improving architecture with provable guarantees
- O(n) linear time complexity (vs O(n²) for transformers)

New in v4.0:
- Automatic gradient consolidation via PyTorch hooks
- Multiplicative input/output gating for ICL
- True multi-scale wavelet decomposition
- Darwin Gödel Machine self-improvement

Author: Sung hun Kwag
License: MIT
Version: 4.0.0
"""

# Version info
__version__ = "4.0.0"
__author__ = "Sung hun Kwag"
__license__ = "MIT"
__description__ = "Auto-Consolidating Gated Multi-Resolution SSM with Self-Improvement"

# Core v3.x components (backward compatibility)
try:
    from dhc_ssm.core.model import (
        DHCSSMModel,
        DHCSSMConfig,
        SpatialEncoder,
        TemporalSSM,
    )
    _v3_available = True
except ImportError:
    _v3_available = False
    DHCSSMModel = None
    DHCSSMConfig = None
    SpatialEncoder = None
    TemporalSSM = None

# V4.0 Core Components - Auto-Consolidating SSM
from dhc_ssm.core.auto_consolidating_ssm import AutoConsolidatingSSM

# V4.0 Gated SSM Components
from dhc_ssm.core.gated_ssm import (
    GatedS6Layer,
    MultiLayerGatedSSM,
)

# V4.0 Multi-Resolution Components
from dhc_ssm.core.multi_resolution_ssm import (
    MultiResolutionSSM,
    WaveletDecomposition,
    ScaleSpecificSSM,
)

# V4.0 Self-Improving Architecture
from dhc_ssm.agi.self_improving_architecture import (
    SelfImprovingArchitecture,
    RecursiveSelfImprovement,
    ArchitectureCandidate,
)

# Utilities
try:
    from dhc_ssm.utils.config import (
        get_default_config,
        get_small_config,
        get_large_config,
        get_debug_config
    )
    _utils_available = True
except ImportError:
    _utils_available = False

# Training
try:
    from dhc_ssm.training.trainer import Trainer
    _trainer_available = True
except ImportError:
    _trainer_available = False
    Trainer = None

# Export list
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    
    # V4.0 Core - Auto-Consolidating SSM
    "AutoConsolidatingSSM",
    
    # V4.0 Gated SSM
    "GatedS6Layer",
    "MultiLayerGatedSSM",
    
    # V4.0 Multi-Resolution
    "MultiResolutionSSM",
    "WaveletDecomposition",
    "ScaleSpecificSSM",
    
    # V4.0 Self-Improvement
    "SelfImprovingArchitecture",
    "RecursiveSelfImprovement",
    "ArchitectureCandidate",
]

# Add v3.x components if available (backward compatibility)
if _v3_available:
    __all__.extend([
        "DHCSSMModel",
        "DHCSSMConfig",
        "SpatialEncoder",
        "TemporalSSM",
    ])

if _utils_available:
    __all__.extend([
        "get_default_config",
        "get_small_config",
        "get_large_config",
        "get_debug_config",
    ])

if _trainer_available:
    __all__.append("Trainer")


def get_version_info():
    """Get detailed version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": __description__,
        "v3_compatible": _v3_available,
        "features": [
            "Auto-Consolidating Gradients",
            "Gated SSM with ICL",
            "Multi-Resolution Wavelet Processing",
            "Self-Improving Architecture",
            "O(n) Linear Complexity",
        ]
    }


def quick_start_example():
    """Print a quick start example."""
    example = """
DHC-SSM v4.0 Quick Start:

```python
import torch
from dhc_ssm import AutoConsolidatingSSM, GatedS6Layer, MultiResolutionSSM

# 1. Auto-Consolidating SSM (no manual gradient accumulation!)
model = AutoConsolidatingSSM(hidden_dim=256, state_dim=128)
x = torch.randn(8, 256)
output = model(x)  # Gradients consolidate automatically

# 2. Gated SSM for In-Context Learning
gated_model = GatedS6Layer(hidden_dim=512, state_dim=256)
x_seq = torch.randn(4, 16, 512)  # (batch, seq_len, hidden_dim)
output, state = gated_model(x_seq)

# 3. Multi-Resolution Wavelet SSM
multi_res = MultiResolutionSSM(hidden_dim=256, state_dim=128)
x = torch.randn(4, 128, 256)
output = multi_res(x)  # Automatic multi-scale processing
```

For more examples, see: https://github.com/sunghunkwag/DHC-SSM-AGI
    """
    print(example)
