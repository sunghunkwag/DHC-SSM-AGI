"""Version information for DHC-SSM Architecture."""

__version__ = "3.1.0"
__author__ = "Sung hun kwag"
__license__ = "MIT"
__description__ = (
    "Deterministic Hierarchical Causal State Space Model - AGI Edition v3.1: "
    "Recursive Self-Improvement with Adaptive Thresholds and Real Uncertainty Quantification"
)

VERSION_INFO = {
    "major": 3,
    "minor": 1,
    "patch": 0,
    "release": "stable",
    "build": "2025.11.10"
}

def get_version() -> str:
    """Get the version string."""
    return __version__

def get_version_info() -> dict:
    """Get detailed version information."""
    return VERSION_INFO
