"""
Pytest Configuration and Shared Fixtures

Provides reusable test fixtures with appropriate scoping
for efficient and isolated testing.
"""

import pytest
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple


@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    Session-scoped fixture for device selection.
    
    Returns:
        torch.device: CUDA if available, else CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    return device


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Path:
    """
    Session-scoped fixture for test data directory.
    
    Args:
        tmp_path_factory: pytest fixture for creating temporary directories
    
    Returns:
        Path: Temporary directory for test data
    """
    data_dir = tmp_path_factory.mktemp("test_data")
    return data_dir


@pytest.fixture(scope="module")
def sample_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Module-scoped fixture for sample batch data.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sample images (32x3x32x32) and labels (32,)
    """
    torch.manual_seed(42)  # Reproducible test data
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    return images, labels


@pytest.fixture(scope="module")
def small_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Module-scoped fixture for small batch (for quick tests).
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Small batch images (4x3x32x32) and labels (4,)
    """
    torch.manual_seed(42)
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    return images, labels


@pytest.fixture(scope="function")
def dhc_config():
    """
    Function-scoped fixture for DHC-SSM configuration.
    
    Returns:
        DHCSSMConfig: Fresh config instance for each test
    """
    from dhc_ssm.core.model import DHCSSMConfig
    
    config = DHCSSMConfig(
        input_channels=3,
        hidden_dim=64,
        ssm_state_dim=64,
        output_dim=10,
    )
    return config


@pytest.fixture(scope="function")
def dhc_model(dhc_config, device):
    """
    Function-scoped fixture for DHC-SSM model.
    
    Args:
        dhc_config: Config fixture
        device: Device fixture
    
    Returns:
        DHCSSMModel: Fresh model instance on appropriate device
    """
    from dhc_ssm.core.model import DHCSSMModel
    
    model = DHCSSMModel(dhc_config).to(device)
    return model


@pytest.fixture(scope="function")
def uncertainty_quantifier():
    """
    Function-scoped fixture for uncertainty quantifier.
    
    Returns:
        UncertaintyQuantifier: Fresh uncertainty quantifier instance
    """
    from dhc_ssm.agi.uncertainty import UncertaintyQuantifier
    
    quantifier = UncertaintyQuantifier(
        input_dim=256,
        output_dim=10,
        num_ensemble_heads=5,
    )
    return quantifier


@pytest.fixture(scope="function")
def threshold_analyzer():
    """
    Function-scoped fixture for threshold analyzer.
    
    Returns:
        RSIThresholdAnalyzer: Fresh threshold analyzer instance
    """
    from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
    
    analyzer = RSIThresholdAnalyzer(
        history_length=100,
        convergence_volatility_threshold=0.03,
    )
    return analyzer


@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Auto-use fixture that resets random seeds before each test.
    
    Ensures reproducible test results.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    yield


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Session-scoped fixture for test environment setup.
    
    Configures PyTorch for deterministic behavior in tests.
    """
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("\n=== Test Environment Setup ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    yield
    
    print("\n=== Test Environment Teardown ===")
    # Cleanup if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
