"""Comprehensive tests for DHC-SSM v4.0 features.

Tests all new components:
1. Auto-consolidating gradients
2. Gated SSM with ICL
3. Multi-resolution wavelet SSM
4. Self-improving architecture
"""

import pytest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '../')

from dhc_ssm.core.auto_consolidating_ssm import AutoConsolidatingSSM
from dhc_ssm.core.gated_ssm import GatedS6Layer, MultiLayerGatedSSM
from dhc_ssm.core.multi_resolution_ssm import MultiResolutionSSM, WaveletDecomposition
from dhc_ssm.agi.self_improving_architecture import SelfImprovingArchitecture, RecursiveSelfImprovement


class TestAutoConsolidatingSSM:
    """Test automatic gradient consolidation."""
    
    def test_initialization(self):
        """Test model initializes correctly."""
        model = AutoConsolidatingSSM(
            hidden_dim=64,
            state_dim=32,
            medium_consolidation_freq=10,
            slow_consolidation_freq=100,
        )
        
        assert model.hidden_dim == 64
        assert model.state_dim == 32
        assert model.medium_freq == 10
        assert model.slow_freq == 100
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = AutoConsolidatingSSM(hidden_dim=64, state_dim=32)
        x = torch.randn(8, 64)  # (batch, hidden_dim)
        
        output = model(x)
        
        assert output.shape == (8, 32)  # (batch, state_dim)
    
    def test_gradient_hooks_registered(self):
        """Test that gradient hooks are registered for slow/medium params."""
        model = AutoConsolidatingSSM(hidden_dim=64, state_dim=32)
        
        hooks_registered = 0
        for name, param in model.named_parameters():
            if hasattr(param, '_backward_hooks') and len(param._backward_hooks) > 0:
                hooks_registered += 1
        
        # Should have hooks for medium and slow parameters
        assert hooks_registered > 0
    
    def test_no_manual_accumulation_needed(self):
        """Test that manual accumulate_gradients() is not needed."""
        model = AutoConsolidatingSSM(hidden_dim=64, state_dim=32)
        optimizer = torch.optim.Adam(model.parameters())
        
        x = torch.randn(8, 64)
        target = torch.randn(8, 32)
        
        # Training loop without manual accumulation
        for _ in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()  # No accumulate_gradients() call!
        
        # Should complete without errors
        assert True


class TestGatedSSM:
    """Test gated SSM for in-context learning."""
    
    def test_gated_layer_initialization(self):
        """Test gated S6 layer initializes."""
        layer = GatedS6Layer(hidden_dim=128, state_dim=64, dt_rank=32)
        
        assert layer.hidden_dim == 128
        assert layer.state_dim == 64
    
    def test_gated_forward_with_sequence(self):
        """Test forward pass with sequence input."""
        layer = GatedS6Layer(hidden_dim=128, state_dim=64)
        x = torch.randn(4, 16, 128)  # (batch, seq_len, hidden_dim)
        
        output, state = layer(x)
        
        assert output.shape == (4, 16, 128)
        assert state.shape == (4, 64)
    
    def test_input_output_gating_works(self):
        """Test that gating actually modulates signal."""
        layer = GatedS6Layer(hidden_dim=64, state_dim=32)
        
        # Two different inputs
        x1 = torch.randn(2, 10, 64)
        x2 = torch.randn(2, 10, 64)
        
        out1, _ = layer(x1)
        out2, _ = layer(x2)
        
        # Outputs should be different (gating is input-dependent)
        assert not torch.allclose(out1, out2)
    
    def test_multilayer_gated_ssm(self):
        """Test stacked gated SSM layers."""
        model = MultiLayerGatedSSM(
            num_layers=3,
            hidden_dim=128,
            state_dim=64
        )
        
        x = torch.randn(4, 16, 128)
        output, states = model(x)
        
        assert output.shape == (4, 16, 128)
        assert len(states) == 3  # 3 layers


class TestMultiResolutionSSM:
    """Test multi-resolution wavelet SSM."""
    
    def test_wavelet_decomposition(self):
        """Test wavelet decomposition produces 3 scales."""
        decomp = WaveletDecomposition(wavelet='db4', level=2)
        x = torch.randn(4, 128, 3)  # (batch, seq_len, channels)
        
        high, mid, low = decomp(x)
        
        # Should return 3 different frequency bands
        assert high.shape[0] == 4  # Same batch size
        assert mid.shape[0] == 4
        assert low.shape[0] == 4
    
    def test_multi_resolution_forward(self):
        """Test full multi-resolution forward pass."""
        model = MultiResolutionSSM(
            hidden_dim=64,
            state_dim=32,
            wavelet='db4'
        )
        
        x = torch.randn(4, 128, 64)  # (batch, seq_len, hidden_dim)
        output = model(x)
        
        # Output should match input dimensions
        assert output.shape == (4, 128, 64)
    
    def test_scale_adaptive_gating(self):
        """Test that scale gating weights are learned."""
        model = MultiResolutionSSM(hidden_dim=64, state_dim=32)
        
        # Check that scale_attention exists and has parameters
        assert hasattr(model, 'scale_attention')
        params = list(model.scale_attention.parameters())
        assert len(params) > 0


class TestSelfImprovingArchitecture:
    """Test self-improvement system."""
    
    def test_initialization(self):
        """Test self-improving wrapper initializes."""
        base_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        improver = SelfImprovingArchitecture(
            base_model=base_model,
            improvement_threshold=0.05,
            confidence_level=0.95
        )
        
        assert improver.improvement_threshold == 0.05
        assert improver.confidence_level == 0.95
    
    def test_candidate_generation(self):
        """Test that modification candidates are generated."""
        base_model = nn.Linear(64, 10)
        improver = SelfImprovingArchitecture(base_model)
        
        candidates = improver.generate_candidate_modifications()
        
        assert len(candidates) > 0
        assert all(hasattr(c, 'description') for c in candidates)
    
    def test_performance_measurement(self):
        """Test baseline performance measurement."""
        base_model = nn.Linear(64, 10)
        improver = SelfImprovingArchitecture(base_model)
        
        # Create dummy validation data
        dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 64),
            torch.randint(0, 10, (100,))
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        performance = improver._measure_performance(loader)
        
        assert isinstance(performance, float)
        assert performance > 0
    
    def test_recursive_improvement(self):
        """Test recursive self-improvement loop."""
        base_model = nn.Linear(64, 10)
        recursive_improver = RecursiveSelfImprovement(
            base_model=base_model,
            max_iterations=3,
            convergence_threshold=0.01
        )
        
        # Should initialize without errors
        assert recursive_improver.max_iterations == 3


class TestIntegration:
    """Integration tests combining multiple v4.0 features."""
    
    def test_auto_consolidating_with_gated_ssm(self):
        """Test combining auto-consolidation with gated SSM."""
        # Create hybrid model
        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.auto_cons = AutoConsolidatingSSM(64, 32)
                self.gated = GatedS6Layer(32, 16)
            
            def forward(self, x):
                x = self.auto_cons(x).unsqueeze(1)  # Add seq dim
                x, _ = self.gated(x)
                return x.squeeze(1)
        
        model = HybridModel()
        x = torch.randn(8, 64)
        output = model(x)
        
        assert output.shape == (8, 16)
    
    def test_full_v4_pipeline(self):
        """Test complete v4.0 pipeline."""
        # Create full pipeline
        class V4Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.multi_res = MultiResolutionSSM(64, 32)
                self.gated = MultiLayerGatedSSM(2, 64, 32)
            
            def forward(self, x):
                x = self.multi_res(x)
                x, _ = self.gated(x)
                return x.mean(dim=1)  # Pool over sequence
        
        model = V4Model()
        x = torch.randn(4, 128, 64)
        output = model(x)
        
        assert output.shape == (4, 64)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
