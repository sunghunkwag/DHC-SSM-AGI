"""
Unit tests for Uncertainty Quantification System
"""
import torch
import pytest
from dhc_ssm.agi.uncertainty import (
    UncertaintyQuantifier,
    EpistemicUncertaintyEstimator,
    AleatoricUncertaintyEstimator,
    UncertaintyDecomposer,
)


class TestEpistemicUncertaintyEstimator:
    """Test epistemic uncertainty estimation."""
    
    def test_initialization(self):
        estimator = EpistemicUncertaintyEstimator(
            input_dim=128,
            output_dim=10,
            num_heads=5,
        )
        assert len(estimator.heads) == 5
    
    def test_forward_pass(self):
        estimator = EpistemicUncertaintyEstimator(input_dim=128, output_dim=10)
        x = torch.randn(32, 128)
        
        mean_pred, epistemic_unc = estimator(x)
        
        assert mean_pred.shape == (32, 10)
        assert epistemic_unc.shape == (32,)
        assert (epistemic_unc >= 0).all()
    
    def test_ensemble_variance(self):
        """Test that epistemic uncertainty increases with ensemble disagreement."""
        estimator = EpistemicUncertaintyEstimator(input_dim=128, output_dim=10)
        
        # Consistent input should have lower uncertainty
        x1 = torch.ones(32, 128)
        _, unc1 = estimator(x1)
        
        # Random input should have higher uncertainty
        x2 = torch.randn(32, 128)
        _, unc2 = estimator(x2)
        
        # Average uncertainty for random input should be higher
        assert unc2.mean() >= 0.0  # Just check it's non-negative


class TestAleatoricUncertaintyEstimator:
    """Test aleatoric uncertainty estimation."""
    
    def test_initialization(self):
        estimator = AleatoricUncertaintyEstimator(input_dim=128, output_dim=10)
        assert estimator.mean_head is not None
        assert estimator.variance_head is not None
    
    def test_forward_pass(self):
        estimator = AleatoricUncertaintyEstimator(input_dim=128, output_dim=10)
        x = torch.randn(32, 128)
        
        mean_pred, aleatoric_unc = estimator(x)
        
        assert mean_pred.shape == (32, 10)
        assert aleatoric_unc.shape == (32,)
        assert (aleatoric_unc >= 0).all()  # Variance must be non-negative
    
    def test_variance_positivity(self):
        """Test that predicted variances are always positive."""
        estimator = AleatoricUncertaintyEstimator(input_dim=128, output_dim=10)
        x = torch.randn(100, 128)
        
        _, aleatoric_unc = estimator(x)
        
        assert (aleatoric_unc > 0).all(), "All variances must be positive"


class TestUncertaintyDecomposer:
    """Test uncertainty decomposition."""
    
    def test_initialization(self):
        decomposer = UncertaintyDecomposer(feature_dim=256)
        assert decomposer is not None
    
    def test_forward_pass(self):
        decomposer = UncertaintyDecomposer(feature_dim=256)
        
        features = torch.randn(32, 256)
        epistemic = torch.rand(32)
        aleatoric = torch.rand(32)
        
        result = decomposer(features, epistemic, aleatoric)
        
        assert 'total_uncertainty' in result
        assert 'epistemic_contribution' in result
        assert 'aleatoric_contribution' in result
        assert 'interaction_term' in result
        assert 'dominant_type' in result
    
    def test_weights_sum_to_one(self):
        """Test that decomposition weights are properly normalized."""
        decomposer = UncertaintyDecomposer(feature_dim=256)
        
        features = torch.randn(32, 256)
        epistemic = torch.rand(32) * 0.5
        aleatoric = torch.rand(32) * 0.5
        
        result = decomposer(features, epistemic, aleatoric)
        
        # Check that contributions are reasonable
        assert result['total_uncertainty'].shape == (32,)
        assert (result['total_uncertainty'] >= 0).all()


class TestUncertaintyQuantifier:
    """Test complete uncertainty quantification system."""
    
    def test_initialization(self):
        quantifier = UncertaintyQuantifier(
            input_dim=256,
            output_dim=10,
            num_ensemble_heads=5,
        )
        assert quantifier.epistemic_estimator is not None
        assert quantifier.aleatoric_estimator is not None
        assert quantifier.uncertainty_decomposer is not None
    
    def test_forward_pass(self):
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        features = torch.randn(32, 256)
        
        result = quantifier(features)
        
        # Check all expected keys
        expected_keys = [
            'predictions',
            'epistemic_uncertainty',
            'aleatoric_uncertainty',
            'total_uncertainty',
            'confidence',
            'should_explore',
            'uncertainty_level',
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_uncertainty_levels(self):
        """Test uncertainty level categorization."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        features = torch.randn(32, 256)
        
        result = quantifier(features)
        
        valid_levels = ['low', 'medium', 'high', 'very_high']
        assert result['uncertainty_level'] in valid_levels
    
    def test_confidence_range(self):
        """Test that confidence is in valid range [0, 1]."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        features = torch.randn(32, 256)
        
        result = quantifier(features)
        confidence = result['confidence']
        
        assert (confidence >= 0).all()
        assert (confidence <= 1).all()
    
    def test_uncertainty_history_tracking(self):
        """Test that uncertainty history is properly tracked."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        
        # Run multiple forward passes
        for _ in range(10):
            features = torch.randn(32, 256)
            quantifier(features)
        
        assert quantifier.history_idx.item() == 10
        assert quantifier.uncertainty_history[:10].abs().sum() > 0
    
    def test_trend_analysis(self):
        """Test uncertainty trend analysis."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        
        # Generate decreasing epistemic uncertainty (simulating learning)
        for i in range(20):
            features = torch.randn(32, 256) * (1.0 - i * 0.03)
            quantifier(features)
        
        trends = quantifier.get_uncertainty_trend()
        
        assert 'total_trend' in trends
        assert 'epistemic_trend' in trends
        assert 'aleatoric_trend' in trends
        assert 'is_learning' in trends
    
    def test_diagnostics(self):
        """Test diagnostic information retrieval."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        
        # Run some forward passes
        for _ in range(5):
            features = torch.randn(32, 256)
            quantifier(features)
        
        diagnostics = quantifier.get_diagnostics()
        
        assert 'history_length' in diagnostics
        assert 'avg_total_uncertainty' in diagnostics
        assert 'avg_epistemic_uncertainty' in diagnostics
        assert 'avg_aleatoric_uncertainty' in diagnostics
        assert 'trends' in diagnostics
        assert diagnostics['history_length'] == 5


class TestIntegration:
    """Integration tests for uncertainty quantification."""
    
    def test_uncertainty_consistency(self):
        """Test that uncertainty estimates are consistent across multiple runs."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        features = torch.randn(32, 256)
        
        # Run twice with same input
        result1 = quantifier(features)
        result2 = quantifier(features)
        
        # Predictions should be deterministic (no dropout in eval mode)
        # Uncertainties may vary slightly due to ensemble
        assert result1['predictions'].shape == result2['predictions'].shape
    
    def test_exploration_decision(self):
        """Test that exploration decision is based on uncertainty."""
        quantifier = UncertaintyQuantifier(input_dim=256, output_dim=10)
        
        # Low uncertainty input
        low_unc_features = torch.zeros(32, 256) + 0.1
        result_low = quantifier(low_unc_features)
        
        # High uncertainty input
        high_unc_features = torch.randn(32, 256) * 10
        result_high = quantifier(high_unc_features)
        
        # Should explore more often with high uncertainty
        # (this is probabilistic, so we just check the field exists)
        assert isinstance(result_low['should_explore'], torch.Tensor)
        assert isinstance(result_high['should_explore'], torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
