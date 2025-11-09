"""
Uncertainty Quantification System

Implements epistemic and aleatoric uncertainty estimation to help
the system understand what it knows and what it doesn't know.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EpistemicUncertaintyEstimator(nn.Module):
    """
    Estimates epistemic (model) uncertainty using ensemble methods.
    
    Epistemic uncertainty reflects the model's lack of knowledge and
    can be reduced with more training data.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 10,
        num_heads: int = 5,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Ensemble of prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, output_dim),
            )
            for _ in range(num_heads)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictions and epistemic uncertainty.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (mean_prediction, epistemic_uncertainty)
        """
        # Get predictions from all heads
        predictions = torch.stack([head(x) for head in self.heads], dim=0)
        
        # Mean prediction
        mean_pred = predictions.mean(dim=0)
        
        # Epistemic uncertainty (variance across heads)
        epistemic_uncertainty = predictions.var(dim=0).mean(dim=-1)
        
        return mean_pred, epistemic_uncertainty


class AleatoricUncertaintyEstimator(nn.Module):
    """
    Estimates aleatoric (data) uncertainty.
    
    Aleatoric uncertainty reflects inherent noise in the data and
    cannot be reduced with more training data.
    """
    
    def __init__(self, input_dim: int = 256, output_dim: int = 10):
        super().__init__()
        
        # Predict both mean and variance
        self.mean_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softplus(),  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute predictions and aleatoric uncertainty.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (mean_prediction, aleatoric_uncertainty)
        """
        mean = self.mean_head(x)
        variance = self.variance_head(x)
        
        # Aleatoric uncertainty (mean of predicted variances)
        aleatoric_uncertainty = variance.mean(dim=-1)
        
        return mean, aleatoric_uncertainty


class UncertaintyDecomposer(nn.Module):
    """
    Decomposes total uncertainty into epistemic and aleatoric components.
    """
    
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        
        # Decomposition network
        self.decomposer = nn.Sequential(
            nn.Linear(feature_dim + 2, 128),  # features + both uncertainties
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [epistemic_weight, aleatoric_weight, interaction]
            nn.Softmax(dim=-1),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose and analyze uncertainty.
        
        Args:
            features: Input features
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty
            
        Returns:
            Dictionary with decomposition results
        """
        # Combine inputs
        combined = torch.cat([
            features,
            epistemic.unsqueeze(-1),
            aleatoric.unsqueeze(-1),
        ], dim=-1)
        
        # Decompose
        weights = self.decomposer(combined)
        
        total_uncertainty = epistemic + aleatoric
        
        return {
            'total_uncertainty': total_uncertainty,
            'epistemic_contribution': weights[:, 0] * epistemic,
            'aleatoric_contribution': weights[:, 1] * aleatoric,
            'interaction_term': weights[:, 2] * (epistemic * aleatoric).sqrt(),
            'dominant_type': 'epistemic' if epistemic.mean() > aleatoric.mean() else 'aleatoric',
        }


class UncertaintyQuantifier(nn.Module):
    """
    Comprehensive Uncertainty Quantification System
    
    Tracks and quantifies different types of uncertainty to help the
    system understand its own knowledge boundaries and make informed
    decisions about when to explore, exploit, or request more information.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 10,
        num_ensemble_heads: int = 5,
    ):
        super().__init__()
        
        self.epistemic_estimator = EpistemicUncertaintyEstimator(
            input_dim, output_dim, num_ensemble_heads
        )
        self.aleatoric_estimator = AleatoricUncertaintyEstimator(
            input_dim, output_dim
        )
        self.uncertainty_decomposer = UncertaintyDecomposer(input_dim)
        
        # Confidence calibrator
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(input_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Uncertainty history for tracking
        self.register_buffer(
            'uncertainty_history',
            torch.zeros(100, 3)  # [total, epistemic, aleatoric]
        )
        self.register_buffer('history_idx', torch.tensor(0))
    
    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive uncertainty estimates.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary containing:
            - predictions: Model predictions
            - epistemic_uncertainty: Model uncertainty
            - aleatoric_uncertainty: Data uncertainty
            - total_uncertainty: Combined uncertainty
            - confidence: Calibrated confidence score
            - should_explore: Whether to explore (high uncertainty)
        """
        # Get epistemic uncertainty
        epistemic_pred, epistemic_unc = self.epistemic_estimator(features)
        
        # Get aleatoric uncertainty
        aleatoric_pred, aleatoric_unc = self.aleatoric_estimator(features)
        
        # Combine predictions (weighted average)
        combined_pred = 0.6 * epistemic_pred + 0.4 * aleatoric_pred
        
        # Decompose uncertainty
        decomposition = self.uncertainty_decomposer(
            features, epistemic_unc, aleatoric_unc
        )
        
        # Calibrate confidence
        confidence_input = torch.cat([
            features,
            epistemic_unc.unsqueeze(-1),
            aleatoric_unc.unsqueeze(-1),
        ], dim=-1)
        confidence = self.confidence_calibrator(confidence_input).squeeze(-1)
        
        # Update history
        self._update_history(
            decomposition['total_uncertainty'],
            epistemic_unc,
            aleatoric_unc,
        )
        
        # Decision thresholds
        high_uncertainty_threshold = 0.7
        should_explore = decomposition['total_uncertainty'] > high_uncertainty_threshold
        
        return {
            'predictions': combined_pred,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': decomposition['total_uncertainty'],
            'epistemic_contribution': decomposition['epistemic_contribution'],
            'aleatoric_contribution': decomposition['aleatoric_contribution'],
            'dominant_type': decomposition['dominant_type'],
            'confidence': confidence,
            'should_explore': should_explore,
            'uncertainty_level': self._categorize_uncertainty(
                decomposition['total_uncertainty']
            ),
        }
    
    def _update_history(
        self,
        total: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ):
        """Update uncertainty history buffer."""
        idx = self.history_idx.item() % 100
        self.uncertainty_history[idx] = torch.stack([
            total.mean(),
            epistemic.mean(),
            aleatoric.mean(),
        ])
        self.history_idx += 1
    
    def _categorize_uncertainty(self, uncertainty: torch.Tensor) -> str:
        """Categorize uncertainty level."""
        unc_value = uncertainty.mean().item()
        
        if unc_value < 0.3:
            return 'low'
        elif unc_value < 0.6:
            return 'medium'
        elif unc_value < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    def get_uncertainty_trend(self) -> Dict[str, float]:
        """
        Analyze uncertainty trends over time.
        
        Returns:
            Dictionary with trend information
        """
        if self.history_idx < 10:
            return {
                'total_trend': 0.0,
                'epistemic_trend': 0.0,
                'aleatoric_trend': 0.0,
            }
        
        # Get recent history
        recent_length = min(self.history_idx.item(), 100)
        history = self.uncertainty_history[:recent_length]
        
        # Calculate trends (simple linear regression slope)
        indices = torch.arange(recent_length, dtype=torch.float32)
        
        def compute_trend(values):
            if len(values) < 2:
                return 0.0
            corr = torch.corrcoef(torch.stack([indices, values]))
            return corr[0, 1].item() if not torch.isnan(corr[0, 1]) else 0.0
        
        return {
            'total_trend': compute_trend(history[:, 0]),
            'epistemic_trend': compute_trend(history[:, 1]),
            'aleatoric_trend': compute_trend(history[:, 2]),
            'is_learning': compute_trend(history[:, 1]) < -0.1,  # Epistemic decreasing
        }
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic information about uncertainty tracking."""
        trends = self.get_uncertainty_trend()
        
        recent_length = min(self.history_idx.item(), 100)
        if recent_length > 0:
            recent_history = self.uncertainty_history[:recent_length]
            avg_uncertainties = recent_history.mean(dim=0)
        else:
            avg_uncertainties = torch.zeros(3)
        
        return {
            'history_length': self.history_idx.item(),
            'avg_total_uncertainty': avg_uncertainties[0].item(),
            'avg_epistemic_uncertainty': avg_uncertainties[1].item(),
            'avg_aleatoric_uncertainty': avg_uncertainties[2].item(),
            'trends': trends,
            'is_learning': trends['is_learning'],
        }
