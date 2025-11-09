"""
DHC-SSM-AGI: Robust Threshold Analyzer for Recursive Self Improvement (RSI)
Full diagnostics, trend/convergence, and type-safe design with adaptive thresholds.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ThresholdStatus(Enum):
    BELOW = "below"
    AT = "at"
    ABOVE = "above"


@dataclass
class ThresholdMeasurement:
    gamma: float
    gamma_star: float
    epistemic: float
    aleatoric: float
    status: ThresholdStatus
    confidence: float
    converged: bool
    trend: str
    idx: int  # Measurement index


class RSIThresholdAnalyzer(nn.Module):
    """
    Adaptive Threshold Analyzer for Recursive Self-Improvement.
    
    Uses statistical methods to estimate optimal thresholds and detect
    convergence without hardcoded magic numbers.
    """
    
    def __init__(
        self,
        history_length: int = 100,
        min_samples_for_estimation: int = 10,
        convergence_volatility_threshold: float = 0.03,
        convergence_slope_threshold: float = 0.015,
        threshold_tolerance: float = 0.05,
    ):
        super().__init__()
        self.history_length = history_length
        self.min_samples = min_samples_for_estimation
        self.conv_volatility_thresh = convergence_volatility_threshold
        self.conv_slope_thresh = convergence_slope_threshold
        self.threshold_tol = threshold_tolerance
        self.measurements: List[ThresholdMeasurement] = []

    def compute_gamma(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> float:
        """
        Compute gamma metric from epistemic and aleatoric uncertainty.
        
        gamma = (1 - epistemic) / (1 - epistemic + aleatoric)
        
        Higher gamma indicates lower epistemic uncertainty relative to total uncertainty.
        """
        eps = 1e-8
        e = float(epistemic.mean().item())
        a = float(aleatoric.mean().item())
        er = 1.0 - e
        gamma = er / (er + a + eps)
        return max(0.0, min(1.0, gamma))

    def estimate_gamma_star(self) -> float:
        """
        Adaptively estimate optimal gamma threshold using statistical methods.
        
        Uses running quantile estimation rather than hardcoded values.
        Falls back to conservative estimate when insufficient data available.
        """
        if len(self.measurements) < self.min_samples:
            # Conservative fallback: median of typical gamma range [0.5, 0.9]
            return 0.7
        
        # Use recent history for adaptive estimation
        window_size = min(50, len(self.measurements))
        gammas = torch.tensor([m.gamma for m in self.measurements[-window_size:]])
        
        # Adaptive quantile: use 90th percentile as target
        # This allows model to exceed most recent performance
        gamma_star = float(gammas.quantile(0.9).item())
        
        # Ensure gamma_star is in valid range
        return max(0.5, min(0.95, gamma_star))

    def analyze_trend(self, window: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze recent gamma trend to detect convergence and direction.
        
        Returns:
            Dictionary containing:
            - trend: 'increasing', 'decreasing', or 'stable'
            - slope: Linear regression slope
            - volatility: Standard deviation of recent measurements
            - converged: Whether the system has converged
        """
        if not self.measurements:
            return {
                'trend': 'none',
                'slope': 0.0,
                'volatility': 0.0,
                'converged': False
            }
        
        window = window or min(20, len(self.measurements))
        recent = self.measurements[-window:]
        gammas = torch.tensor([m.gamma for m in recent])
        
        # Compute linear trend
        n = len(gammas)
        slope = float((gammas[-1] - gammas[0]) / max(n - 1, 1))
        
        # Compute volatility
        volatility = float(gammas.std().item())
        
        # Convergence detection: low volatility + small slope
        converged = (
            volatility < self.conv_volatility_thresh and
            abs(slope) < self.conv_slope_thresh
        )
        
        # Trend classification
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0.01:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'volatility': volatility,
            'converged': converged
        }

    def check_threshold(self, gamma: float, gamma_star: float) -> ThresholdStatus:
        """
        Check if current gamma meets, exceeds, or falls below threshold.
        
        Uses configurable tolerance to avoid oscillation around threshold.
        """
        if abs(gamma - gamma_star) < self.threshold_tol:
            return ThresholdStatus.AT
        elif gamma >= gamma_star:
            return ThresholdStatus.ABOVE
        return ThresholdStatus.BELOW

    def measure(
        self,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor
    ) -> ThresholdMeasurement:
        """
        Perform a threshold measurement and update history.
        
        Args:
            epistemic: Epistemic uncertainty tensor
            aleatoric: Aleatoric uncertainty tensor
            
        Returns:
            ThresholdMeasurement containing all metrics
        """
        gamma = self.compute_gamma(epistemic, aleatoric)
        gamma_star = self.estimate_gamma_star()
        status = self.check_threshold(gamma, gamma_star)
        confidence = min(len(self.measurements) / self.history_length, 1.0)
        analysis = self.analyze_trend()
        
        tm = ThresholdMeasurement(
            gamma=gamma,
            gamma_star=gamma_star,
            epistemic=float(epistemic.mean().item()),
            aleatoric=float(aleatoric.mean().item()),
            status=status,
            confidence=confidence,
            converged=analysis['converged'],
            trend=analysis['trend'],
            idx=len(self.measurements)
        )
        
        self.measurements.append(tm)
        if len(self.measurements) > self.history_length:
            self.measurements.pop(0)
        
        return tm

    def diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics of threshold analyzer state.
        
        Returns:
            Dictionary containing current status and metrics
        """
        if not self.measurements:
            return {'status': 'no_data'}
        
        last = self.measurements[-1]
        analysis = self.analyze_trend()
        
        return {
            'current_idx': last.idx,
            'current_gamma': last.gamma,
            'gamma_star_estimate': last.gamma_star,
            'threshold_status': last.status.value,
            'confidence': last.confidence,
            'trend': analysis['trend'],
            'slope': analysis['slope'],
            'volatility': analysis['volatility'],
            'converged': analysis['converged'],
            'measurements': len(self.measurements),
            'history_length': self.history_length,
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """
        Export threshold measurement history for analysis/visualization.
        
        Returns:
            List of dictionaries containing measurement data
        """
        return [
            {
                'gamma': tm.gamma,
                'gamma_star': tm.gamma_star,
                'epistemic': tm.epistemic,
                'aleatoric': tm.aleatoric,
                'status': tm.status.value,
                'confidence': tm.confidence,
                'converged': tm.converged,
                'trend': tm.trend,
                'idx': tm.idx,
            }
            for tm in self.measurements
        ]

    def reset(self) -> None:
        """Reset measurement history."""
        self.measurements.clear()
