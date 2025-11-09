"""
RSIThresholdAnalyzer: Complete Threshold Analysis with Diagnostics and Model Integration
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional
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

class RSIThresholdAnalyzer(nn.Module):
    def __init__(self, history_length: int = 100):
        super().__init__()
        self.history_length = history_length
        self.measurements: List[ThresholdMeasurement] = []

    def compute_gamma(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> float:
        eps = 1e-8
        e = float(epistemic.mean().item())
        a = float(aleatoric.mean().item())
        er = 1.0 - e
        gamma = er / (er + a + eps)
        return max(0.0, min(1.0, gamma))

    def estimate_gamma_star(self) -> float:
        if len(self.measurements) < 10:
            return 0.7
        gammas = [m.gamma for m in self.measurements[-50:]]
        return float(torch.tensor(gammas).quantile(0.9).item())

    def analyze_trend(self, window: Optional[int] = None) -> Dict[str, any]:
        if not self.measurements:
            return {'trend': 'none', 'slope': 0.0, 'volatility': 0.0, 'converged': False}
        window = window or min(20, len(self.measurements))
        recent = self.measurements[-window:]
        gammas = torch.tensor([m.gamma for m in recent])
        slope = (gammas[-1] - gammas[0]) / len(gammas)
        volatility = float(gammas.std().item())
        converged = volatility < 0.03
        trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
        return { 'trend': trend, 'slope': float(slope), 'volatility': volatility, 'converged': converged }

    def check_threshold(self, gamma: float, gamma_star: float) -> ThresholdStatus:
        tol = 0.05
        if abs(gamma - gamma_star) < tol:
            return ThresholdStatus.AT
        elif gamma >= gamma_star:
            return ThresholdStatus.ABOVE
        return ThresholdStatus.BELOW

    def measure(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> ThresholdMeasurement:
        gamma = self.compute_gamma(epistemic, aleatoric)
        gamma_star = self.estimate_gamma_star()
        status = self.check_threshold(gamma, gamma_star)
        confidence = min(len(self.measurements)/self.history_length, 1.0)
        analysis = self.analyze_trend()
        tm = ThresholdMeasurement(gamma, gamma_star, float(epistemic.mean().item()), float(aleatoric.mean().item()), status, confidence, analysis['converged'], analysis['trend'])
        self.measurements.append(tm)
        if len(self.measurements) > self.history_length:
            self.measurements.pop(0)
        return tm

    def diagnostics(self) -> Dict[str, any]:
        if not self.measurements:
            return {'status': 'no_data'}
        last = self.measurements[-1]
        analysis = self.analyze_trend()
        return {
            'current_gamma': last.gamma,
            'gamma_star_estimate': last.gamma_star,
            'threshold_status': last.status.value,
            'confidence': last.confidence,
            'trend': analysis['trend'],
            'converged': analysis['converged'],
            'measurements': len(self.measurements),
        }
