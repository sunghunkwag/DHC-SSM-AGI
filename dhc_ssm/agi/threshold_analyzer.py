"""
Threshold Analysis for Recursive Self-Improvement (RSI)

Analyzes whether the system has crossed the critical threshold (Γ★) required
for recursive self-improvement, based on epistemic and aleatoric uncertainties.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
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

class RSIThresholdAnalyzer(nn.Module):
    def __init__(self, history_length: int = 100):
        super().__init__()
        self.history_length = history_length
        self.measurements: List[ThresholdMeasurement] = []

    def compute_gamma(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> float:
        eps = 1e-8
        e = epistemic.mean().item() if epistemic.numel() > 1 else epistemic.item()
        a = aleatoric.mean().item() if aleatoric.numel() > 1 else aleatoric.item()
        er = 1.0 - e
        gamma = er / (er + a + eps)
        return max(0.0, min(1.0, gamma))

    def estimate_gamma_star(self) -> float:
        if len(self.measurements) < 10:
            return 0.7
        gammas = [m.gamma for m in self.measurements[-50:]]
        return torch.tensor(gammas).quantile(0.9).item()

    def check_threshold(self, gamma: float, gamma_star: float) -> ThresholdStatus:
        tol = 0.05
        if abs(gamma - gamma_star) < tol:
            return ThresholdStatus.AT
        elif gamma >= gamma_star:
            return ThresholdStatus.ABOVE
        else:
            return ThresholdStatus.BELOW

    def measure(self, epistemic: torch.Tensor, aleatoric: torch.Tensor) -> ThresholdMeasurement:
        gamma = self.compute_gamma(epistemic, aleatoric)
        gamma_star = self.estimate_gamma_star()
        status = self.check_threshold(gamma, gamma_star)
        confidence = min(len(self.measurements)/self.history_length, 1.0)
        tm = ThresholdMeasurement(gamma, gamma_star, epistemic.mean().item(), aleatoric.mean().item(), status, confidence)
        self.measurements.append(tm)
        if len(self.measurements) > self.history_length:
            self.measurements.pop(0)
        return tm
