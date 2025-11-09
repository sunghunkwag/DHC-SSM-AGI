"""
DHC-SSM-AGI: Robust Self-Improvement Core
Type-stable, enum-safe, diagnostic-aware. Includes all strategies, history, and validation.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

class ImprovementStrategy(Enum):
    ARCHITECTURE_MODIFICATION = "architecture"
    HYPERPARAMETER_TUNING = "hyperparameter"
    LEARNING_RATE_ADAPTATION = "learning_rate"
    CAPACITY_EXPANSION = "capacity"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "distillation"

@dataclass
class ImprovementHypothesis:
    strategy: ImprovementStrategy
    description: str
    expected_improvement: float
    confidence: float
    parameters: Dict[str, Any]
    id: int = -1

@dataclass
class ExperimentResult:
    hypothesis: ImprovementHypothesis
    actual_improvement: float
    validation_metrics: Dict[str, float]
    success: bool
    learned_patterns: Optional[Dict[str, Any]] = None
    idx: int = -1

class HypothesisGenerator(nn.Module):
    def __init__(self, state_dim: int = 128, num_strategies: int = 6):
        super().__init__()
        self.num_strategies = num_strategies
        self.hypothesis_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, num_strategies * 3),
        )
    def forward(self, state: torch.Tensor, top_k: int = 3) -> List[ImprovementHypothesis]:
        output = self.hypothesis_net(state)
        output = output.view(self.num_strategies, 3)
        strategy_scores = torch.softmax(output[:, 0], dim=0)
        expected_improvements = torch.sigmoid(output[:, 1])
        confidences = torch.sigmoid(output[:, 2])
        top_indices = torch.topk(strategy_scores, k=min(top_k, self.num_strategies)).indices
        strategies = list(ImprovementStrategy)
        hypotheses = []
        for rank, idx in enumerate(top_indices):
            idx = idx.item()
            hypotheses.append(ImprovementHypothesis(
                strategy=strategies[idx],
                description=f"Apply {strategies[idx].value} improvement",
                expected_improvement=expected_improvements[idx].item(),
                confidence=confidences[idx].item(),
                parameters={},
                id=rank
            ))
        return hypotheses

class ExperimentValidator(nn.Module):
    def __init__(self, metric_dim: int = 32):
        super().__init__()
        # Validation as before...
        self.validator = nn.Sequential(
            nn.Linear(metric_dim * 2, 128), nn.ReLU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )
        self.pattern_learner = nn.LSTM(metric_dim * 2 + 1, 64, 2, batch_first=True)
    def validate_experiment(self, before_metrics: torch.Tensor, after_metrics: torch.Tensor, threshold: float = 0.05) -> (bool, float):
        metrics_concat = torch.cat([before_metrics, after_metrics], dim=-1)
        improvement_score = self.validator(metrics_concat).item()
        actual_improvement = (after_metrics - before_metrics).mean().item()
        success = actual_improvement > threshold and improvement_score > 0.5
        return success, improvement_score
    def learn_from_result(self, result: ExperimentResult, before_metrics: torch.Tensor, after_metrics: torch.Tensor):
        success_flag = torch.tensor([1.0 if result.success else 0.0])
        pattern_input = torch.cat([before_metrics, after_metrics, success_flag]).unsqueeze(0).unsqueeze(0)
        _, (hidden, cell) = self.pattern_learner(pattern_input)
        result.learned_patterns = {'hidden_state': hidden.detach(), 'cell_state': cell.detach()}

class RecursiveSelfImprovement(nn.Module):
    def __init__(self, state_dim: int = 128, metric_dim: int = 32, num_strategies: int = 6):
        super().__init__()
        self.hypothesis_generator = HypothesisGenerator(state_dim, num_strategies)
        self.experiment_validator = ExperimentValidator(metric_dim)
        self.experiment_history: List[ExperimentResult] = []
        self.strategy_optimizer = nn.Sequential(
            nn.Linear(state_dim + metric_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_strategies), nn.Softmax(dim=-1),
        )
    def generate_hypotheses(self, current_state: torch.Tensor, top_k: int = 3) -> List[ImprovementHypothesis]:
        return self.hypothesis_generator(current_state, top_k)
    def validate_improvement(self, hypothesis: ImprovementHypothesis, before_metrics: torch.Tensor, after_metrics: torch.Tensor) -> ExperimentResult:
        success, improvement_score = self.experiment_validator.validate_experiment(before_metrics, after_metrics)
        actual_improvement = (after_metrics - before_metrics).mean().item()
        result = ExperimentResult(
            hypothesis=hypothesis,
            actual_improvement=actual_improvement,
            validation_metrics={'improvement_score': improvement_score, 'mean_improvement': actual_improvement},
            success=success,
            idx=len(self.experiment_history),
        )
        self.experiment_validator.learn_from_result(result, before_metrics, after_metrics)
        self.experiment_history.append(result)
        return result
    def get_success_rate(self) -> float:
        s = [r.success for r in self.experiment_history]
        return (sum(s)/len(s)) if s else 0.0
    def get_best_strategies(self, top_k: int = 3) -> List[ImprovementStrategy]:
        if not self.experiment_history:
            return list(ImprovementStrategy)[:top_k]
        scores = {s: {'success':0,'total':0} for s in ImprovementStrategy}
        for r in self.experiment_history:
            scores[r.hypothesis.strategy]['total'] += 1
            if r.success:
                scores[r.hypothesis.strategy]['success'] += 1
        strat_rates = {k: (v['success']/v['total']) if v['total']>0 else 0 for k,v in scores.items()}
        return sorted(strat_rates, key=lambda k: strat_rates[k], reverse=True)[:top_k]
    def get_diagnostics(self)->Dict[str,Any]:
        return {
            'total_experiments': len(self.experiment_history),
            'success_rate': self.get_success_rate(),
            'best_strategies': [s.value for s in self.get_best_strategies()],
            'recent_results': [str(r) for r in self.experiment_history[-5:]],
        }
