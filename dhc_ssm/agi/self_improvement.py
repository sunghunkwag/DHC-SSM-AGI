"""
Recursive Self-Improvement (RSI) System

Implements mechanisms for the system to improve its own architecture,
learning strategies, and decision-making processes through structured
hypothesis-experiment-validation loops.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ImprovementStrategy(Enum):
    """Types of improvement strategies."""
    ARCHITECTURE_MODIFICATION = "architecture"
    HYPERPARAMETER_TUNING = "hyperparameter"
    LEARNING_RATE_ADAPTATION = "learning_rate"
    CAPACITY_EXPANSION = "capacity"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "distillation"


@dataclass
class ImprovementHypothesis:
    """
    Represents a hypothesis about how to improve the system.
    """
    strategy: ImprovementStrategy
    description: str
    expected_improvement: float
    confidence: float
    parameters: Dict[str, any]
    
    def __repr__(self):
        return (f"Hypothesis({self.strategy.value}, "
                f"expected={self.expected_improvement:.3f}, "
                f"confidence={self.confidence:.3f})")


@dataclass
class ExperimentResult:
    """
    Results from testing an improvement hypothesis.
    """
    hypothesis: ImprovementHypothesis
    actual_improvement: float
    validation_metrics: Dict[str, float]
    success: bool
    learned_patterns: Optional[Dict[str, any]] = None
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (f"Result({status}, actual={self.actual_improvement:.3f}, "
                f"expected={self.hypothesis.expected_improvement:.3f})")


class HypothesisGenerator(nn.Module):
    """
    Generates improvement hypotheses based on current state and history.
    """
    
    def __init__(self, state_dim: int = 128, num_strategies: int = 6):
        super().__init__()
        self.num_strategies = num_strategies
        
        # Hypothesis generation network
        self.hypothesis_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, num_strategies * 3),  # [strategy_logits, expected_improvement, confidence]
        )
    
    def forward(
        self,
        state: torch.Tensor,
        top_k: int = 3
    ) -> List[ImprovementHypothesis]:
        """
        Generate improvement hypotheses.
        
        Args:
            state: Current system state embedding
            top_k: Number of top hypotheses to return
            
        Returns:
            List of improvement hypotheses
        """
        output = self.hypothesis_net(state)
        output = output.view(self.num_strategies, 3)
        
        strategy_scores = torch.softmax(output[:, 0], dim=0)
        expected_improvements = torch.sigmoid(output[:, 1])
        confidences = torch.sigmoid(output[:, 2])
        
        # Select top-k strategies
        top_indices = torch.topk(strategy_scores, k=min(top_k, self.num_strategies)).indices
        
        hypotheses = []
        strategies = list(ImprovementStrategy)
        
        for idx in top_indices:
            idx = idx.item()
            hypothesis = ImprovementHypothesis(
                strategy=strategies[idx],
                description=f"Apply {strategies[idx].value} improvement",
                expected_improvement=expected_improvements[idx].item(),
                confidence=confidences[idx].item(),
                parameters={},
            )
            hypotheses.append(hypothesis)
        
        return hypotheses


class ExperimentValidator(nn.Module):
    """
    Validates improvement experiments and learns from results.
    """
    
    def __init__(self, metric_dim: int = 32):
        super().__init__()
        self.metric_dim = metric_dim
        
        # Validation network
        self.validator = nn.Sequential(
            nn.Linear(metric_dim * 2, 128),  # before and after metrics
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Pattern learner
        self.pattern_learner = nn.LSTM(
            input_size=metric_dim * 2 + 1,  # metrics + success flag
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
    
    def validate_experiment(
        self,
        before_metrics: torch.Tensor,
        after_metrics: torch.Tensor,
        threshold: float = 0.05,
    ) -> Tuple[bool, float]:
        """
        Validate if an experiment was successful.
        
        Args:
            before_metrics: Metrics before improvement
            after_metrics: Metrics after improvement
            threshold: Minimum improvement threshold
            
        Returns:
            Tuple of (success, improvement_score)
        """
        metrics_concat = torch.cat([before_metrics, after_metrics], dim=-1)
        improvement_score = self.validator(metrics_concat).item()
        
        # Calculate actual improvement
        actual_improvement = (after_metrics - before_metrics).mean().item()
        
        success = actual_improvement > threshold and improvement_score > 0.5
        
        return success, improvement_score
    
    def learn_from_result(
        self,
        result: ExperimentResult,
        before_metrics: torch.Tensor,
        after_metrics: torch.Tensor,
    ):
        """
        Learn patterns from experiment results.
        
        This updates the internal model to better predict future improvements.
        """
        success_flag = torch.tensor([1.0 if result.success else 0.0])
        pattern_input = torch.cat([
            before_metrics,
            after_metrics,
            success_flag,
        ]).unsqueeze(0).unsqueeze(0)
        
        # Update pattern memory
        _, (hidden, cell) = self.pattern_learner(pattern_input)
        
        # Store learned patterns (in practice, this would update a memory bank)
        result.learned_patterns = {
            'hidden_state': hidden.detach(),
            'cell_state': cell.detach(),
        }


class RecursiveSelfImprovement(nn.Module):
    """
    Recursive Self-Improvement System
    
    Implements a complete RSI loop:
    1. Generate improvement hypotheses
    2. Test hypotheses through experiments
    3. Validate results
    4. Learn from outcomes (including failures)
    5. Update improvement strategies
    """
    
    def __init__(
        self,
        state_dim: int = 128,
        metric_dim: int = 32,
        num_strategies: int = 6,
    ):
        super().__init__()
        
        self.hypothesis_generator = HypothesisGenerator(state_dim, num_strategies)
        self.experiment_validator = ExperimentValidator(metric_dim)
        
        # Experiment history
        self.experiment_history: List[ExperimentResult] = []
        
        # Meta-learner for improvement strategy
        self.strategy_optimizer = nn.Sequential(
            nn.Linear(state_dim + metric_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_strategies),
            nn.Softmax(dim=-1),
        )
    
    def generate_hypotheses(
        self,
        current_state: torch.Tensor,
        top_k: int = 3,
    ) -> List[ImprovementHypothesis]:
        """
        Generate improvement hypotheses based on current state.
        
        Args:
            current_state: Current system state embedding
            top_k: Number of hypotheses to generate
            
        Returns:
            List of improvement hypotheses
        """
        return self.hypothesis_generator(current_state, top_k)
    
    def validate_improvement(
        self,
        hypothesis: ImprovementHypothesis,
        before_metrics: torch.Tensor,
        after_metrics: torch.Tensor,
    ) -> ExperimentResult:
        """
        Validate an improvement experiment.
        
        Args:
            hypothesis: The tested hypothesis
            before_metrics: Metrics before improvement
            after_metrics: Metrics after improvement
            
        Returns:
            Experiment result
        """
        success, improvement_score = self.experiment_validator.validate_experiment(
            before_metrics, after_metrics
        )
        
        actual_improvement = (after_metrics - before_metrics).mean().item()
        
        result = ExperimentResult(
            hypothesis=hypothesis,
            actual_improvement=actual_improvement,
            validation_metrics={
                'improvement_score': improvement_score,
                'mean_improvement': actual_improvement,
            },
            success=success,
        )
        
        # Learn from result
        self.experiment_validator.learn_from_result(result, before_metrics, after_metrics)
        
        # Add to history
        self.experiment_history.append(result)
        
        return result
    
    def optimize_strategy(
        self,
        current_state: torch.Tensor,
        current_metrics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimize improvement strategy based on history.
        
        Args:
            current_state: Current system state
            current_metrics: Current performance metrics
            
        Returns:
            Optimized strategy distribution
        """
        strategy_input = torch.cat([current_state, current_metrics], dim=-1)
        strategy_distribution = self.strategy_optimizer(strategy_input)
        
        return strategy_distribution
    
    def get_success_rate(self) -> float:
        """Calculate success rate of past experiments."""
        if not self.experiment_history:
            return 0.0
        
        successes = sum(1 for result in self.experiment_history if result.success)
        return successes / len(self.experiment_history)
    
    def get_best_strategies(self, top_k: int = 3) -> List[ImprovementStrategy]:
        """
        Get the most successful improvement strategies from history.
        
        Args:
            top_k: Number of top strategies to return
            
        Returns:
            List of best strategies
        """
        if not self.experiment_history:
            return list(ImprovementStrategy)[:top_k]
        
        # Count successes by strategy
        strategy_scores = {}
        for result in self.experiment_history:
            strategy = result.hypothesis.strategy
            if strategy not in strategy_scores:
                strategy_scores[strategy] = {'success': 0, 'total': 0}
            
            strategy_scores[strategy]['total'] += 1
            if result.success:
                strategy_scores[strategy]['success'] += 1
        
        # Calculate success rates
        strategy_rates = {
            strategy: scores['success'] / scores['total']
            for strategy, scores in strategy_scores.items()
        }
        
        # Sort by success rate
        sorted_strategies = sorted(
            strategy_rates.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [strategy for strategy, _ in sorted_strategies[:top_k]]
    
    def get_diagnostics(self) -> Dict[str, any]:
        """Get diagnostic information about RSI system."""
        return {
            'total_experiments': len(self.experiment_history),
            'success_rate': self.get_success_rate(),
            'best_strategies': [s.value for s in self.get_best_strategies()],
            'recent_results': [str(r) for r in self.experiment_history[-5:]],
        }
