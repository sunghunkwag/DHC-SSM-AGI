"""
AGI-Enhanced DHC-SSM Model

Integrates all AGI components with the base DHC-SSM architecture to create
a system capable of meta-cognition, self-improvement, and adaptive learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from ..core.model import DHCSSMModel
from .metacognition import MetaCognitiveLayer
from .self_improvement import RecursiveSelfImprovement
from .goal_system import DynamicGoalSystem
from .uncertainty import UncertaintyQuantifier
from .meta_learning import MetaLearningEngine

logger = logging.getLogger(__name__)


class AGIEnhancedDHCSSM(nn.Module):
    """
    AGI-Enhanced DHC-SSM Model
    
    Combines the efficient DHC-SSM architecture with advanced AGI capabilities:
    - Meta-cognitive self-awareness
    - Recursive self-improvement
    - Dynamic goal redefinition
    - Uncertainty quantification
    - Meta-learning for rapid adaptation
    
    This creates a system that not only processes information efficiently
    but also understands itself, improves itself, and adapts to new challenges.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base DHC-SSM model
        self.base_model = DHCSSMModel(config)
        
        # Get dimensions from config
        hidden_dim = getattr(config, 'hidden_dim', 64) * 4
        output_dim = getattr(config, 'output_dim', 10)
        
        # AGI components
        self.metacognition = MetaCognitiveLayer(
            config_dim=128,
            metric_dim=32,
            embedding_dim=64,
            history_length=100,
        )
        
        self.self_improvement = RecursiveSelfImprovement(
            state_dim=128,
            metric_dim=32,
            num_strategies=6,
        )
        
        self.goal_system = DynamicGoalSystem(
            context_dim=128,
            goal_dim=64,
            num_goal_types=7,
        )
        
        self.uncertainty = UncertaintyQuantifier(
            input_dim=hidden_dim,
            output_dim=output_dim,
            num_ensemble_heads=5,
        )
        
        # Meta-learning engine (initialized after base model)
        self.meta_learning = MetaLearningEngine(
            model=self.base_model,
            input_dim=hidden_dim,
            task_dim=64,
            inner_lr=0.01,
            outer_lr=0.001,
            num_inner_steps=5,
        )
        
        # Context aggregator
        self.context_aggregator = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 32, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )
        
        # Configuration vectorizer
        self.config_vectorizer = nn.Linear(10, 128)
        
        # Mode flags
        self.training_mode = True
        self.meta_learning_mode = False
        self.self_improvement_mode = False
    
    def vectorize_config(self) -> torch.Tensor:
        """Convert configuration to vector representation."""
        config_features = torch.tensor([
            getattr(self.config, 'hidden_dim', 64),
            getattr(self.config, 'ssm_state_dim', 64),
            getattr(self.config, 'output_dim', 10),
            getattr(self.config, 'learning_rate', 0.001),
            1.0 if self.training_mode else 0.0,
            1.0 if self.meta_learning_mode else 0.0,
            1.0 if self.self_improvement_mode else 0.0,
            float(self.metacognition.performance_monitor.history_idx.item()),
            float(len(self.self_improvement.experiment_history)),
            float(len(self.goal_system.active_goals)),
        ], dtype=torch.float32)
        
        return self.config_vectorizer(config_features)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute performance metrics."""
        metrics = torch.zeros(32)
        
        if targets is not None:
            # Accuracy
            pred_classes = predictions.argmax(dim=-1)
            accuracy = (pred_classes == targets).float().mean()
            metrics[0] = accuracy
            
            # Loss
            loss = nn.functional.cross_entropy(predictions, targets)
            metrics[1] = loss.item()
            
            # Confidence
            confidence = torch.softmax(predictions, dim=-1).max(dim=-1)[0].mean()
            metrics[2] = confidence
        else:
            # Prediction entropy
            probs = torch.softmax(predictions, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            metrics[0] = entropy
        
        return metrics
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        enable_agi: bool = True,
    ) -> Dict[str, any]:
        """
        Forward pass with AGI enhancements.
        
        Args:
            x: Input tensor
            targets: Optional target labels
            enable_agi: Whether to enable AGI components
            
        Returns:
            Dictionary containing:
            - predictions: Model predictions
            - uncertainty: Uncertainty estimates
            - metacognition: Meta-cognitive analysis
            - should_adapt: Whether adaptation is recommended
            - diagnostics: System diagnostics
        """
        # Base model forward pass
        base_output = self.base_model(x)
        
        if not enable_agi:
            return {'predictions': base_output}
        
        # Get features from base model (before final classifier)
        with torch.no_grad():
            spatial_features = self.base_model.spatial_encoder(x)
            temporal_features = self.base_model.temporal_ssm(spatial_features)
        
        # Uncertainty quantification
        uncertainty_results = self.uncertainty(temporal_features)
        
        # Compute current metrics
        current_metrics = self.compute_metrics(base_output, targets)
        
        # Vectorize configuration
        config_vector = self.vectorize_config()
        
        # Meta-cognitive analysis
        metacog_results = self.metacognition(config_vector, current_metrics)
        
        # Aggregate context
        context = self.context_aggregator(torch.cat([
            temporal_features.mean(dim=0),
            metacog_results['structural_embedding'],
            current_metrics,
        ]))
        
        # Goal system evaluation (if in self-improvement mode)
        goal_results = None
        if self.self_improvement_mode:
            goal_results = self.goal_system.redefine_goals(context)
        
        # Determine if adaptation is needed
        should_adapt = (
            metacog_results['should_adapt'] or
            uncertainty_results['should_explore'] or
            (goal_results and len(goal_results['goals_reconsidered']) > 0)
        )
        
        return {
            'predictions': uncertainty_results['predictions'],
            'base_predictions': base_output,
            'uncertainty': {
                'total': uncertainty_results['total_uncertainty'],
                'epistemic': uncertainty_results['epistemic_uncertainty'],
                'aleatoric': uncertainty_results['aleatoric_uncertainty'],
                'confidence': uncertainty_results['confidence'],
                'level': uncertainty_results['uncertainty_level'],
            },
            'metacognition': {
                'structural_embedding': metacog_results['structural_embedding'],
                'performance_trend': metacog_results['performance_analysis']['trend'],
                'weakness_score': metacog_results['weakness_score'],
                'meta_decision': metacog_results['meta_decision'],
            },
            'should_adapt': should_adapt,
            'should_explore': uncertainty_results['should_explore'],
            'goal_results': goal_results,
        }
    
    def adapt_to_task(
        self,
        task_data: Tuple[torch.Tensor, torch.Tensor],
        task_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Adapt the model to a new task using meta-learning.
        
        Args:
            task_data: Tuple of (inputs, targets)
            task_id: Optional task identifier
            
        Returns:
            Adaptation results
        """
        self.meta_learning_mode = True
        adaptation_results = self.meta_learning.adapt_to_task(task_data, task_id)
        self.meta_learning_mode = False
        
        return adaptation_results
    
    def improve_self(
        self,
        current_state: torch.Tensor,
        current_metrics: torch.Tensor,
    ) -> Dict[str, any]:
        """
        Trigger self-improvement process.
        
        Args:
            current_state: Current system state
            current_metrics: Current performance metrics
            
        Returns:
            Self-improvement results
        """
        self.self_improvement_mode = True
        
        # Generate improvement hypotheses
        hypotheses = self.self_improvement.generate_hypotheses(current_state, top_k=3)
        
        # Optimize improvement strategy
        strategy_distribution = self.self_improvement.optimize_strategy(
            current_state, current_metrics
        )
        
        self.self_improvement_mode = False
        
        return {
            'hypotheses': hypotheses,
            'strategy_distribution': strategy_distribution,
            'success_rate': self.self_improvement.get_success_rate(),
            'best_strategies': self.self_improvement.get_best_strategies(),
        }
    
    def get_comprehensive_diagnostics(self) -> Dict[str, any]:
        """
        Get comprehensive diagnostics from all AGI components.
        
        Returns:
            Complete system diagnostics
        """
        return {
            'base_model': self.base_model.get_diagnostics(),
            'metacognition': self.metacognition.get_diagnostics(),
            'self_improvement': self.self_improvement.get_diagnostics(),
            'goal_system': self.goal_system.get_diagnostics(),
            'uncertainty': self.uncertainty.get_diagnostics(),
            'meta_learning': self.meta_learning.get_diagnostics(),
            'modes': {
                'training': self.training_mode,
                'meta_learning': self.meta_learning_mode,
                'self_improvement': self.self_improvement_mode,
            },
        }
    
    def enable_training_mode(self):
        """Enable training mode."""
        self.training_mode = True
        self.train()
    
    def enable_eval_mode(self):
        """Enable evaluation mode."""
        self.training_mode = False
        self.eval()
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint including all AGI components.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'base_model': self.base_model.state_dict(),
            'metacognition': self.metacognition.state_dict(),
            'self_improvement': {
                'state_dict': self.self_improvement.state_dict(),
                'history': self.self_improvement.experiment_history,
            },
            'goal_system': {
                'state_dict': self.goal_system.state_dict(),
                'active_goals': self.goal_system.active_goals,
                'goal_history': self.goal_system.goal_history,
            },
            'uncertainty': self.uncertainty.state_dict(),
            'meta_learning': self.meta_learning.state_dict(),
            'config': self.config,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint including all AGI components.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path)
        
        self.base_model.load_state_dict(checkpoint['base_model'])
        self.metacognition.load_state_dict(checkpoint['metacognition'])
        self.self_improvement.load_state_dict(checkpoint['self_improvement']['state_dict'])
        self.self_improvement.experiment_history = checkpoint['self_improvement']['history']
        self.goal_system.load_state_dict(checkpoint['goal_system']['state_dict'])
        self.goal_system.active_goals = checkpoint['goal_system']['active_goals']
        self.goal_system.goal_history = checkpoint['goal_system']['goal_history']
        self.uncertainty.load_state_dict(checkpoint['uncertainty'])
        self.meta_learning.load_state_dict(checkpoint['meta_learning'])
        
        logger.info(f"Checkpoint loaded from {path}")
