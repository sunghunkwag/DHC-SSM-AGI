"""
DHC-SSM-AGI: Robust Self-Improvement Executor
Safely applies improvement strategies to models, supports rollback,
validates improvements with integrated threshold analysis and real uncertainty quantification.
"""
import copy
import torch
import torch.nn as nn
from typing import Tuple, Any, Dict, Optional
import logging

from dhc_ssm.agi.self_improvement import (
    RecursiveSelfImprovement,
    ImprovementHypothesis,
    ImprovementStrategy
)
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
from dhc_ssm.agi.uncertainty import UncertaintyQuantifier

logger = logging.getLogger(__name__)


class SelfImprovementExecutor:
    """
    Executor for recursive self-improvement with real uncertainty quantification.
    
    Integrates:
    - Real UncertaintyQuantifier for epistemic/aleatoric estimation
    - Threshold-based improvement validation
    - Multiple improvement strategies
    - Safe rollback mechanism
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        validation_data: Tuple[torch.Tensor, torch.Tensor],
        threshold_analyzer: Optional[RSIThresholdAnalyzer] = None,
        uncertainty_quantifier: Optional[UncertaintyQuantifier] = None,
        feature_dim: int = 256,
    ):
        self.base_model = base_model
        self.validation_data = validation_data
        self.feature_dim = feature_dim
        
        # Initialize or use provided components
        self.threshold_analyzer = threshold_analyzer or RSIThresholdAnalyzer()
        
        # Initialize uncertainty quantifier with proper dimensions
        if uncertainty_quantifier is None:
            output_dim = self._infer_output_dim(base_model)
            self.uncertainty_quantifier = UncertaintyQuantifier(
                input_dim=feature_dim,
                output_dim=output_dim,
                num_ensemble_heads=5,
            )
        else:
            self.uncertainty_quantifier = uncertainty_quantifier
        
        self.rsi = RecursiveSelfImprovement()
        self.history = []
        
        # Feature extractor (will be updated based on model type)
        self._setup_feature_extractor()
    
    def _infer_output_dim(self, model: nn.Module) -> int:
        """Infer output dimension from model."""
        # Try to find the last linear layer
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        # Default fallback
        return 10
    
    def _setup_feature_extractor(self):
        """Setup feature extraction based on model architecture."""
        # Simple feature extractor - can be customized per model type
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)) if self._has_conv_layers() else nn.Identity(),
            nn.Flatten(),
        )
    
    def _has_conv_layers(self) -> bool:
        """Check if model has convolutional layers."""
        for module in self.base_model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                return True
        return False
    
    def _extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from model for uncertainty quantification.
        
        Args:
            model: Model to extract features from
            x: Input data
            
        Returns:
            Feature tensor
        """
        model.eval()
        with torch.no_grad():
            # Try to get intermediate features
            if hasattr(model, 'get_features'):
                features = model.get_features(x)
            else:
                # Fallback: use model output as features
                output = model(x)
                features = output
            
            # Ensure correct dimensionality
            if features.dim() > 2:
                features = self.feature_extractor(features)
            
            # Project to expected feature dimension if needed
            if features.shape[-1] != self.feature_dim:
                if not hasattr(self, '_feature_projector'):
                    self._feature_projector = nn.Linear(
                        features.shape[-1], self.feature_dim
                    ).to(features.device)
                features = self._feature_projector(features)
        
        return features
    
    def evaluate_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate model with real uncertainty quantification.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Dictionary containing accuracy and uncertainty estimates
        """
        model.eval()
        x, y = self.validation_data
        
        with torch.no_grad():
            # Extract features
            features = self._extract_features(model, x)
            
            # Get uncertainty estimates from real quantifier
            uncertainty_results = self.uncertainty_quantifier(features)
            
            # Get model predictions
            outputs = model(x)
            pred_classes = outputs.argmax(dim=-1)
            accuracy = (pred_classes == y).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'epistemic_uncertainty': uncertainty_results['epistemic_uncertainty'],
            'aleatoric_uncertainty': uncertainty_results['aleatoric_uncertainty'],
            'total_uncertainty': uncertainty_results['total_uncertainty'],
            'confidence': uncertainty_results['confidence'],
            'should_explore': uncertainty_results['should_explore'],
        }
    
    def apply_hypothesis(
        self,
        model: nn.Module,
        hypothesis: ImprovementHypothesis
    ) -> nn.Module:
        """
        Apply improvement hypothesis to model.
        
        Supports multiple improvement strategies:
        - Capacity expansion
        - Learning rate adaptation
        - Architecture refinement
        
        Args:
            model: Model to improve
            hypothesis: Improvement hypothesis to apply
            
        Returns:
            Modified model
        """
        new_model = copy.deepcopy(model)
        
        try:
            if hypothesis.strategy == ImprovementStrategy.CAPACITY_EXPANSION:
                new_model = self._expand_capacity(new_model)
            
            elif hypothesis.strategy == ImprovementStrategy.LEARNING_RATE_ADAPTATION:
                # This would require optimizer integration
                logger.info("Learning rate adaptation requires optimizer context")
            
            elif hypothesis.strategy == ImprovementStrategy.ARCHITECTURE_REFINEMENT:
                new_model = self._refine_architecture(new_model)
            
            else:
                logger.warning(f"Unhandled strategy: {hypothesis.strategy}")
        
        except Exception as e:
            logger.error(f"Failed to apply hypothesis: {e}")
            return model  # Return original on failure
        
        return new_model
    
    def _expand_capacity(self, model: nn.Module, expansion_factor: float = 1.2) -> nn.Module:
        """
        Expand model capacity by increasing layer dimensions.
        
        Args:
            model: Model to expand
            expansion_factor: Factor to expand by (default 1.2 = 20% increase)
            
        Returns:
            Expanded model
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                new_out = int(module.out_features * expansion_factor)
                new_linear = nn.Linear(module.in_features, new_out)
                
                # Initialize with existing weights + new random weights
                with torch.no_grad():
                    new_linear.weight[:module.out_features] = module.weight
                    new_linear.bias[:module.out_features] = module.bias
                
                setattr(model, name, new_linear)
                break  # Only expand first linear layer for now
        
        return model
    
    def _refine_architecture(self, model: nn.Module) -> nn.Module:
        """
        Refine architecture by adding regularization or normalization.
        
        Args:
            model: Model to refine
            
        Returns:
            Refined model
        """
        # Add dropout if not present
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and not isinstance(module, nn.Dropout):
                # Wrap linear layer with dropout
                refined = nn.Sequential(
                    module,
                    nn.Dropout(0.1),
                )
                setattr(model, name, refined)
                break
        
        return model
    
    def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete self-improvement cycle.
        
        Returns:
            Dictionary containing:
            - result: Validation result
            - threshold: Threshold measurement
            - action: 'update' or 'rollback'
            - rollback: Whether rollback occurred
        """
        current_model = copy.deepcopy(self.base_model)
        state = torch.randn(128)
        
        # Evaluate current model
        before_metrics = self.evaluate_model(current_model)
        logger.info(f"Before: accuracy={before_metrics['accuracy']:.3f}")
        
        # Generate and apply hypothesis
        hypotheses = self.rsi.generate_hypotheses(state, top_k=1)
        hypothesis = hypotheses[0]
        
        modified_model = self.apply_hypothesis(current_model, hypothesis)
        
        # Evaluate modified model
        after_metrics = self.evaluate_model(modified_model)
        logger.info(f"After: accuracy={after_metrics['accuracy']:.3f}")
        
        # Threshold analysis with real uncertainties
        tm = self.threshold_analyzer.measure(
            after_metrics['epistemic_uncertainty'],
            after_metrics['aleatoric_uncertainty']
        )
        
        # Validate improvement
        result = self.rsi.validate_improvement(
            hypothesis,
            torch.tensor([before_metrics['accuracy']]),
            torch.tensor([after_metrics['accuracy']])
        )
        result.threshold_analysis = tm
        
        # Decide: update or rollback
        # Update only if: improvement successful AND threshold exceeded
        improvement_valid = result.success and after_metrics['accuracy'] > before_metrics['accuracy']
        threshold_valid = tm.status != tm.status.BELOW
        
        rollback = not (improvement_valid and threshold_valid)
        
        if not rollback:
            self.base_model = modified_model
            action = 'update'
            logger.info(f"✓ Model updated (gamma={tm.gamma:.3f}, gamma*={tm.gamma_star:.3f})")
        else:
            action = 'rollback'
            logger.info(f"✗ Rolled back (improvement={improvement_valid}, threshold={threshold_valid})")
        
        # Record history
        cycle_info = {
            'result': result,
            'threshold': tm,
            'action': action,
            'rollback': rollback,
            'before_accuracy': before_metrics['accuracy'],
            'after_accuracy': after_metrics['accuracy'],
        }
        self.history.append(cycle_info)
        
        return cycle_info
    
    def get_last_model(self) -> nn.Module:
        """Return the latest active model (after last cycle)."""
        return self.base_model
    
    def diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics.
        
        Returns:
            Dictionary containing executor state and history
        """
        return {
            'cycle_count': len(self.history),
            'last_action': self.history[-1]['action'] if self.history else None,
            'last_threshold': self.history[-1]['threshold'] if self.history else None,
            'last_result_success': self.history[-1]['result'].success if self.history else None,
            'threshold_diagnostics': self.threshold_analyzer.diagnostics(),
            'uncertainty_diagnostics': self.uncertainty_quantifier.get_diagnostics(),
            'rsi_diagnostics': self.rsi.get_diagnostics(),
        }
