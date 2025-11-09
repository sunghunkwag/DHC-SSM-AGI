"""
DHC-SSM-AGI: Robust Self-Improvement Executor
Safely applies improvement strategies to models, supports rollback,
validates improvements with integrated threshold analysis, and diagnostics.
"""
import copy
import torch
from typing import Tuple, Any, Dict
from dhc_ssm.agi.self_improvement import RecursiveSelfImprovement, ImprovementHypothesis, ImprovementStrategy
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer

class SelfImprovementExecutor:
    def __init__(self, base_model: torch.nn.Module, validation_data: Tuple[torch.Tensor, torch.Tensor], threshold_analyzer: RSIThresholdAnalyzer = None):
        self.base_model = base_model
        self.validation_data = validation_data
        self.threshold_analyzer = threshold_analyzer if threshold_analyzer else RSIThresholdAnalyzer()
        self.rsi = RecursiveSelfImprovement()
        self.history = []

    def evaluate_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        model.eval()
        x, y = self.validation_data
        with torch.no_grad():
            outputs = model(x)
            pred_classes = outputs.argmax(dim=-1)
            accuracy = (pred_classes == y).float().mean().item()
            epistemic = torch.FloatTensor([1.0 - accuracy]*10)
            aleatoric = torch.FloatTensor([0.1 + 0.1*abs(torch.randn(1).item())]*10)
        return {'accuracy': accuracy, 'epistemic_uncertainty': epistemic, 'aleatoric_uncertainty': aleatoric}

    def apply_hypothesis(self, model: torch.nn.Module, hypothesis: ImprovementHypothesis) -> torch.nn.Module:
        new_model = copy.deepcopy(model)
        if hypothesis.strategy == ImprovementStrategy.CAPACITY_EXPANSION:
            for m in new_model.modules():
                if isinstance(m, torch.nn.Linear):
                    out_features = int(m.out_features * 1.2)
                    weight = m.weight.detach().cpu()
                    bias = m.bias.detach().cpu()
                    m.out_features = out_features
                    m.weight = torch.nn.Parameter(torch.cat([weight, torch.randn((out_features - weight.shape[0], weight.shape[1]))], dim=0))
                    m.bias = torch.nn.Parameter(torch.cat([bias, torch.randn(out_features - bias.shape[0])], dim=0))
                    break
        elif hypothesis.strategy == ImprovementStrategy.LEARNING_RATE_ADAPTATION:
            # If optimizer present, this would alter its learning rate
            pass
        # Other strategies can be extended here
        return new_model

    def execute_cycle(self) -> Dict[str, Any]:
        current_model = copy.deepcopy(self.base_model)
        state = torch.randn(128)
        before_metrics = self.evaluate_model(current_model)
        hypotheses = self.rsi.generate_hypotheses(state, top_k=1)
        hypothesis = hypotheses[0]
        modified_model = self.apply_hypothesis(current_model, hypothesis)
        after_metrics = self.evaluate_model(modified_model)
        tm = self.threshold_analyzer.measure(after_metrics['epistemic_uncertainty'], after_metrics['aleatoric_uncertainty'])
        result = self.rsi.validate_improvement(hypothesis, torch.tensor([before_metrics['accuracy']]), torch.tensor([after_metrics['accuracy']]))
        result.threshold_analysis = tm
        rollback = not (result.success and tm.status == tm.status.ABOVE)
        if not rollback:
            self.base_model = modified_model  # Only update if real improvement & threshold!
        action = 'update' if not rollback else 'rollback'
        self.history.append({'result': result, 'threshold': tm, 'action': action, 'rollback': rollback})
        return {'result': result, 'threshold': tm, 'action': action, 'rollback': rollback}

    def get_last_model(self):
        """Return the latest active model (after last cycle)."""
        return self.base_model

    def diagnostics(self) -> Dict[str, Any]:
        return {
            'cycle_count': len(self.history),
            'last_action': self.history[-1]['action'] if self.history else None,
            'last_threshold': self.history[-1]['threshold'] if self.history else None,
            'last_result_success': self.history[-1]['result'].success if self.history else None,
        }
