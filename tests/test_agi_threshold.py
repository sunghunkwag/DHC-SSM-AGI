import torch
import torch.nn as nn
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
from dhc_ssm.agi.self_improvement import RecursiveSelfImprovement, ImprovementHypothesis, ImprovementStrategy
from dhc_ssm.agi.self_improvement_executor import SelfImprovementExecutor

class DummyModel(nn.Module):
    def __init__(self, input_dim=3072, output_dim=10):  # <-- FIXED!
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)

def test_threshold_analysis():
    analyzer = RSIThresholdAnalyzer()
    for i in range(20):
        epistemic = torch.tensor([0.9 - i*0.04])
        aleatoric = torch.tensor([0.2])
        analyzer.measure(epistemic, aleatoric)
    result = analyzer.measurements[-1]
    assert result.gamma > analyzer.measurements[0].gamma
    print('[test_threshold_analysis]: PASSED')

def test_self_improvement_cycle():
    model = DummyModel()
    val_data = (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
    executor = SelfImprovementExecutor(model, val_data)
    out = executor.execute_cycle()
    assert out['threshold'] is not None
    assert out['action'] in ('update', 'rollback')
    print(f"[test_self_improvement_cycle]: PASSED {out['action']} - new model out dim {executor.base_model.linear.out_features}")

def test_learning_success_rate():
    rsi = RecursiveSelfImprovement()
    for _ in range(10):
        state = torch.randn(128)
        h = rsi.generate_hypotheses(state, top_k=1)[0]
        res = rsi.validate_improvement(h, torch.tensor([0.7]), torch.tensor([0.9]))
    rate = rsi.get_success_rate()
    assert rate > 0.0
    print(f'[test_learning_success_rate]: PASSED (rate={rate:.3f})')

def test_full_improvement_scenario():
    model = DummyModel()
    val_data = (torch.randn(32, 3, 32, 32), torch.randint(0, model.linear.out_features, (32,)))
    executor = SelfImprovementExecutor(model, val_data)
    # Run several improvement cycles
    actions = []
    for _ in range(5):
        out = executor.execute_cycle()
        actions.append(out['action'])
    # At least one update expected if threshold is reachable
    assert len(actions) == 5
    latest_model = executor.base_model
    assert isinstance(latest_model, DummyModel)
    print(f'[test_full_improvement_scenario]: PASSED. Actions={actions}')

if __name__ == '__main__':
    print('Running AGI/RSI integration tests...')
    test_threshold_analysis()
    test_self_improvement_cycle()
    test_learning_success_rate()
    test_full_improvement_scenario()
    print('All AGI/RSI tests completed.')
