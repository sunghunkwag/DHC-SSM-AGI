import torch
from dhc_ssm.agi.threshold_analyzer import RSIThresholdAnalyzer
from dhc_ssm.agi.self_improvement import RecursiveSelfImprovement, ImprovementHypothesis
from dhc_ssm.agi.self_improvement_executor import SelfImprovementExecutor

def test_threshold_analysis():
    analyzer = RSIThresholdAnalyzer()
    for i in range(20):
        epistemic = torch.tensor([0.9 - i*0.04])
        aleatoric = torch.tensor([0.2])
        gamma = analyzer.compute_gamma(epistemic, aleatoric)
        analyzer.measure(epistemic, aleatoric)
    assert analyzer.measurements[-1].gamma > analyzer.measurements[0].gamma

def test_self_improvement_cycle():
    class DummyModel:
        def __call__(self, x):
            return torch.randn(x.shape[0], 10)
    model = DummyModel()
    val_data = (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
    executor = SelfImprovementExecutor(model, val_data)
    result = executor.execute_cycle()
    assert result['threshold'] is not None
    assert result['action'] in ('update', 'rollback')

def test_learning_success_rate():
    rsi = RecursiveSelfImprovement()
    for _ in range(10):
        state = torch.randn(128)
        h = rsi.generate_hypotheses(state, top_k=1)[0]
        res = rsi.validate_improvement(h, torch.tensor([0.7]), torch.tensor([0.9]))
    rate = rsi.get_success_rate()
    assert rate > 0.0
