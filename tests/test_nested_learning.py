"""
Tests for Nested Learning Components

Comprehensive test suite for:
- Nested State Space Model (NestedSSM)
- Continuum Memory System (CMS)
- Deep Momentum Optimizers
"""

import pytest
import torch
import torch.nn as nn
from dhc_ssm.core.nested_ssm import (
    NestedStateSpaceModel,
    ContinuumMemoryBlock,
    NestedSSMConfig,
)
from dhc_ssm.training.deep_optimizer import (
    DeepMomentumSGD,
    AdaptiveDeepMomentum,
    MomentumMemory,
)


class TestContinuumMemoryBlock:
    """Test individual memory blocks in CMS."""
    
    def test_initialization(self):
        """Test memory block initialization."""
        block = ContinuumMemoryBlock(
            input_dim=64,
            hidden_dim=128,
            update_frequency=10,
        )
        
        assert block.input_dim == 64
        assert block.hidden_dim == 128
        assert block.update_frequency == 10
        assert block.step_counter.item() == 0
    
    def test_forward_with_update(self):
        """Test forward pass when update should occur."""
        block = ContinuumMemoryBlock(
            input_dim=64,
            hidden_dim=128,
            update_frequency=5,
        )
        
        x = torch.randn(32, 64)
        
        # First call should update (step 0 % 5 == 0)
        output, was_updated = block(x)
        assert output.shape == (32, 64)
        assert was_updated == True
        assert block.step_counter.item() == 1
    
    def test_frequency_based_updates(self):
        """Test that updates occur at correct frequency."""
        block = ContinuumMemoryBlock(
            input_dim=64,
            hidden_dim=128,
            update_frequency=10,
        )
        
        x = torch.randn(32, 64)
        update_counts = 0
        
        for i in range(30):
            _, was_updated = block(x)
            if was_updated:
                update_counts += 1
        
        # Should update at steps: 0, 10, 20 = 3 times
        assert update_counts == 3
        assert block.step_counter.item() == 30
    
    def test_reset_state(self):
        """Test state reset functionality."""
        block = ContinuumMemoryBlock(input_dim=64, hidden_dim=128)
        
        x = torch.randn(32, 64)
        block(x)
        block(x)
        
        assert block.step_counter.item() == 2
        
        block.reset_state()
        assert block.step_counter.item() == 0
        assert torch.all(block.compressed_context == 0)


class TestNestedStateSpaceModel:
    """Test full Nested SSM with CMS."""
    
    def test_initialization(self):
        """Test model initialization."""
        config = NestedSSMConfig(
            hidden_dim=128,
            state_dim=32,
            fast_freq=1,
            medium_freq=10,
            slow_freq=100,
        )
        
        model = NestedStateSpaceModel(
            hidden_dim=config.hidden_dim,
            state_dim=config.state_dim,
            fast_freq=config.fast_freq,
            medium_freq=config.medium_freq,
            slow_freq=config.slow_freq,
        )
        
        assert model.hidden_dim == 128
        assert model.state_dim == 32
        assert model.global_step.item() == 0
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        model = NestedStateSpaceModel(hidden_dim=128, state_dim=32)
        x = torch.randn(16, 128)
        
        output = model(x)
        
        assert output.shape == (16, 128)
        assert model.global_step.item() == 1
    
    def test_forward_with_diagnostics(self):
        """Test diagnostic output."""
        model = NestedStateSpaceModel(
            hidden_dim=128,
            state_dim=32,
            fast_freq=1,
            medium_freq=5,
            slow_freq=10,
        )
        
        x = torch.randn(16, 128)
        output, diagnostics = model(x, return_diagnostics=True)
        
        assert 'global_step' in diagnostics
        assert 'fast_updated' in diagnostics
        assert 'medium_updated' in diagnostics
        assert 'slow_updated' in diagnostics
        assert 'memory_weights' in diagnostics
        
        assert diagnostics['fast_updated'] == True
        assert len(diagnostics['memory_weights']) == 3
    
    def test_multi_level_updates(self):
        """Test that different memory levels update at correct frequencies."""
        model = NestedStateSpaceModel(
            hidden_dim=128,
            state_dim=32,
            fast_freq=1,
            medium_freq=10,
            slow_freq=100,
        )
        
        x = torch.randn(16, 128)
        
        # Run for 150 steps and track updates
        fast_updates = 0
        medium_updates = 0
        slow_updates = 0
        
        for i in range(150):
            _, diag = model(x, return_diagnostics=True)
            if diag['fast_updated']:
                fast_updates += 1
            if diag['medium_updated']:
                medium_updates += 1
            if diag['slow_updated']:
                slow_updates += 1
        
        # Fast should update every step
        assert fast_updates == 150
        # Medium should update every 10 steps
        assert medium_updates == 15
        # Slow should update every 100 steps
        assert slow_updates == 2  # steps 0 and 100
    
    def test_memory_utilization(self):
        """Test memory utilization statistics."""
        model = NestedStateSpaceModel(
            hidden_dim=128,
            fast_freq=1,
            medium_freq=10,
            slow_freq=100,
        )
        
        x = torch.randn(16, 128)
        
        # Run for some steps
        for _ in range(50):
            model(x)
        
        stats = model.get_memory_utilization()
        
        assert stats['total_steps'] == 50
        assert stats['fast_updates'] == 50
        assert stats['medium_updates'] == 5
        assert stats['slow_updates'] == 1
    
    def test_reset_state(self):
        """Test state reset across all memory levels."""
        model = NestedStateSpaceModel(hidden_dim=128)
        x = torch.randn(16, 128)
        
        for _ in range(20):
            model(x)
        
        assert model.global_step.item() == 20
        
        model.reset_state()
        
        assert model.global_step.item() == 0
        assert model.fast_memory.step_counter.item() == 0
        assert model.medium_memory.step_counter.item() == 0
        assert model.slow_memory.step_counter.item() == 0


class TestMomentumMemory:
    """Test neural network-based momentum."""
    
    def test_initialization(self):
        """Test momentum memory initialization."""
        param_shape = (64, 32)
        momentum_mem = MomentumMemory(
            param_shape=param_shape,
            hidden_dim=128,
            num_layers=2,
        )
        
        assert momentum_mem.param_shape == param_shape
        assert momentum_mem.param_numel == 64 * 32
    
    def test_forward(self):
        """Test momentum computation."""
        param_shape = (64, 32)
        momentum_mem = MomentumMemory(param_shape=param_shape)
        
        grad = torch.randn(param_shape)
        momentum_update = momentum_mem(grad, alpha=0.9)
        
        assert momentum_update.shape == param_shape
    
    def test_state_accumulation(self):
        """Test that momentum state accumulates over time."""
        param_shape = (32, 16)
        momentum_mem = MomentumMemory(param_shape=param_shape)
        
        initial_state = momentum_mem.state.clone()
        
        grad = torch.randn(param_shape)
        momentum_mem(grad, alpha=0.9)
        
        # State should have changed
        assert not torch.allclose(momentum_mem.state, initial_state)
    
    def test_reset(self):
        """Test momentum reset."""
        momentum_mem = MomentumMemory(param_shape=(32, 16))
        
        grad = torch.randn(32, 16)
        momentum_mem(grad)
        
        assert not torch.all(momentum_mem.state == 0)
        
        momentum_mem.reset()
        assert torch.all(momentum_mem.state == 0)


class TestDeepMomentumSGD:
    """Test Deep Momentum SGD optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = nn.Linear(10, 5)
        optimizer = DeepMomentumSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            hidden_dim=64,
        )
        
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_step(self):
        """Test optimization step."""
        model = nn.Linear(10, 5)
        optimizer = DeepMomentumSGD(model.parameters(), lr=0.01)
        
        # Forward and backward
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        
        # Store initial params
        initial_weight = model.weight.data.clone()
        
        # Optimization step
        optimizer.step()
        
        # Parameters should have changed
        assert not torch.allclose(model.weight.data, initial_weight)
    
    def test_training_convergence(self):
        """Test that optimizer can minimize a simple function."""
        # Simple quadratic: f(x) = (x - 3)^2
        x = nn.Parameter(torch.tensor([0.0]))
        optimizer = DeepMomentumSGD([x], lr=0.1, hidden_dim=32, num_layers=1)
        
        initial_x = x.item()
        
        for _ in range(50):
            loss = (x - 3.0) ** 2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Should move closer to optimal x=3
        final_x = x.item()
        assert abs(final_x - 3.0) < abs(initial_x - 3.0)
    
    def test_reset_momentum(self):
        """Test momentum reset functionality."""
        model = nn.Linear(10, 5)
        optimizer = DeepMomentumSGD(model.parameters(), lr=0.01)
        
        # Do a step to initialize momentum
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        
        assert len(optimizer.momentum_nets) > 0
        
        optimizer.reset_momentum()
        
        # Check that states are reset
        for momentum_net in optimizer.momentum_nets.values():
            assert torch.all(momentum_net.state == 0)


class TestAdaptiveDeepMomentum:
    """Test Adaptive Deep Momentum optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = nn.Linear(10, 5)
        optimizer = AdaptiveDeepMomentum(
            model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
        )
        
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['betas'] == (0.9, 0.999)
    
    def test_step(self):
        """Test optimization step."""
        model = nn.Linear(10, 5)
        optimizer = AdaptiveDeepMomentum(model.parameters(), lr=0.001)
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)
        
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        
        initial_weight = model.weight.data.clone()
        optimizer.step()
        
        assert not torch.allclose(model.weight.data, initial_weight)
    
    def test_training_convergence(self):
        """Test convergence on simple optimization problem."""
        x = nn.Parameter(torch.tensor([5.0]))
        optimizer = AdaptiveDeepMomentum([x], lr=0.1, hidden_dim=32)
        
        losses = []
        for _ in range(100):
            loss = (x - 2.0) ** 2
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Loss should decrease
        assert losses[-1] < losses[0]
        # Should converge close to x=2
        assert abs(x.item() - 2.0) < 0.5


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.slow
    def test_nested_ssm_with_deep_optimizer(self):
        """Test NestedSSM trained with DeepMomentumSGD."""
        model = NestedStateSpaceModel(
            hidden_dim=64,
            state_dim=32,
            fast_freq=1,
            medium_freq=5,
            slow_freq=20,
        )
        
        optimizer = DeepMomentumSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            hidden_dim=32,
            num_layers=2,
        )
        
        # Simple training loop
        x = torch.randn(16, 64)
        target = torch.randn(16, 64)
        
        initial_loss = None
        
        for i in range(20):
            output = model(x)
            loss = nn.functional.mse_loss(output, target)
            
            if i == 0:
                initial_loss = loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        final_loss = loss.item()
        
        # Loss should decrease
        assert final_loss < initial_loss
    
    def test_memory_consolidation_pattern(self):
        """Test that memory consolidation follows expected pattern."""
        model = NestedStateSpaceModel(
            hidden_dim=128,
            fast_freq=1,
            medium_freq=10,
            slow_freq=100,
        )
        
        x = torch.randn(8, 128)
        
        # Track which levels update at each step
        update_pattern = {
            'fast': [],
            'medium': [],
            'slow': [],
        }
        
        for i in range(120):
            _, diag = model(x, return_diagnostics=True)
            
            if diag['fast_updated']:
                update_pattern['fast'].append(i)
            if diag['medium_updated']:
                update_pattern['medium'].append(i)
            if diag['slow_updated']:
                update_pattern['slow'].append(i)
        
        # Fast should update every step
        assert len(update_pattern['fast']) == 120
        
        # Medium should update at: 0, 10, 20, ..., 110
        assert len(update_pattern['medium']) == 12
        assert update_pattern['medium'][0] == 0
        assert update_pattern['medium'][1] == 10
        
        # Slow should update at: 0, 100
        assert len(update_pattern['slow']) == 2
        assert update_pattern['slow'] == [0, 100]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
