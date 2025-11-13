"""Auto-Consolidating SSM with Gradient Hooks.

Automates multi-timescale gradient consolidation using PyTorch backward hooks,
eliminating the need for manual accumulate_gradients() calls.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
import threading


class AutoConsolidatingSSM(nn.Module):
    """SSM with automatic gradient consolidation via backward hooks.
    
    Implements three-tier memory hierarchy:
    - Fast weights: Updated every step (standard optimizer)
    - Medium weights: Consolidated every 10 steps
    - Slow weights: Consolidated every 100 steps
    
    Consolidation happens automatically via registered backward hooks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        fast_lr: float = 1e-3,
        medium_consolidation_freq: int = 10,
        slow_consolidation_freq: int = 100,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.medium_freq = medium_consolidation_freq
        self.slow_freq = slow_consolidation_freq
        
        # Fast pathway (updated every step)
        self.fast_weights = nn.Linear(hidden_dim, state_dim)
        self.fast_bias = nn.Parameter(torch.zeros(state_dim))
        
        # Medium pathway (consolidated every 10 steps)
        self.medium_weights = nn.Linear(hidden_dim, state_dim)
        self.medium_bias = nn.Parameter(torch.zeros(state_dim))
        
        # Slow pathway (consolidated every 100 steps)
        self.slow_weights = nn.Linear(hidden_dim, state_dim)
        self.slow_bias = nn.Parameter(torch.zeros(state_dim))
        
        # Learnable interpolation coefficients
        self.interpolation = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Gradient buffers (thread-safe)
        self.grad_buffers: Dict[str, torch.Tensor] = {}
        self.step_counter = 0
        self.lock = threading.Lock()
        
        # Register backward hooks for consolidation
        self._register_consolidation_hooks()
    
    def _register_consolidation_hooks(self):
        """Register backward hooks for medium and slow parameters."""
        
        for name, param in self.named_parameters():
            if 'medium' in name:
                param.register_hook(
                    self._make_consolidation_hook(name, self.medium_freq)
                )
            elif 'slow' in name:
                param.register_hook(
                    self._make_consolidation_hook(name, self.slow_freq)
                )
    
    def _make_consolidation_hook(self, param_name: str, freq: int) -> Callable:
        """Create consolidation hook for a parameter.
        
        Args:
            param_name: Name of the parameter
            freq: Consolidation frequency (steps)
        
        Returns:
            Hook function that consolidates gradients
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            with self.lock:
                # Initialize buffer if needed
                if param_name not in self.grad_buffers:
                    self.grad_buffers[param_name] = torch.zeros_like(grad)
                
                # Accumulate gradient
                self.grad_buffers[param_name] += grad
                
                # Check if consolidation step
                if (self.step_counter + 1) % freq == 0:
                    # Normalize and return consolidated gradient
                    consolidated = self.grad_buffers[param_name] / freq
                    
                    # Add small epsilon for numerical stability
                    norm = consolidated.norm() + 1e-8
                    consolidated = consolidated / norm * grad.norm()
                    
                    # Reset buffer
                    self.grad_buffers[param_name].zero_()
                    
                    return consolidated
                else:
                    # Skip update for non-consolidation steps
                    return torch.zeros_like(grad)
        
        return hook
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-timescale processing.
        
        Args:
            x: Input tensor of shape (batch, hidden_dim)
        
        Returns:
            Output tensor of shape (batch, state_dim)
        """
        # Increment step counter
        with self.lock:
            self.step_counter += 1
        
        # Process through each pathway
        fast_out = self.fast_weights(x) + self.fast_bias
        medium_out = self.medium_weights(x) + self.medium_bias
        slow_out = self.slow_weights(x) + self.slow_bias
        
        # Learnable interpolation (normalized)
        weights = torch.softmax(self.interpolation, dim=0)
        
        output = (
            weights[0] * fast_out +
            weights[1] * medium_out +
            weights[2] * slow_out
        )
        
        return output
    
    def reset_buffers(self):
        """Reset gradient buffers (e.g., at epoch boundaries)."""
        with self.lock:
            for buffer in self.grad_buffers.values():
                buffer.zero_()
            self.step_counter = 0
