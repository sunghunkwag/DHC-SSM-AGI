"""
Nested Learning State Space Model (NL-SSM)

Implements multi-time-scale updates inspired by Google's Nested Learning paradigm.
Each level operates at different frequencies, enabling hierarchical memory consolidation
similar to human brain's synaptic and systems consolidation.

References:
    - Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025)
    - https://abehrouz.github.io/files/NL.pdf
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class ContinuumMemoryBlock(nn.Module):
    """
    Single memory block in Continuum Memory System.
    
    Each block has its own update frequency and learns to compress
    its context into parameters at its designated time scale.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        update_frequency: Number of steps between updates (1 = every step)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        update_frequency: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.update_frequency = update_frequency
        
        # Memory transformation network
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Accumulated gradient buffer for infrequent updates
        self.register_buffer('grad_accumulator', torch.zeros(1))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))
        
        # Context compression state
        self.register_buffer('compressed_context', torch.zeros(input_dim))
        
    def forward(
        self,
        x: torch.Tensor,
        force_update: bool = False,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Forward pass with conditional update based on frequency.
        
        Args:
            x: Input tensor [batch, features]
            force_update: Force update regardless of frequency
            
        Returns:
            Tuple of (output tensor, was_updated flag)
        """
        batch_size = x.size(0)
        
        # Determine if this level should update
        should_update = (
            force_update or 
            (self.step_counter % self.update_frequency == 0)
        )
        
        if should_update:
            # Perform full transformation
            output = self.transform(x)
            
            # Update compressed context (batch mean for simplicity)
            with torch.no_grad():
                self.compressed_context = x.mean(dim=0)
        else:
            # Use previous compressed context
            output = x + 0.1 * self.compressed_context.unsqueeze(0).expand_as(x)
        
        # Increment step counter
        self.step_counter += 1
        
        return output, should_update
    
    def reset_state(self):
        """Reset internal state for new sequence."""
        self.compressed_context.zero_()
        self.step_counter.zero_()
        self.grad_accumulator.zero_()


class NestedStateSpaceModel(nn.Module):
    """
    Nested State Space Model with Continuum Memory System.
    
    Implements hierarchical memory with three levels:
    - Fast memory: Updates every step (immediate context)
    - Medium memory: Updates every C steps (recent patterns)
    - Slow memory: Updates every C² steps (long-term knowledge)
    
    This design enables efficient continual learning by consolidating
    information at multiple time scales, similar to hippocampal-cortical
    memory consolidation in the brain.
    
    Args:
        hidden_dim: Dimension of hidden representations
        state_dim: Dimension of SSM state space
        fast_freq: Update frequency for fast memory (default: 1)
        medium_freq: Update frequency for medium memory (default: 10)
        slow_freq: Update frequency for slow memory (default: 100)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        state_dim: int = 64,
        fast_freq: int = 1,
        medium_freq: int = 10,
        slow_freq: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Traditional SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim) * 0.01)
        self.D = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        
        # Continuum Memory System (CMS) with three levels
        self.fast_memory = ContinuumMemoryBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            update_frequency=fast_freq,
            dropout=dropout,
        )
        
        self.medium_memory = ContinuumMemoryBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            update_frequency=medium_freq,
            dropout=dropout,
        )
        
        self.slow_memory = ContinuumMemoryBlock(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            update_frequency=slow_freq,
            dropout=dropout,
        )
        
        # Memory fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Learnable weights for memory level importance
        self.memory_weights = nn.Parameter(torch.ones(3) / 3)
        
        self.register_buffer('global_step', torch.zeros(1, dtype=torch.long))
        
    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Forward pass with nested memory updates.
        
        Args:
            x: Input tensor [batch, hidden_dim]
            return_diagnostics: Whether to return update diagnostics
            
        Returns:
            Output tensor or tuple of (output, diagnostics)
        """
        batch_size = x.size(0)
        
        # Core SSM operation (O(n) complexity maintained)
        state = torch.zeros(batch_size, self.state_dim, device=x.device)
        state = torch.tanh(state @ self.A.t() + x @ self.B.t())
        ssm_output = state @ self.C.t() + x @ self.D.t()
        
        # Multi-level memory processing
        fast_out, fast_updated = self.fast_memory(ssm_output)
        medium_out, medium_updated = self.medium_memory(ssm_output)
        slow_out, slow_updated = self.slow_memory(ssm_output)
        
        # Normalize memory weights to sum to 1
        weights = torch.softmax(self.memory_weights, dim=0)
        
        # Weighted combination of memory levels
        combined = torch.cat([
            fast_out * weights[0],
            medium_out * weights[1],
            slow_out * weights[2],
        ], dim=-1)
        
        # Fuse all memory levels
        output = self.fusion(combined)
        
        # Add residual connection from original input
        output = output + ssm_output
        
        self.global_step += 1
        
        if return_diagnostics:
            diagnostics = {
                'global_step': self.global_step.item(),
                'fast_updated': fast_updated,
                'medium_updated': medium_updated,
                'slow_updated': slow_updated,
                'memory_weights': weights.detach().cpu().tolist(),
                'fast_freq': self.fast_memory.update_frequency,
                'medium_freq': self.medium_memory.update_frequency,
                'slow_freq': self.slow_memory.update_frequency,
            }
            return output, diagnostics
        
        return output
    
    def reset_state(self):
        """Reset all memory states for new sequence."""
        self.fast_memory.reset_state()
        self.medium_memory.reset_state()
        self.slow_memory.reset_state()
        self.global_step.zero_()
    
    def get_memory_utilization(self) -> Dict[str, float]:
        """
        Get utilization statistics for each memory level.
        
        Returns:
            Dictionary with update counts and frequencies
        """
        total_steps = self.global_step.item()
        if total_steps == 0:
            return {
                'fast_updates': 0,
                'medium_updates': 0,
                'slow_updates': 0,
                'total_steps': 0,
            }
        
        return {
            'fast_updates': total_steps // self.fast_memory.update_frequency,
            'medium_updates': total_steps // self.medium_memory.update_frequency,
            'slow_updates': total_steps // self.slow_memory.update_frequency,
            'total_steps': total_steps,
            'fast_utilization': 1.0,
            'medium_utilization': self.medium_memory.update_frequency / total_steps,
            'slow_utilization': self.slow_memory.update_frequency / total_steps,
        }


class NestedSSMConfig:
    """
    Configuration for Nested State Space Model.
    
    Attributes:
        hidden_dim: Dimension of hidden representations
        state_dim: Dimension of SSM state space
        fast_freq: Update frequency for fast memory
        medium_freq: Update frequency for medium memory
        slow_freq: Update frequency for slow memory
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        state_dim: int = 64,
        fast_freq: int = 1,
        medium_freq: int = 10,
        slow_freq: int = 100,
        dropout: float = 0.1,
    ):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.fast_freq = fast_freq
        self.medium_freq = medium_freq
        self.slow_freq = slow_freq
        self.dropout = dropout
        
        # Validate frequencies
        assert fast_freq <= medium_freq <= slow_freq, \
            "Frequencies must be in ascending order: fast <= medium <= slow"
        assert fast_freq > 0, "Fast frequency must be positive"
