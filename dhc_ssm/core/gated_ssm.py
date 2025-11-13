"""Gated State Space Model for In-Context Learning.

Implements multiplicative input/output gating to enable SSMs to perform
gradient-based in-context learning, based on:

"State-space models can learn in-context by gradient descent" (2024)
arXiv:2410.xxxxx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedS6Layer(nn.Module):
    """Gated S6/Mamba layer with in-context learning capabilities.
    
    Key innovation: Multiplicative input and output gates enable SSMs
    to reproduce gradient descent within their forward pass.
    
    Architecture:
        x → Input Gate → SSM → Output Gate → output
                ↓              ↓
              g_in           g_out
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        dt_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # S6 core parameters (simplified Mamba-style)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) / state_dim)
        self.B = nn.Linear(hidden_dim, state_dim)
        self.C = nn.Linear(state_dim, hidden_dim)
        
        # Delta (timestep) projection
        self.dt_proj = nn.Linear(dt_rank, state_dim)
        self.dt_rank_proj = nn.Linear(hidden_dim, dt_rank)
        
        # Input gating (controls what information enters SSM)
        self.input_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        
        # Output gating (controls what information leaves SSM)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )
        
        # Residual projection
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for stability."""
        # A matrix: negative diagonal for stability
        with torch.no_grad():
            self.A.data = -torch.eye(self.state_dim) - 0.1 * torch.randn(self.state_dim, self.state_dim)
        
        # Small initialization for dt projection
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.01)
        nn.init.zeros_(self.dt_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gated SSM.
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            state: Previous SSM state (batch, state_dim) or None
        
        Returns:
            output: Gated output (batch, seq_len, hidden_dim)
            new_state: Updated SSM state (batch, state_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize state if needed
        if state is None:
            state = torch.zeros(batch, self.state_dim, device=x.device)
        
        # === Input Gating ===
        # Determines which input features are relevant for current context
        g_in = self.input_gate(x)  # (batch, seq_len, hidden_dim)
        x_gated = x * g_in
        
        # === SSM Processing ===
        outputs = []
        current_state = state
        
        for t in range(seq_len):
            x_t = x_gated[:, t, :]  # (batch, hidden_dim)
            
            # Compute adaptive timestep
            dt_rank = self.dt_rank_proj(x_t)  # (batch, dt_rank)
            dt = F.softplus(self.dt_proj(dt_rank))  # (batch, state_dim)
            
            # SSM state update: s_{t+1} = A*s_t + B*x_t
            # Discretized with learned timestep dt
            A_discrete = torch.exp(dt.unsqueeze(-1) * self.A)  # (batch, state_dim, state_dim)
            B_t = self.B(x_t)  # (batch, state_dim)
            
            # State update
            current_state = torch.einsum('bsd,bd->bs', A_discrete, current_state) + B_t
            
            # Output: y_t = C*s_t
            y_t = self.C(current_state)  # (batch, hidden_dim)
            outputs.append(y_t)
        
        ssm_output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        
        # === Output Gating ===
        # Determines which SSM outputs are relevant for prediction
        g_out = self.output_gate(ssm_output)
        gated_output = ssm_output * g_out
        
        # === Residual Connection ===
        residual = self.residual_proj(x)
        output = gated_output + residual
        
        return output, current_state


class MultiLayerGatedSSM(nn.Module):
    """Stack of gated SSM layers for deep in-context learning."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        state_dim: int,
        dt_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            GatedS6Layer(hidden_dim, state_dim, dt_rank, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Forward through all layers.
        
        Args:
            x: Input (batch, seq_len, hidden_dim)
            states: List of previous states for each layer
        
        Returns:
            output: Final output (batch, seq_len, hidden_dim)
            new_states: List of updated states
        """
        if states is None:
            states = [None] * len(self.layers)
        
        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            x = self.layer_norm(x)
            new_states.append(new_state)
        
        return x, new_states
