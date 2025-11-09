"""
Deep Optimizers with Nested Learning

Implements optimizers as associative memory modules that compress gradients.
Extends traditional momentum-based methods with deep neural architectures
for more expressive gradient memorization.

References:
    - Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025)
    - Section 2.3: Optimizers as Learning Modules
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class MomentumMemory(nn.Module):
    """
    Neural network-based momentum that learns to compress gradients.
    
    Traditional momentum is a linear (matrix-valued) associative memory.
    This extends it to a deep network for more expressive gradient compression.
    
    Args:
        param_shape: Shape of parameters this momentum tracks
        hidden_dim: Hidden dimension for MLP
        num_layers: Number of layers in momentum network
    """
    
    def __init__(
        self,
        param_shape: torch.Size,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.param_shape = param_shape
        self.param_numel = torch.prod(torch.tensor(param_shape)).item()
        
        # Build momentum network
        layers = []
        in_dim = self.param_numel
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.param_numel
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
            
            in_dim = out_dim
        
        self.momentum_net = nn.Sequential(*layers)
        
        # Internal momentum state
        self.register_buffer('state', torch.zeros(self.param_numel))
        
    def forward(self, grad: torch.Tensor, alpha: float = 0.9) -> torch.Tensor:
        """
        Compute momentum update using neural network.
        
        Args:
            grad: Gradient tensor (any shape matching param_shape)
            alpha: Momentum decay coefficient
            
        Returns:
            Momentum update tensor
        """
        # Flatten gradient
        grad_flat = grad.flatten()
        
        # Combine current gradient with momentum state
        combined = grad_flat + alpha * self.state
        
        # Transform through momentum network
        momentum_update = self.momentum_net(combined)
        
        # Update internal state
        self.state = momentum_update.detach()
        
        # Reshape to original parameter shape
        return momentum_update.reshape(self.param_shape)
    
    def reset(self):
        """Reset momentum state."""
        self.state.zero_()


class DeepMomentumSGD(Optimizer):
    """
    Deep Momentum Stochastic Gradient Descent.
    
    Uses neural network-based momentum that learns to compress and
    transform gradients more effectively than linear momentum.
    
    Traditional momentum:
        m_t = α*m_{t-1} - η*∇L(θ_t)
        θ_{t+1} = θ_t + m_t
    
    Deep momentum:
        m_t = MLP(∇L(θ_t), m_{t-1})
        θ_{t+1} = θ_t + m_t
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0)
        hidden_dim: Hidden dimension for momentum network
        num_layers: Number of layers in momentum network
        use_delta_rule: Use delta-rule based update for better capacity
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        use_delta_rule: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_delta_rule=use_delta_rule,
        )
        super().__init__(params, defaults)
        
        # Initialize momentum networks for each parameter
        self.momentum_nets = {}
        
    def _init_momentum_net(self, param: torch.Tensor, group: Dict) -> MomentumMemory:
        """Initialize momentum network for a parameter."""
        return MomentumMemory(
            param_shape=param.shape,
            hidden_dim=group['hidden_dim'],
            num_layers=group['num_layers'],
        ).to(param.device)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            use_delta_rule = group['use_delta_rule']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Get gradient
                grad = p.grad
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                
                # Get or create momentum network for this parameter
                param_id = id(p)
                if param_id not in self.momentum_nets:
                    self.momentum_nets[param_id] = self._init_momentum_net(p, group)
                
                momentum_net = self.momentum_nets[param_id]
                
                # Compute momentum update using neural network
                if use_delta_rule:
                    # Delta-rule: Better capacity management
                    # m_t = (α*I - ∇∇^T)*m_{t-1} - η*P*∇
                    # Simplified version without Hessian for efficiency
                    momentum_update = momentum_net(grad, alpha=momentum)
                    
                    # Apply correction term (simplified)
                    grad_norm_sq = (grad * grad).sum()
                    if grad_norm_sq > 1e-8:
                        correction = (momentum_update * grad).sum() / grad_norm_sq
                        momentum_update = momentum_update - correction * grad
                else:
                    # Standard deep momentum
                    momentum_update = momentum_net(grad, alpha=momentum)
                
                # Update parameters
                p.add_(momentum_update, alpha=-lr)
        
        return loss
    
    def reset_momentum(self):
        """Reset all momentum states."""
        for momentum_net in self.momentum_nets.values():
            momentum_net.reset()
    
    def state_dict(self):
        """Return state dict including momentum networks."""
        state = super().state_dict()
        state['momentum_nets'] = {
            k: net.state_dict() for k, net in self.momentum_nets.items()
        }
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dict including momentum networks."""
        momentum_nets_state = state_dict.pop('momentum_nets', {})
        super().load_state_dict(state_dict)
        
        for k, net_state in momentum_nets_state.items():
            if k in self.momentum_nets:
                self.momentum_nets[k].load_state_dict(net_state)


class AdaptiveDeepMomentum(Optimizer):
    """
    Adaptive Deep Momentum with preconditioning.
    
    Extends Deep Momentum with adaptive per-parameter learning rates,
    similar to Adam but with neural network-based momentum.
    
    Update rule:
        m_t = MLP(∇L(θ_t), m_{t-1})
        v_t = β*v_{t-1} + (1-β)*∇L(θ_t)²
        θ_{t+1} = θ_t - η * m_t / (√v_t + ε)
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        betas: Coefficients for momentum and variance
        eps: Term added to denominator for numerical stability
        weight_decay: Weight decay coefficient
        hidden_dim: Hidden dimension for momentum network
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        hidden_dim: int = 128,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
        )
        super().__init__(params, defaults)
        
        self.momentum_nets = {}
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    # Initialize momentum network
                    param_id = id(p)
                    self.momentum_nets[param_id] = MomentumMemory(
                        param_shape=p.shape,
                        hidden_dim=group['hidden_dim'],
                        num_layers=2,
                    ).to(p.device)
                
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                
                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute bias-corrected second moment estimate
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Get deep momentum
                param_id = id(p)
                momentum_net = self.momentum_nets[param_id]
                exp_avg = momentum_net(grad, alpha=beta1)
                
                # Compute step size with bias correction
                step_size = group['lr'] / bias_correction2
                
                # Compute denominator
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Update parameters
                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
