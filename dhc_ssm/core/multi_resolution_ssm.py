"""Multi-Resolution State Space Model with Wavelet Decomposition.

Implements true multi-scale SSM processing by decomposing inputs into
different frequency bands and processing each with specialized SSMs.

Based on:
- MS-SSM: Multi-Scale State Space Model (2025)
- Multi-Scale VMamba research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import pywt
import numpy as np


class WaveletDecomposition(nn.Module):
    """Wavelet decomposition for multi-resolution analysis.
    
    Decomposes input signal into high, medium, and low frequency components
    using Discrete Wavelet Transform (DWT).
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 2):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose input into frequency bands.
        
        Args:
            x: Input tensor (batch, seq_len, channels)
        
        Returns:
            high_freq: Detail coefficients (high frequency)
            mid_freq: Mid-level approximation
            low_freq: Coarse approximation (low frequency)
        """
        batch, seq_len, channels = x.shape
        device = x.device
        
        # Process each channel independently
        high_list, mid_list, low_list = [], [], []
        
        for b in range(batch):
            for c in range(channels):
                signal = x[b, :, c].cpu().numpy()
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                
                # coeffs = [cAn, cDn, cDn-1, ..., cD1]
                # cAn: coarsest approximation (low freq)
                # cDn: detail at level n (mid freq)
                # cD1: finest detail (high freq)
                
                # Safe indexing with fallback
                low_list.append(torch.tensor(coeffs[0], dtype=torch.float32))
                
                # Handle cases where there might not be enough levels
                if len(coeffs) > 1:
                    mid_list.append(torch.tensor(coeffs[1], dtype=torch.float32))
                else:
                    mid_list.append(torch.tensor(coeffs[0], dtype=torch.float32))
                
                if len(coeffs) > 2:
                    high_list.append(torch.tensor(coeffs[2], dtype=torch.float32))
                elif len(coeffs) > 1:
                    high_list.append(torch.tensor(coeffs[1], dtype=torch.float32))
                else:
                    high_list.append(torch.tensor(coeffs[0], dtype=torch.float32))
        
        # Stack and move to device
        # Note: Different resolutions have different lengths
        # Pad to max length for batching
        max_len_high = max(t.shape[0] for t in high_list)
        max_len_mid = max(t.shape[0] for t in mid_list)
        max_len_low = max(t.shape[0] for t in low_list)
        
        high_freq = self._pad_and_stack(high_list, max_len_high, batch, channels).to(device)
        mid_freq = self._pad_and_stack(mid_list, max_len_mid, batch, channels).to(device)
        low_freq = self._pad_and_stack(low_list, max_len_low, batch, channels).to(device)
        
        return high_freq, mid_freq, low_freq
    
    def _pad_and_stack(self, coeff_list: List[torch.Tensor], max_len: int, batch: int, channels: int) -> torch.Tensor:
        """Pad coefficients to max length and stack."""
        padded = []
        for i in range(batch * channels):
            coeff = coeff_list[i]
            if coeff.shape[0] < max_len:
                padding = torch.zeros(max_len - coeff.shape[0])
                coeff = torch.cat([coeff, padding])
            padded.append(coeff)
        
        stacked = torch.stack(padded).reshape(batch, channels, max_len)
        return stacked.permute(0, 2, 1)  # (batch, seq_len, channels)


class ScaleSpecificSSM(nn.Module):
    """SSM specialized for a specific frequency scale."""
    
    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Scale-specific parameters
        self.A = nn.Parameter(-torch.ones(state_dim, state_dim))  # Negative for stability
        self.B = nn.Linear(hidden_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, hidden_dim, bias=False)
        self.D = nn.Parameter(torch.zeros(hidden_dim))  # Skip connection
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through scale-specific SSM.
        
        Args:
            x: Input (batch, seq_len, hidden_dim)
        
        Returns:
            output: Processed signal (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Initialize state
        state = torch.zeros(batch, self.state_dim, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, hidden_dim)
            
            # SSM update: s_{t+1} = A*s_t + B*x_t
            state = torch.matmul(state, self.A.T) + self.B(x_t)
            
            # Output: y_t = C*s_t + D*x_t
            y_t = self.C(state) + self.D * x_t
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        return self.norm(output)


class MultiResolutionSSM(nn.Module):
    """Multi-Resolution SSM with wavelet decomposition and adaptive gating.
    
    Architecture:
        Input → Wavelet Decomp → [High SSM, Mid SSM, Low SSM] → Adaptive Gating → Output
    """
    
    def __init__(
        self,
        hidden_dim: int,
        state_dim: int,
        wavelet: str = 'db4',
        decomposition_level: int = 2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        
        # Wavelet decomposition
        self.wavelet_decomp = WaveletDecomposition(wavelet, decomposition_level)
        
        # Scale-specific SSMs
        self.ssm_high = ScaleSpecificSSM(hidden_dim, state_dim)  # High frequency
        self.ssm_mid = ScaleSpecificSSM(hidden_dim, state_dim)   # Medium frequency
        self.ssm_low = ScaleSpecificSSM(hidden_dim, state_dim)   # Low frequency
        
        # Adaptive scale gating network
        # Learns which scales are important for current input
        self.scale_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 scales
            nn.Softmax(dim=-1)
        )
        
        # Upsampling layers to match original resolution
        self.upsample_high = nn.Linear(hidden_dim, hidden_dim)
        self.upsample_mid = nn.Linear(hidden_dim, hidden_dim)
        self.upsample_low = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-resolution forward pass.
        
        Args:
            x: Input (batch, seq_len, hidden_dim)
        
        Returns:
            output: Multi-scale processed output (batch, seq_len, hidden_dim)
        """
        batch, seq_len, hidden_dim = x.shape
        
        # Wavelet decomposition
        high_freq, mid_freq, low_freq = self.wavelet_decomp(x)
        
        # Process each scale with specialized SSM
        out_high = self.ssm_high(high_freq)
        out_mid = self.ssm_mid(mid_freq)
        out_low = self.ssm_low(low_freq)
        
        # Upsample to original resolution (interpolation)
        out_high = F.interpolate(out_high.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False).permute(0, 2, 1)
        out_mid = F.interpolate(out_mid.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False).permute(0, 2, 1)
        out_low = F.interpolate(out_low.permute(0, 2, 1), size=seq_len, mode='linear', align_corners=False).permute(0, 2, 1)
        
        out_high = self.upsample_high(out_high)
        out_mid = self.upsample_mid(out_mid)
        out_low = self.upsample_low(out_low)
        
        # Adaptive scale gating
        # Use mean-pooled input as query for attention
        x_pooled = x.mean(dim=1)  # (batch, hidden_dim)
        scale_weights = self.scale_attention(x_pooled)  # (batch, 3)
        
        # Weighted combination of scales
        output = (
            scale_weights[:, 0:1, None] * out_high +
            scale_weights[:, 1:2, None] * out_mid +
            scale_weights[:, 2:3, None] * out_low
        )
        
        return output
