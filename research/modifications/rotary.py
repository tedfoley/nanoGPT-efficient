"""Rotary Embeddings implementation.

This module implements Rotary Position Embeddings (RoPE) as described in the paper:
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
https://arxiv.org/abs/2104.09864

Key benefits:
- Better handling of sequence length extrapolation
- More efficient than absolute positional embeddings
- Improved relative position modeling
"""

import torch
import torch.nn as nn
import math

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings.
    """
    def __init__(self, dim, base=10000, precision=torch.float):
        """
        Args:
            dim: Dimension of the model
            base: Base for the angle calculations
            precision: Torch precision type
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.precision = precision

    def forward(self, x, seq_len=None):
        """
        Args:
            x: Input tensor (batch, seq_len, head_dim)
            seq_len: Sequence length
        Returns:
            Tensor with position embeddings added
        """
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type(self.precision)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self._apply_rotary_pos_emb(x, self.cos_cached, self.sin_cached)

    @staticmethod
    def _apply_rotary_pos_emb(x, cos, sin):
        # Apply rotary position embeddings
        cos = cos[:, :, :x.shape[1], :]
        sin = sin[:, :, :x.shape[1], :]
        return (x * cos) + (rotate_half(x) * sin)

class RotaryTransformerLayer(nn.Module):
    """
    A transformer layer using rotary embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.n_embd
        self.n_head = config.n_head
        self.rotary_emb = RotaryEmbedding(
            dim=self.dim // self.n_head,
            base=10000
        )
        
        # Standard transformer components
        self.attention = nn.MultiheadAttention(
            self.dim,
            self.n_head,
            dropout=config.dropout,
            bias=config.bias
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, seq_len, dim)
        Returns:
            Output tensor with same shape
        """
        # Apply rotary embeddings to queries and keys
        batch_size, seq_len, _ = x.shape
        
        # Reshape for rotary embeddings
        x_rotary = x.view(batch_size, seq_len, self.n_head, -1)
        x_rotary = self.rotary_emb(x_rotary, seq_len=seq_len)
        x_rotary = x_rotary.view(batch_size, seq_len, -1)
        
        # Self-attention block
        attn_out = self.attention(
            x_rotary, x_rotary, x,
            need_weights=False
        )[0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # MLP block
        x = x + self.mlp(x)
        x = self.norm2(x)
        
        return x

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """
    Apply rotary position embeddings to queries and keys.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine of angles
        sin: Sine of angles
        position_ids: Position indices
        
    Returns:
        Tuple of transformed (q, k)
    """
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed