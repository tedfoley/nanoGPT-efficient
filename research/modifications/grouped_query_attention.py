"""Grouped-Query Attention implementation.

This module implements Grouped-Query Attention (GQA) as described in:
"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
https://arxiv.org/abs/2305.13245

Key benefits:
- Reduced memory footprint compared to MHA
- Better inference efficiency
- Maintains model quality with fewer parameters
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class GroupedQueryAttention(nn.Module):
    """
    Implementation of Grouped-Query Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        assert self.n_head % self.n_kv_heads == 0, \
            f"Number of heads ({self.n_head}) must be divisible by number of KV heads ({self.n_kv_heads})"
        
        self.n_queries_per_kv = self.n_head // self.n_kv_heads
        self.head_dim = config.n_embd // config.n_head
        
        # Linear projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd // self.n_queries_per_kv, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd // self.n_queries_per_kv, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Optional mask of shape (batch_size, seq_len)
            causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Repeat keys and values for each query head
        k = k.repeat_interleave(self.n_queries_per_kv, dim=2)
        v = v.repeat_interleave(self.n_queries_per_kv, dim=2)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        k = k.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_head, seq_len, head_dim)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if required
        if causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_weights.masked_fill_(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output
        out = torch.matmul(attn_weights, v)  # (batch_size, n_head, seq_len, head_dim)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_head, head_dim)
        out = out.view(batch_size, seq_len, self.n_embd)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        
        return out

class GQABlock(nn.Module):
    """
    Transformer block using Grouped-Query Attention.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def convert_to_gqa(model, n_kv_heads):
    """
    Convert a standard transformer model to use GQA.
    
    Args:
        model: The model to convert
        n_kv_heads: Number of key/value heads to use
        
    Returns:
        Modified model using GQA
    """
    for name, module in model.named_modules():
        if "attn" in name and isinstance(module, nn.Module):
            config = module.config if hasattr(module, 'config') else None
            if config is not None:
                config.n_kv_heads = n_kv_heads
                setattr(model, name, GroupedQueryAttention(config))
    return model