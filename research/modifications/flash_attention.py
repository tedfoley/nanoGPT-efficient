"""Flash Attention implementation for efficient attention computation.

This module implements Flash Attention as described in the paper:
"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
https://arxiv.org/abs/2205.14135

Key modifications from standard attention:
- O(N) memory complexity instead of O(N²)
- Improved performance through better memory access patterns
- Tiled matrix multiplication for efficient GPU utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func

class FlashSelfAttention(nn.Module):
    """
    Efficient implementation of self-attention using Flash Attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Projection matrices
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape q, k, v for flash attention
        # Flash attention expects (batch_size, seqlen, nheads, head_dim)
        head_dim = self.n_embd // self.n_head
        q = q.view(B, T, self.n_head, head_dim)
        k = k.view(B, T, self.n_head, head_dim)
        v = v.view(B, T, self.n_head, head_dim)
        
        # Pack QKV into single tensor for flash attention
        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.view(B, T, 3, self.n_head, head_dim)
        
        # Apply flash attention
        output = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True
        )
        
        # Reshape output and apply output projection
        output = output.view(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        
        return output

class FlashAttentionWrapper:
    """
    Wrapper class to easily replace standard attention with Flash Attention.
    """
    @staticmethod
    def replace_attention(model):
        """
        Replaces standard attention layers with Flash Attention in a model.
        
        Args:
            model: The GPT model to modify
            
        Returns:
            Modified model with Flash Attention
        """
        for name, module in model.named_modules():
            if "attn" in name and isinstance(module, nn.Module):
                config = module.config if hasattr(module, 'config') else None
                if config is not None:
                    setattr(model, name, FlashSelfAttention(config))
        return model

def compute_flash_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    Standalone function to compute flash attention given Q, K, V matrices.
    
    Args:
        q: Query matrix (B, T, H, D)
        k: Key matrix (B, T, H, D)
        v: Value matrix (B, T, H, D)
        mask: Optional attention mask
        dropout_p: Dropout probability
        
    Returns:
        Attention output
    """
    # Pack QKV into format expected by flash attention
    qkv = torch.stack([q, k, v], dim=2)
    
    # Apply flash attention
    output = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=True
    )
    
    return output