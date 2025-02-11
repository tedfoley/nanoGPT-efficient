"""
Efficient training script for nanoGPT with optimizations.

Key features:
- Mixed precision training
- Gradient checkpointing
- Flash Attention
- Rotary Embeddings
- Grouped-Query Attention
"""

import os
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional

from ..modifications.flash_attention import FlashSelfAttention
from ..modifications.rotary import RotaryEmbedding
from ..modifications.grouped_query_attention import GroupedQueryAttention
from .model import CompactGPTConfig, CompactGPT
from ..evaluation.benchmarks import ModelBenchmark

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Optimization
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Training duration
    max_epochs: int = 10
    steps_per_epoch: int = 1000
    warmup_steps: int = 2000
    
    # Learning rate schedule
    lr_decay: bool = True
    lr_decay_epochs: int = 2
    min_lr: float = 1e-5
    
    # Logging and checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 1000
    eval_interval: int = 500
    
    # System
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.float16
    compile: bool = True

def create_optimizer(model, config):
    """
    Create optimizer with weight decay handling.
    """
    # Separate parameters that should have weight decay from those that shouldn't
    decay = set()
    no_decay = set()
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, (nn.Linear, nn.Conv1d)):
                decay.add(fpn)
            else:
                no_decay.add(fpn)
    
    param_dict = {pn: p for pn, p in model.named_parameters()}
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
    ]
    
    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2)
    )

def get_batch(data, batch_size, block_size, device):
    """Get a random batch of data."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size]) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size]) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_lr(config, step):
    """Get learning rate with warmup and decay."""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if not config.lr_decay:
        return config.learning_rate
    
    # Cosine decay after warmup
    decay_ratio = (step - config.warmup_steps) / (config.steps_per_epoch * config.max_epochs - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def estimate_loss(model, data, config, eval_iters=20):
    """Estimate loss on train and validation splits."""
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        losses[split] = 0
        for _ in range(eval_iters):
            x, y = get_batch(data[split], config.batch_size, model.config.block_size, config.device)
            with torch.no_grad():
                with autocast():
                    _, loss = model(x, y)
                losses[split] += loss.item()
        losses[split] /= eval_iters
    model.train()
    return losses

def train(
    data_train,
    data_val,
    model_config: CompactGPTConfig,
    train_config: TrainingConfig,
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: str = 'checkpoints',
    checkpoint_path: Optional[str] = None
):
    """
    Main training loop with optimizations.
    
    Args:
        data_train: Training data
        data_val: Validation data
        model_config: Configuration for model architecture
        train_config: Configuration for training
        wandb_run_name: Optional name for W&B run
        checkpoint_dir: Directory to save checkpoints
        checkpoint_path: Optional path to load checkpoint from
    """
    
    # Initialize wandb if requested
    if wandb_run_name:
        wandb.init(project="nanoGPT-efficient", name=wandb_run_name)
        wandb.config.update(model_config.__dict__)
        wandb.config.update(train_config.__dict__)
    
    # Create model
    model = CompactGPT(model_config)
    
    # Load checkpoint if provided
    start_step = 0
    best_val_loss = float('inf')
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=train_config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint['step']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    model.to(train_config.device)
    
    # Apply efficiency modifications
    model = FlashSelfAttention.replace_attention(model)
    model = GroupedQueryAttention.convert_to_gqa(model, model_config.n_kv_heads)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Compile model if requested
    if train_config.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Create optimizer
    optimizer = create_optimizer(model, train_config)
    if checkpoint_path:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scaler = GradScaler()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    step = start_step
    
    for epoch in range(train_config.max_epochs):
        pbar = tqdm(range(train_config.steps_per_epoch), desc=f"Epoch {epoch+1}")
        for _ in pbar:
            # Get batch
            x, y = get_batch(data_train, train_config.batch_size, model_config.block_size, train_config.device)
            
            # Update learning rate
            lr = get_lr(train_config, step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward and backward passes with mixed precision
            with autocast():
                logits, loss = model(x, y)
            
            # Scale loss and compute gradients
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if train_config.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if step % train_config.log_interval == 0:
                losses = estimate_loss(model, {'train': data_train, 'val': data_val}, train_config)
                pbar.set_description(
                    f"epoch {epoch+1} iter {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                
                if wandb_run_name:
                    wandb.log({
                        'train/loss': losses['train'],
                        'val/loss': losses['val'],
                        'train/lr': lr,
                        'epoch': epoch,
                        'step': step,
                    })
                
                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': model_config,
                        'step': step,
                        'best_val_loss': best_val_loss,
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            
            # Regular checkpointing
            if step % train_config.checkpoint_interval == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'step': step,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_{step}.pt'))
            
            # Evaluation
            if step % train_config.eval_interval == 0:
                model.eval()
                benchmark = ModelBenchmark(model, train_config.device)
                metrics = benchmark.measure_perplexity(
                    DataLoader(data_val, batch_size=train_config.batch_size),
                    max_batches=50
                )
                if wandb_run_name:
                    wandb.log({
                        'eval/perplexity': metrics,
                        'step': step,
                    })
                model.train()
            
            step += 1
    
    # Final evaluation
    print("Running final evaluation...")
    model.eval()
    benchmark = ModelBenchmark(model, train_config.device)
    train_loader = DataLoader(data_train, batch_size=train_config.batch_size)
    val_loader = DataLoader(data_val, batch_size=train_config.batch_size)
    
    final_metrics = benchmark.run_comprehensive_benchmark(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        run_name=f"{wandb_run_name}_final" if wandb_run_name else None
    )
    
    return model, optimizer, final_metrics

if __name__ == '__main__':
    # Example usage
    model_config = CompactGPTConfig()
    train_config = TrainingConfig()
    
    # Load your data here
    data_train = None  # Replace with actual data
    data_val = None    # Replace with actual data
    
    model, optimizer, metrics = train(
        data_train=data_train,
        data_val=data_val,
        model_config=model_config,
        train_config=train_config,
        wandb_run_name="efficient-gpt-test"
    )