"""Benchmarking utilities for model evaluation.

This module provides tools for evaluating model performance across various metrics:
- Training throughput (tokens/second)
- Memory usage
- Loss curves
- Inference speed
- Model perplexity
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import wandb
from tqdm import tqdm

class ModelBenchmark:
    """
    Comprehensive benchmarking suite for transformer models.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        metrics: List[str] = ['perplexity', 'throughput', 'memory', 'inference_speed']
    ):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.results = {}

    def measure_throughput(
        self,
        input_ids: torch.Tensor,
        batch_size: int = 32,
        num_batches: int = 50,
        warmup_batches: int = 10
    ) -> float:
        """
        Measure training throughput in tokens/second.
        """
        self.model.train()
        total_tokens = 0
        total_time = 0

        # Warmup
        for _ in range(warmup_batches):
            batch = input_ids[:batch_size]
            with torch.cuda.amp.autocast():
                _ = self.model(batch)

        # Actual measurement
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_batches):
            batch = input_ids[:batch_size]
            total_tokens += batch.numel()
            
            with torch.cuda.amp.autocast():
                _ = self.model(batch)
                
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        throughput = total_tokens / total_time
        self.results['throughput'] = throughput
        return throughput

    def measure_memory(self, input_ids: torch.Tensor, batch_size: int = 32) -> Dict[str, float]:
        """
        Measure peak memory usage during forward and backward pass.
        """
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        
        batch = input_ids[:batch_size]
        
        # Forward pass
        with torch.cuda.amp.autocast():
            output = self.model(batch)
            loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        memory_stats = {
            'peak_memory': torch.cuda.max_memory_allocated() / 1024**2,  # MB
            'current_memory': torch.cuda.memory_allocated() / 1024**2,    # MB
            'peak_memory_reserved': torch.cuda.max_memory_reserved() / 1024**2  # MB
        }
        
        self.results['memory'] = memory_stats
        return memory_stats

    def measure_perplexity(
        self,
        eval_dataloader: torch.utils.data.DataLoader,
        max_batches: Optional[int] = None
    ) -> float:
        """
        Calculate model perplexity on validation data.
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(eval_dataloader)):
                if max_batches and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids)
                    loss = outputs.mean()
                
                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.results['perplexity'] = perplexity
        return perplexity

    def measure_inference_speed(
        self,
        input_ids: torch.Tensor,
        num_tokens: int = 100,
        num_runs: int = 50,
        warmup_runs: int = 10
    ) -> float:
        """
        Measure inference speed (tokens/second) for text generation.
        """
        self.model.eval()
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=False
                )
        
        # Actual measurement
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    max_new_tokens=num_tokens,
                    do_sample=False
                )
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        tokens_per_second = (num_tokens * num_runs) / total_time
        self.results['inference_speed'] = tokens_per_second
        return tokens_per_second

    def log_metrics(self, run_name: str):
        """Log all computed metrics to W&B."""
        if wandb.run is None:
            wandb.init(project="nanoGPT-efficient", name=run_name)
        
        wandb.log(self.results)

def run_comprehensive_benchmark(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    run_name: str,
    device: torch.device = torch.device('cuda')
) -> Dict[str, float]:
    """
    Run all benchmarks and return results.
    """
    benchmark = ModelBenchmark(model, device)
    
    # Get sample batch for throughput and memory measurements
    sample_batch = next(iter(train_dataloader))['input_ids']
    
    # Run all benchmarks
    print("Measuring training throughput...")
    benchmark.measure_throughput(sample_batch)
    
    print("Measuring memory usage...")
    benchmark.measure_memory(sample_batch)
    
    print("Calculating perplexity...")
    benchmark.measure_perplexity(eval_dataloader)
    
    print("Measuring inference speed...")
    benchmark.measure_inference_speed(sample_batch[:1])
    
    # Log results to W&B
    benchmark.log_metrics(run_name)
    
    return benchmark.results