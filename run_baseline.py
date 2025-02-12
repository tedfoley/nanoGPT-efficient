"""Script to run the baseline nanoGPT implementation.

This script provides a way to run the original nanoGPT implementation
with the same data and evaluation setup as our optimized version.
"""

import argparse
import sys
from pathlib import Path
import torch
import wandb

# Add baseline directory to path
baseline_dir = Path("research/baseline/original")
sys.path.insert(0, str(baseline_dir))

from research.baseline.original.model import GPTConfig, GPT
from research.evaluation.benchmarks import ModelBenchmark

def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline nanoGPT')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--out-dir', type=str, default='out-baseline',
                      help='Output directory')
    parser.add_argument('--wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create directories
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(project="nanoGPT-efficient", name="baseline")
    
    # Create model with original nanoGPT configuration
    config = GPTConfig(
        n_layer=12,
        n_head=12,
        n_embd=768,
        block_size=1024,
        bias=True,
        vocab_size=50257,
        dropout=0.0,
    )
    
    model = GPT(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
        # Import original training code
        sys.path.insert(0, str(baseline_dir))
        from train import train
        
        # Run training using original code
        train(
            config=config,
            train_data=args.data_dir + '/train.bin',
            val_data=args.data_dir + '/val.bin',
            max_iters=100000,
            eval_interval=1000,
            log_interval=10,
            out_dir=out_dir,
            # Original parameters from nanoGPT
            batch_size=12,
            learning_rate=6e-4,
            min_lr=6e-5,
            warmup_iters=2000,
        )
        
        # Run our benchmarks on the trained model
        print("\nRunning benchmarks on baseline model...")
        benchmark = ModelBenchmark(model, device)
        metrics = benchmark.run_comprehensive_benchmark(
            model=model,
            train_data=args.data_dir + '/train.bin',
            val_data=args.data_dir + '/val.bin',
        )
        
        if args.wandb:
            wandb.log(metrics)
        
        print("\nBaseline Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
            
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        if args.wandb:
            wandb.finish()

if __name__ == '__main__':
    main()