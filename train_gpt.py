#!/usr/bin/env python
"""Main training script for efficient GPT training.

Example usage:
    # Train with default configuration
    python train_gpt.py --data-dir data/

    # Train with custom configuration
    python train_gpt.py --config configs/my_config.json --data-dir data/
    
    # Resume from checkpoint
    python train_gpt.py --config configs/my_config.json --data-dir data/ --resume checkpoints/checkpoint_1000.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import wandb

from research.baseline.config import ExperimentConfig, get_default_config, get_small_test_config
from research.baseline.data import create_dataloaders, prepare_sample_data
from research.baseline.train import train

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train an efficient GPT model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str,
                      help='Path to configuration file')
    parser.add_argument('--test-mode', action='store_true',
                      help='Use small test configuration for debugging')
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Directory containing training data')
    parser.add_argument('--cache-dir', type=str,
                      help='Directory for caching preprocessed data')
    parser.add_argument('--tokenizer-path', type=str,
                      help='Path to custom tokenizer')
    
    # Training
    parser.add_argument('--resume', type=str,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default='outputs',
                      help='Directory to save outputs')
    parser.add_argument('--wandb-run-name', type=str,
                      help='Weights & Biases run name')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable Weights & Biases logging')
    
    # Hardware
    parser.add_argument('--device', type=str,
                      help='Device to train on (cuda or cpu)')
    parser.add_argument('--num-workers', type=int,
                      help='Number of data loading workers')
    
    return parser.parse_args()

def setup_experiment(args) -> ExperimentConfig:
    """Set up experiment configuration."""
    # Load or create config
    if args.test_mode:
        config = get_small_test_config()
    elif args.config:
        config = ExperimentConfig.load(args.config)
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    if args.device:
        config.training.device = torch.device(args.device)
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.cache_dir:
        config.data.cache_dir = Path(args.cache_dir)
    if args.wandb_run_name:
        config.training.wandb_run_name = args.wandb_run_name
    if args.no_wandb:
        config.training.wandb_run_name = None
    
    # Set up directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config.training.checkpoint_dir = output_dir / "checkpoints"
    config.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = output_dir / "config.json"
    config.save(config_path)
    logger.info(f"Saved configuration to {config_path}")
    
    return config

def prepare_data(args, config: ExperimentConfig):
    """Prepare training and validation data."""
    data_dir = Path(args.data_dir)
    
    # Check if data needs to be split
    if (data_dir / "train.txt").exists() and (data_dir / "val.txt").exists():
        logger.info("Using existing train/val split")
        config.data.train_data = data_dir / "train.txt"
        config.data.val_data = data_dir / "val.txt"
    else:
        logger.info("Creating train/val split")
        text_files = list(data_dir.glob("*.txt"))
        if not text_files:
            raise ValueError(f"No .txt files found in {data_dir}")
        
        # Use the first text file
        train_path, val_path = prepare_sample_data(
            text_files[0],
            data_dir,
            val_split=0.1
        )
        config.data.train_data = train_path
        config.data.val_data = val_path

def main():
    """Main training function."""
    args = parse_args()
    
    # Set up experiment
    config = setup_experiment(args)
    logger.info(f"Running experiment: {config.name}")
    
    # Prepare data
    prepare_data(args, config)
    
    # Create dataloaders
    train_loader, val_loader, tokenizer = create_dataloaders(
        config,
        tokenizer_path=args.tokenizer_path
    )
    
    # Initialize wandb if enabled
    if config.training.wandb_run_name:
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.wandb_run_name,
            config=config.__dict__
        )
    
    try:
        # Train model
        model, optimizer, metrics = train(
            data_train=train_loader.dataset,
            data_val=val_loader.dataset,
            model_config=config.model,
            train_config=config.training,
            wandb_run_name=config.training.wandb_run_name,
            checkpoint_dir=config.training.checkpoint_dir,
            checkpoint_path=args.resume
        )
        
        # Save final results
        results_path = Path(args.output_dir) / "results.json"
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved final results to {results_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.exception("Training failed")
        raise e
    finally:
        if wandb.run is not None:
            wandb.finish()

if __name__ == '__main__':
    main()
