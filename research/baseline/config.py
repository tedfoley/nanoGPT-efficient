"""Configuration classes for model and training.

This module provides configuration classes for:
1. Model architecture (CompactGPTConfig)
2. Training settings (TrainingConfig)
3. Data processing (DataConfig)
"""

from dataclasses import dataclass
import torch
from typing import Optional, Tuple, List, Union
from pathlib import Path

@dataclass
class CompactGPTConfig:
    """Model architecture configuration."""
    # Model dimensions
    n_layer: int = 8          # Number of transformer layers
    n_head: int = 8           # Number of attention heads
    n_embd: int = 512        # Embedding dimension
    block_size: int = 512     # Context window size
    
    # Attention configuration
    n_kv_heads: int = 2       # Number of key/value heads for GQA
    flash_attention: bool = True  # Whether to use Flash Attention
    rotary_embeddings: bool = True  # Whether to use Rotary Embeddings
    
    # Dropout settings
    dropout: float = 0.2      # Dropout probability
    embd_dropout: float = 0.1 # Embedding dropout
    
    # Other architecture settings
    vocab_size: int = 50257   # GPT-2 vocabulary size
    bias: bool = False        # Whether to use bias in linear layers
    
    def __post_init__(self):
        assert self.n_head % self.n_kv_heads == 0, \
            f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})"

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Hardware settings
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype: torch.dtype = torch.float16
    compile: bool = True
    gradient_checkpointing: bool = True
    
    # Training hyperparameters
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
    wandb_project: str = "nanoGPT-efficient"
    wandb_run_name: Optional[str] = None
    checkpoint_dir: Path = Path("checkpoints")
    
    # Evaluation settings
    eval_batch_size: int = 64
    eval_iters: int = 20
    
    def __post_init__(self):
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class DataConfig:
    """Data processing configuration."""
    # Data paths
    train_data: Union[str, Path]
    val_data: Union[str, Path]
    
    # Tokenizer settings
    tokenizer_path: Optional[str] = None
    tokenizer_type: str = "gpt2"  # one of ["gpt2", "custom"]
    
    # Processing settings
    max_seq_length: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Data loading
    streaming: bool = False  # Whether to stream data from disk
    cache_dir: Optional[Path] = None
    
    def __post_init__(self):
        self.train_data = Path(self.train_data)
        self.val_data = Path(self.val_data)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: CompactGPTConfig
    training: TrainingConfig
    data: DataConfig
    
    # Experiment metadata
    name: str
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        # Set wandb run name if not specified
        if self.training.wandb_run_name is None:
            self.training.wandb_run_name = self.name
    
    def save(self, path: Union[str, Path]):
        """Save configuration to disk."""
        import json
        path = Path(path)
        
        # Convert configs to dictionary
        config_dict = {
            'model': self.model.__dict__,
            'training': {k: str(v) if isinstance(v, (Path, torch.device)) else v 
                        for k, v in self.training.__dict__.items()},
            'data': {k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.data.__dict__.items()},
            'name': self.name,
            'description': self.description,
            'tags': self.tags
        }
        
        # Save to JSON
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from disk."""
        import json
        path = Path(path)
        
        with open(path) as f:
            config_dict = json.load(f)
        
        # Reconstruct config objects
        model_config = CompactGPTConfig(**config_dict['model'])
        training_config = TrainingConfig(**{k: torch.device(v) if k == 'device' else v 
                                          for k, v in config_dict['training'].items()})
        data_config = DataConfig(**config_dict['data'])
        
        return cls(
            model=model_config,
            training=training_config,
            data=data_config,
            name=config_dict['name'],
            description=config_dict['description'],
            tags=config_dict['tags']
        )

# Example configurations
def get_default_config() -> ExperimentConfig:
    """Get default configuration for training."""
    return ExperimentConfig(
        model=CompactGPTConfig(),
        training=TrainingConfig(),
        data=DataConfig(
            train_data="data/train",
            val_data="data/val"
        ),
        name="default-experiment"
    )

def get_small_test_config() -> ExperimentConfig:
    """Get configuration for quick testing."""
    config = get_default_config()
    
    # Reduce model size
    config.model.n_layer = 4
    config.model.n_head = 4
    config.model.n_embd = 256
    config.model.block_size = 256
    
    # Reduce training time
    config.training.max_epochs = 1
    config.training.steps_per_epoch = 100
    config.training.batch_size = 8
    config.training.eval_batch_size = 8
    
    config.name = "small-test"
    return config