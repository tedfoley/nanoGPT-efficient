"""Data processing utilities for efficient training.

This module provides:
1. Dataset classes for efficient data loading
2. Tokenization utilities
3. Data processing pipelines
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, List, Tuple
import json
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import mmap
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataStats:
    """Statistics about a dataset."""
    num_tokens: int
    vocab_size: int
    max_seq_length: int
    num_sequences: int
    token_frequency: Dict[int, int]

class MemoryMappedDataset(Dataset):
    """Memory-mapped dataset for efficient data loading."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        block_size: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        cache_dir: Optional[Path] = None
    ):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Get file size and create memory map
        self.file_size = os.path.getsize(self.data_path)
        self._mmap = None
        
        # Cache computations if directory provided
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_or_create_cache()
        
        # Initialize statistics
        self.stats = None
    
    def _create_mmap(self):
        """Create memory map for efficient file access."""
        fd = os.open(self.data_path, os.O_RDONLY)
        self._mmap = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        os.close(fd)
    
    def _load_or_create_cache(self):
        """Load or create cache of valid indices."""
        cache_path = self.cache_dir / f"{self.data_path.stem}_indices.json"
        
        if cache_path.exists():
            logger.info(f"Loading index cache from {cache_path}")
            with open(cache_path) as f:
                self._valid_indices = json.load(f)
        else:
            logger.info("Creating index cache...")
            # Find all valid starting positions
            self._valid_indices = []
            
            if self._mmap is None:
                self._create_mmap()
            
            for i in tqdm(range(0, self.file_size - self.block_size), desc="Creating index cache"):
                # Check if position starts a valid sequence
                try:
                    self._mmap[i:i+self.block_size].decode('utf-8')
                    self._valid_indices.append(i)
                except UnicodeDecodeError:
                    continue
            
            # Save cache
            with open(cache_path, 'w') as f:
                json.dump(self._valid_indices, f)
            logger.info(f"Saved index cache to {cache_path}")
    
    def compute_statistics(self) -> DataStats:
        """Compute dataset statistics."""
        if self.stats is not None:
            return self.stats
        
        logger.info("Computing dataset statistics...")
        num_tokens = 0
        token_freq = {}
        max_seq_len = 0
        
        for i in tqdm(range(min(1000, len(self))), desc="Computing statistics"):
            item = self[i]
            if isinstance(item, torch.Tensor):
                seq_len = len(item)
                tokens = item.tolist()
            else:
                if self.tokenizer is None:
                    raise ValueError("Cannot compute token statistics without tokenizer")
                tokens = self.tokenizer.encode(item)
                seq_len = len(tokens)
            
            num_tokens += seq_len
            max_seq_len = max(max_seq_len, seq_len)
            
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Extrapolate total tokens for full dataset
        num_tokens = int(num_tokens * (len(self) / 1000)) if len(self) > 1000 else num_tokens
        
        self.stats = DataStats(
            num_tokens=num_tokens,
            vocab_size=len(token_freq),
            max_seq_length=max_seq_len,
            num_sequences=len(self),
            token_frequency=token_freq
        )
        
        return self.stats
    
    def __len__(self):
        """Return number of possible sequences."""
        if hasattr(self, '_valid_indices'):
            return len(self._valid_indices)
        return max(0, self.file_size - self.block_size)
    
    def __getitem__(self, idx) -> Union[str, torch.Tensor]:
        """Get sequence at index."""
        if self._mmap is None:
            self._create_mmap()
        
        # Get starting position
        if hasattr(self, '_valid_indices'):
            pos = self._valid_indices[idx]
        else:
            pos = idx
        
        # Read sequence
        text = self._mmap[pos:pos+self.block_size].decode('utf-8')
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=self.block_size,
                return_tensors="pt"
            )
            return tokens['input_ids'][0]
        
        return text

class StreamingDataset(Dataset):
    """Streaming dataset for handling large files."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        block_size: int,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        buffer_size: int = 1000
    ):
        self.data_path = Path(data_path)
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        
        # Initialize buffer
        self._current_buffer = []
        self._buffer_start = 0
        self._file = open(self.data_path, 'r')
        
        # Statistics
        self.stats = None
    
    def _fill_buffer(self):
        """Fill buffer with new sequences."""
        self._current_buffer = []
        
        while len(self._current_buffer) < self.buffer_size:
            text = self._file.read(self.block_size)
            if not text:
                # Reset file if end reached
                self._file.seek(0)
                text = self._file.read(self.block_size)
            
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.block_size,
                    return_tensors="pt"
                )
                self._current_buffer.append(tokens['input_ids'][0])
            else:
                self._current_buffer.append(text)
    
    def compute_statistics(self) -> DataStats:
        """Compute approximate dataset statistics from buffer."""
        if self.stats is not None:
            return self.stats
        
        logger.info("Computing approximate dataset statistics from buffer...")
        num_tokens = 0
        token_freq = {}
        max_seq_len = 0
        
        # Fill buffer if empty
        if not self._current_buffer:
            self._fill_buffer()
        
        for item in self._current_buffer:
            if isinstance(item, torch.Tensor):
                seq_len = len(item)
                tokens = item.tolist()
            else:
                if self.tokenizer is None:
                    raise ValueError("Cannot compute token statistics without tokenizer")
                tokens = self.tokenizer.encode(item)
                seq_len = len(tokens)
            
            num_tokens += seq_len
            max_seq_len = max(max_seq_len, seq_len)
            
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        self.stats = DataStats(
            num_tokens=num_tokens,
            vocab_size=len(token_freq),
            max_seq_length=max_seq_len,
            num_sequences=self.buffer_size,
            token_frequency=token_freq
        )
        
        return self.stats
    
    def __len__(self):
        """Return buffer size as approximate length."""
        return self.buffer_size
    
    def __getitem__(self, idx) -> Union[str, torch.Tensor]:
        """Get item from buffer, refilling if necessary."""
        buffer_idx = idx - self._buffer_start
        
        if buffer_idx >= len(self._current_buffer) or buffer_idx < 0:
            self._buffer_start = idx
            self._fill_buffer()
            buffer_idx = 0
        
        return self._current_buffer[buffer_idx]
    
    def __del__(self):
        """Close file when dataset is deleted."""
        if hasattr(self, '_file'):
            self._file.close()

def create_dataloaders(
    config,
    tokenizer_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, PreTrainedTokenizer]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: ExperimentConfig object
        tokenizer_path: Optional path to tokenizer
        
    Returns:
        Tuple containing:
        - Training dataloader
        - Validation dataloader
        - Tokenizer instance
    """
    # Load or create tokenizer
    if tokenizer_path is None and config.data.tokenizer_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif tokenizer_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        raise ValueError("Must provide tokenizer_path if not using gpt2 tokenizer")
    
    # Create datasets
    dataset_class = StreamingDataset if config.data.streaming else MemoryMappedDataset
    
    train_dataset = dataset_class(
        config.data.train_data,
        config.model.block_size,
        tokenizer=tokenizer,
        buffer_size=config.training.batch_size * 10 if config.data.streaming else None,
        cache_dir=config.data.cache_dir
    )
    
    val_dataset = dataset_class(
        config.data.val_data,
        config.model.block_size,
        tokenizer=tokenizer,
        buffer_size=config.training.eval_batch_size * 10 if config.data.streaming else None,
        cache_dir=config.data.cache_dir
    )
    
    # Compute and log dataset statistics
    train_stats = train_dataset.compute_statistics()
    val_stats = val_dataset.compute_statistics()
    
    logger.info("Dataset statistics:")
    logger.info(f"Training set: {train_stats}")
    logger.info(f"Validation set: {val_stats}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers
    )
    
    return train_loader, val_loader, tokenizer

def prepare_sample_data(
    text_path: Union[str, Path],
    output_dir: Union[str, Path],
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Path, Path]:
    """
    Prepare sample data by splitting into train/val sets.
    
    Args:
        text_path: Path to input text file
        output_dir: Directory to save split files
        val_split: Fraction of data to use for validation
        seed: Random seed
        
    Returns:
        Tuple of (train_path, val_path)
    """
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read text
    with open(text_path) as f:
        text = f.read()
    
    # Split into lines
    lines = text.splitlines()
    np.random.seed(seed)
    np.random.shuffle(lines)
    
    # Split into train/val
    split_idx = int(len(lines) * (1 - val_split))
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    # Save splits
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_lines))
    
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_lines))
    
    logger.info(f"Saved {len(train_lines)} training examples to {train_path}")
    logger.info(f"Saved {len(val_lines)} validation examples to {val_path}")
    
    return train_path, val_path