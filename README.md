# nanoGPT-efficient

An optimized implementation of [nanoGPT](https://github.com/karpathy/nanoGPT) focusing on algorithmic efficiency and memory optimizations.

## Key Features

- Flash Attention implementation for O(n) complexity
- Rotary Embeddings for improved position encoding
- Grouped-Query Attention for efficient attention computation
- Memory optimizations reducing parameter count from 124M to ~50M
- Efficient training pipeline with gradient checkpointing and mixed precision

## Project Structure

```
research/
├── baseline/         # Original nanoGPT implementation
├── modifications/    # Efficiency optimizations
├── evaluation/       # Benchmarking tools
├── analysis/         # Training analysis notebooks
└── docs/            # Documentation
```

## Setup

```bash
# Clone the repository
git clone https://github.com/tedfoley/nanoGPT-efficient.git
cd nanoGPT-efficient

# Install dependencies
pip install -r requirements.txt
```

## Usage

Detailed usage instructions coming soon.

## License

MIT License - see LICENSE file for details.