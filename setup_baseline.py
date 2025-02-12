"""Script to set up baseline nanoGPT implementation.

This script:
1. Clones the original nanoGPT repository
2. Sets up the baseline directory structure
3. Copies necessary files to our project
"""

import os
import shutil
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clone_original_nanogpt():
    """Clone the original nanoGPT repository."""
    nanogpt_dir = Path("baseline_nanogpt")
    if nanogpt_dir.exists():
        logger.info("nanoGPT directory already exists, skipping clone")
        return nanogpt_dir
    
    logger.info("Cloning original nanoGPT repository...")
    subprocess.run([
        "git", "clone",
        "https://github.com/karpathy/nanoGPT.git",
        str(nanogpt_dir)
    ], check=True)
    
    return nanogpt_dir

def setup_baseline():
    """Set up the baseline implementation."""
    # Clone original repo
    nanogpt_dir = clone_original_nanogpt()
    
    # Create baseline directory
    baseline_dir = Path("research/baseline/original")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy core files
    files_to_copy = [
        "model.py",
        "config.py",
        "train.py",
        "data/shakespeare_char/prepare.py",
    ]
    
    for file in files_to_copy:
        src = nanogpt_dir / file
        dst = baseline_dir / Path(file).name
        if src.exists():
            logger.info(f"Copying {src} to {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            logger.warning(f"Source file {src} not found")

if __name__ == "__main__":
    setup_baseline()