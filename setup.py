"""Setup script for nanoGPT-efficient package."""

from setuptools import setup, find_packages

setup(
    name="nanogpt-efficient",
    version="0.1.0",
    description="An optimized implementation of nanoGPT focusing on algorithmic efficiency",
    author="Ted Foley",
    author_email="tedfoley@github.com",  # Replace with your email
    url="https://github.com/tedfoley/nanoGPT-efficient",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "flash-attn>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "train-gpt=train_gpt:main",
        ],
    },
)
