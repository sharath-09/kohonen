# Kohonen Self-Organizing Map

A Python implementation of Self-Organizing Maps (SOMs) with both batch and stochastic training methods, deployable as a serverless FastAPI application on AWS.

## Features

- ✅ Vectorized NumPy operations for efficiency
- ✅ Configurable learning rate and neighborhood radius decay
- ✅ Stochastic mini-batch training support
- ✅ Reproducible training with seed control

## TODO

- FastAPI REST endpoints
- AWS Batch + S3 integration

## Installation

```bash
# Install dependencies with uv
uv sync
source .venv/bin/activate
```

## Usage

### Training locally

**Network training:**
```bash
python main.py
```

**Stochastic training:**
```bash
python main.py --stochastic
```

**Custom parameters:**
```bash
python main.py --stochastic --batch-size 16 --seed 42
```