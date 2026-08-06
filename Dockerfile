# WorldModelLens Dockerfile

# CPU Version
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

# Default command
CMD ["python", "-c", "print('WorldModelLens ready')"]


# GPU Version (use this for CUDA)
# docker build -f Dockerfile.gpu -t worldmodel-lens:gpu .
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir -e .

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "world_model_lens.deployment.api"]
