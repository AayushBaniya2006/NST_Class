# Dockerfile for Vertex AI Custom Training Job
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

ENV PYTHONPATH=/app

# Default entrypoint for Vertex AI custom training
ENTRYPOINT ["python", "scripts/train.py"]
