# Password Guesser - Dockerfile
# Multi-stage build for smaller image

# ---- Stage 1: Build ----
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Stage 2: Runtime ----
FROM python:3.11-slim

LABEL maintainer="your@email.com"
LABEL description="AI-powered targeted password guessing system"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages
COPY --from=builder /install /usr/local

# Set working directory
WORKDIR /app

# Copy project
COPY . .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/checkpoints /app/data

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Default: start web server
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
