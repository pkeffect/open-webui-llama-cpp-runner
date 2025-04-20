FROM python:3.11-slim

LABEL maintainer="Open WebUI (Timothy Jaeryang Baek)"
LABEL description="LlamaCpp Runner - CPU version"
LABEL version="0.0.1"

WORKDIR /app

# Install only essential packages and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is up to date
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only necessary files
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/

# Install basic dependencies and the package in one layer to reduce image size
RUN pip install --no-cache-dir "requests>=2.28.0" "fastapi>=0.95.0" "uvicorn>=0.21.0" && \
    pip install --no-cache-dir -e .

# Create volume mount points and ensure they exist
RUN mkdir -p /models /cache
VOLUME /models
VOLUME /cache

# Create non-root user for better security
RUN groupadd -r llamauser && useradd -r -g llamauser llamauser && \
    chown -R llamauser:llamauser /app /models /cache

# Switch to non-root user
USER llamauser

# Expose the API port
EXPOSE 10000

# Set environment variables with reasonable defaults
ENV PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    CACHE_DIR=/cache \
    VERBOSE=true \
    TIMEOUT_MINUTES=30 \
    PORT=10000 \
    HOST=0.0.0.0 \
    LOG_LEVEL=info

# Health check with improved resilience
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use shell form for proper environment variable expansion
CMD python -m llama_cpp_runner.unified_server \
    --models-dir ${MODELS_DIR} \
    --cache-dir ${CACHE_DIR} \
    --port ${PORT} \
    --host ${HOST} \
    --timeout ${TIMEOUT_MINUTES} \
    --log-level ${LOG_LEVEL} \
    ${VERBOSE:+--verbose}