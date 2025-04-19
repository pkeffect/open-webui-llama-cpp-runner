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

# Install basic dependencies first
RUN pip install --no-cache-dir "requests>=2.28.0" "fastapi>=0.95.0" "uvicorn>=0.21.0"

# Then install the package in development mode
RUN pip install --no-cache-dir -e .

# Create volume mount points
VOLUME /models
VOLUME /cache

# Expose the API port
EXPOSE 10000

# Set environment variables with reasonable defaults
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/models
ENV CACHE_DIR=/cache
ENV VERBOSE=true
ENV TIMEOUT_MINUTES=30
ENV PORT=10000
ENV HOST=0.0.0.0
ENV LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Command to run when the container starts
CMD ["python", "-m", "llama_cpp_runner.main", "--models-dir", "${MODELS_DIR}", "--cache-dir", "${CACHE_DIR}", "--port", "${PORT}", "--host", "${HOST}", "--timeout", "${TIMEOUT_MINUTES}", "--log-level", "${LOG_LEVEL}", "--verbose"]