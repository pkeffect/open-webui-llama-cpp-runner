"""
LlamaCpp GPU Proxy Server

This script provides a GPU-accelerated FastAPI server for llama-cpp-runner
with beautiful formatting, consistent endpoints, and proper logging.
"""

import os
import sys
import uvicorn
import subprocess
from fastapi import FastAPI

# Import our modules
from llama_cpp_runner.main_gpu import LlamaCpp
from llama_cpp_runner.api import create_api
from llama_cpp_runner.logger import setup_logger, get_logger

# Configure logging
logger = setup_logger("llama_cpp_runner.gpu_server", console=True, file=True)

# Print banner
print("\n" + "=" * 60)
print("🦙 LlamaCpp Proxy Server (GPU Version)")
print("=" * 60 + "\n")

# Get environment variables
models_dir = os.environ.get("MODELS_DIR", "/models")
cache_dir = os.environ.get("CACHE_DIR", "/cache")
verbose = os.environ.get("VERBOSE", "true").lower() == "true"
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))

# GPU Configuration
enable_gpu = os.environ.get("ENABLE_GPU", "true").lower() == "true"
gpu_layers = os.environ.get("GPU_LAYERS", "-1")
try:
    gpu_layers = int(gpu_layers) if gpu_layers != "-1" else -1
except ValueError:
    logger.warning(f"⚠️  Invalid GPU_LAYERS value: {gpu_layers}. Using default -1 (all layers).")
    gpu_layers = -1

# Log configuration
logger.info(f"🗂️  Models directory: {models_dir}")
logger.info(f"💾 Cache directory: {cache_dir}")
logger.info(f"🔊 Verbose mode: {verbose}")
logger.info(f"⏱️  Timeout: {timeout} minutes")
logger.info(f"🖥️  GPU enabled: {enable_gpu}")
logger.info(f"🔢 GPU layers: {gpu_layers}")

# Check for GPU availability
try:
    logger.info("🔍 Checking GPU availability...")
    nvidia_smi = subprocess.run(
        ["nvidia-smi"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    if nvidia_smi.returncode == 0:
        # Extract GPU info from nvidia-smi output
        gpu_name = "Unknown"
        cuda_version = "Unknown"
        
        for line in nvidia_smi.stdout.split('\n'):
            if "NVIDIA" in line and "Driver Version" not in line:
                gpu_name = line.strip()
            if "CUDA Version" in line:
                cuda_version = line.split("CUDA Version:")[1].strip()
        
        logger.info(f"✅ GPU detected: {gpu_name}")
        logger.info(f"📊 CUDA Version: {cuda_version}")
    else:
        logger.warning("⚠️  No GPU detected or NVIDIA drivers not installed!")
        if enable_gpu:
            logger.warning("⚠️  GPU mode enabled but no GPU detected. Will run in CPU-only mode.")
except FileNotFoundError:
    logger.warning("⚠️  nvidia-smi not found. GPU support will be limited.")
except Exception as e:
    logger.error(f"❌ Error checking GPU: {e}")

# Create the LlamaCpp instance with GPU support
try:
    logger.info("🔄 Initializing GPU-enabled LlamaCpp...")
    llama_runner = LlamaCpp(
        models_dir=models_dir,
        cache_dir=cache_dir, 
        verbose=verbose, 
        timeout_minutes=timeout,
        enable_gpu=enable_gpu,
        gpu_layers=gpu_layers
    )
    
    # List available models
    models = llama_runner.list_models()
    if models:
        logger.info(f"📚 Found {len(models)} models: {', '.join(models)}")
    else:
        logger.warning("⚠️  No models found in the models directory!")
    
    # Create the API with our unified API module
    logger.info("🔄 Creating GPU-enabled API...")
    app = create_api(llama_runner, enable_gpu=enable_gpu, gpu_layers=gpu_layers)
    
except Exception as e:
    logger.error(f"❌ Failed to initialize LlamaCpp: {e}")
    sys.exit(1)

# Print registered routes for debugging
if verbose:
    logger.info("📋 Registered API endpoints:")
    for route in app.routes:
        methods = ", ".join(route.methods) if hasattr(route, "methods") and route.methods else "GET"
        logger.info(f"  • {methods:7} {route.path}")

if __name__ == "__main__":
    # Print server URL
    logger.info("\n🚀 Starting LlamaCpp GPU Proxy Server on port 10000")
    logger.info("🌐 Server will be available at: http://localhost:10000")
    logger.info("🔍 Try: curl http://localhost:10000/health\n")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=10000)