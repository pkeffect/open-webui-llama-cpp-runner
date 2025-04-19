"""
LlamaCpp Proxy Server (CPU Version)

This script provides a FastAPI server for llama-cpp-runner with
beautiful formatting, consistent endpoints, and proper logging.
"""

import os
import sys
import uvicorn
from fastapi import FastAPI

# Import our modules
from llama_cpp_runner.main import LlamaCpp
from llama_cpp_runner.api import create_api
from llama_cpp_runner.logger import setup_logger, get_logger

# Configure logging
logger = setup_logger("llama_cpp_runner.server", console=True, file=True)

# Print banner
print("\n" + "=" * 60)
print("🦙 LlamaCpp Proxy Server (CPU Version)")
print("=" * 60 + "\n")

# Get environment variables
models_dir = os.environ.get("MODELS_DIR", "/models")
cache_dir = os.environ.get("CACHE_DIR", "/cache")
verbose = os.environ.get("VERBOSE", "true").lower() == "true"
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))

# Log configuration
logger.info(f"🗂️  Models directory: {models_dir}")
logger.info(f"💾 Cache directory: {cache_dir}")
logger.info(f"🔊 Verbose mode: {verbose}")
logger.info(f"⏱️  Timeout: {timeout} minutes")

# Create the LlamaCpp instance
try:
    logger.info("🔄 Initializing LlamaCpp...")
    llama_runner = LlamaCpp(
        models_dir=models_dir,
        cache_dir=cache_dir, 
        verbose=verbose, 
        timeout_minutes=timeout
    )
    
    # List available models
    models = llama_runner.list_models()
    if models:
        logger.info(f"📚 Found {len(models)} models: {', '.join(models)}")
    else:
        logger.warning("⚠️  No models found in the models directory!")
    
    # Create the API with our unified API module
    logger.info("🔄 Creating API...")
    app = create_api(llama_runner, enable_gpu=False)
    
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
    logger.info("\n🚀 Starting LlamaCpp Proxy Server on port 10000")
    logger.info("🌐 Server will be available at: http://localhost:10000")
    logger.info("🔍 Try: curl http://localhost:10000/health\n")
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=10000)