#!/usr/bin/env python3
"""
Unified LlamaCpp Proxy Server

This script provides a FastAPI server for llama-cpp-runner with 
support for both CPU and GPU modes, beautiful formatting, 
consistent endpoints, and proper logging.
"""

import os
import sys
import argparse
import time
import uvicorn
from typing import Dict, Any

# Import our unified API
from llama_cpp_runner.logger import setup_logger, get_logger
from llama_cpp_runner.cpu import LlamaCpp as CpuLlamaCpp
from llama_cpp_runner.gpu import GpuLlamaCpp
from llama_cpp_runner.api import create_api
from llama_cpp_runner.utils import detect_gpu, get_system_info

# Set up logger
logger = get_logger("llama_cpp_runner.unified_server")

# Default configuration
DEFAULT_CONFIG = {
    "models_dir": os.path.expanduser("~/models"),
    "cache_dir": os.path.expanduser("~/.llama_cpp_runner"),
    "verbose": False,
    "timeout_minutes": 30,
    "port": 10000,
    "host": "0.0.0.0",
    "enable_gpu": False,
    "gpu_layers": -1,
    "log_level": "info"
}

def main():
    """Main entry point for the unified proxy server"""
    parser = argparse.ArgumentParser(description="LlamaCpp Unified Proxy Server")
    
    parser.add_argument(
        "--models-dir",
        default=os.environ.get("MODELS_DIR", DEFAULT_CONFIG["models_dir"]),
        help="Directory containing GGUF models"
    )
    
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("CACHE_DIR", DEFAULT_CONFIG["cache_dir"]),
        help="Directory for caching llama.cpp binaries"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", DEFAULT_CONFIG["port"])),
        help="Port to run the server on"
    )
    
    parser.add_argument(
        "--host",
        default=os.environ.get("HOST", DEFAULT_CONFIG["host"]),
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("TIMEOUT_MINUTES", DEFAULT_CONFIG["timeout_minutes"])),
        help="Timeout in minutes for shutting down idle servers"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=os.environ.get("VERBOSE", "").lower() == "true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=os.environ.get("LOG_LEVEL", DEFAULT_CONFIG["log_level"]),
        help="Logging level"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=os.environ.get("ENABLE_GPU", "").lower() == "true",
        help="Enable GPU acceleration"
    )
    
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=int(os.environ.get("GPU_LAYERS", DEFAULT_CONFIG["gpu_layers"])),
        help="Number of layers to offload to GPU (-1 for all)"
    )
    
    args = parser.parse_args()
    
    # Print banner
    banner_type = "GPU-accelerated" if args.gpu else "CPU"
    print("\n" + "=" * 60)
    print(f"🦙 LlamaCpp Unified Proxy Server ({banner_type} Version)")
    print("=" * 60 + "\n")
    
    # Configure logging
    setup_logger(
        "llama_cpp_runner", 
        level=args.log_level.upper(), 
        console=True, 
        file=args.verbose
    )
    
    logger = get_logger("unified_server")
    
    # Convert args to config
    config = {
        "models_dir": args.models_dir,
        "cache_dir": args.cache_dir,
        "port": args.port,
        "host": args.host,
        "timeout_minutes": args.timeout,
        "verbose": args.verbose,
        "log_level": args.log_level,
        "enable_gpu": args.gpu,
        "gpu_layers": args.gpu_layers
    }
    
    # Log configuration
    logger.info(f"🗂️  Models directory: {config['models_dir']}")
    logger.info(f"💾 Cache directory: {config['cache_dir']}")
    logger.info(f"🔊 Verbose mode: {config['verbose']}")
    logger.info(f"⏱️  Timeout: {config['timeout_minutes']} minutes")
    logger.info(f"🖥️  GPU enabled: {config['enable_gpu']}")
    if config['enable_gpu']:
        logger.info(f"🔢 GPU layers: {config['gpu_layers']}")
    
    # Check for GPU availability if GPU is enabled
    if config['enable_gpu']:
        gpu_info = detect_gpu()
        if gpu_info["available"]:
            logger.info(f"✅ GPU detected: {gpu_info['gpus'][0]['name'] if gpu_info['gpus'] else 'Unknown'}")
            if 'cuda_version' in gpu_info:
                logger.info(f"📊 CUDA Version: {gpu_info['cuda_version']}")
        else:
            logger.warning("⚠️  GPU requested but not available! Falling back to CPU mode.")
            config['enable_gpu'] = False
    
    # Initialize the appropriate runner based on configuration
    try:
        # Create directories if they don't exist
        os.makedirs(config['models_dir'], exist_ok=True)
        os.makedirs(config['cache_dir'], exist_ok=True)
        
        start_time = time.time()
        
        if config['enable_gpu']:
            logger.info("🔄 Initializing GPU-enabled LlamaCpp...")
            llama_runner = GpuLlamaCpp(
                models_dir=config['models_dir'],
                cache_dir=config['cache_dir'],
                verbose=config['verbose'],
                timeout_minutes=config['timeout_minutes'],
                enable_gpu=True,
                gpu_layers=config['gpu_layers']
            )
        else:
            logger.info("🔄 Initializing CPU-optimized LlamaCpp...")
            llama_runner = CpuLlamaCpp(
                models_dir=config['models_dir'],
                cache_dir=config['cache_dir'],
                verbose=config['verbose'],
                timeout_minutes=config['timeout_minutes']
            )
        
        init_time = time.time() - start_time
        logger.info(f"✅ Initialization completed in {init_time:.2f} seconds")
        
        # List available models
        models = llama_runner.list_models()
        if models:
            logger.info(f"📚 Found {len(models)} models: {', '.join(models)}")
        else:
            logger.warning("⚠️  No models found in the models directory!")
        
        # Create the API
        app_title = "LlamaCpp GPU API" if config['enable_gpu'] else "LlamaCpp API"
        app = create_api(
            llama_runner, 
            title=app_title,
            version="0.0.1",
            enable_gpu=config['enable_gpu'],
            gpu_layers=config['gpu_layers']
        )
        
        # Print registered routes for debugging
        if config['verbose']:
            logger.info("📋 Registered API endpoints:")
            for route in app.routes:
                methods = ", ".join(route.methods) if hasattr(route, "methods") and route.methods else "GET"
                logger.info(f"  • {methods:7} {route.path}")
        
        # Print server URL
        logger.info(f"\n🚀 Starting LlamaCpp Server on port {config['port']}")
        logger.info(f"🌐 Server will be available at: http://{config['host']}:{config['port']}")
        logger.info(f"📘 API documentation: http://{config['host']}:{config['port']}/docs")
        logger.info("🔍 Try: curl http://localhost:10000/health\n")
        
        # Store start time in app state
        app.state.start_time = time.time()
        
        # Start the server with uvicorn
        uvicorn.run(
            app, 
            host=config['host'], 
            port=config['port'],
            log_level=config['log_level']
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {e}")
        if config['verbose']:
            logger.exception("Detailed error information:")
        sys.exit(1)

if __name__ == "__main__":
    main()