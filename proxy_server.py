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
from typing import Dict, Any

# Import our unified API
from llama_cpp_runner.main import run_server, DEFAULT_CONFIG
from llama_cpp_runner.logger import setup_logger, get_logger

def main():
    """Main entry point for the proxy server"""
    parser = argparse.ArgumentParser(description="LlamaCpp Proxy Server")
    
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
    print(f"🦙 LlamaCpp Proxy Server ({banner_type} Version)")
    print("=" * 60 + "\n")
    
    # Configure logging
    setup_logger(
        "llama_cpp_runner", 
        level=args.log_level.upper(), 
        console=True, 
        file=args.verbose
    )
    
    logger = get_logger("proxy_server")
    
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
    
    # Run the server
    try:
        run_server(config)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()