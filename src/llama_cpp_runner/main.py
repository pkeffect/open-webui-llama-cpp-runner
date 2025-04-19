"""
Main entry point for llama-cpp-runner

This module provides the main entry point for the library, handling
both CPU and GPU modes with unified configuration.
"""

import os
import sys
import argparse
import uvicorn
import platform
from typing import Dict, Any, Optional, List, Union

from llama_cpp_runner.logger import get_logger, setup_logger
from llama_cpp_runner.cpu import LlamaCpp as CpuLlamaCpp
from llama_cpp_runner.gpu import GpuLlamaCpp
from llama_cpp_runner.api import create_api
from llama_cpp_runner.utils import detect_gpu, get_system_info

# Set up logger
logger = get_logger("llama_cpp_runner.main")

# Default configuration
DEFAULT_CONFIG = {
    "models_dir": os.path.expanduser("~/models"),
    "cache_dir": os.path.expanduser("~/.llama_cpp_runner"),
    "verbose": False,
    "timeout_minutes": 30,
    "port": 8080,
    "host": "0.0.0.0",
    "enable_gpu": False,
    "gpu_layers": -1,
    "log_level": "info"
}

def create_runner(config: Dict[str, Any]) -> Union[CpuLlamaCpp, GpuLlamaCpp]:
    """
    Create the appropriate runner based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LlamaCpp instance (CPU or GPU)
    """
    # Create the models directory if it doesn't exist
    os.makedirs(config["models_dir"], exist_ok=True)
    
    # Set up logging
    log_level = config.get("log_level", "info").upper()
    setup_logger(
        "llama_cpp_runner",
        level=log_level,
        console=True,
        file=config.get("verbose", False)
    )
    
    # Check if GPU is enabled and available
    if config.get("enable_gpu", False):
        gpu_info = detect_gpu()
        if gpu_info["available"]:
            logger.info("GPU detected, using GPU-accelerated mode")
            logger.info(f"GPU: {gpu_info['gpus'][0]['name'] if gpu_info['gpus'] else 'Unknown'}")
            
            return GpuLlamaCpp(
                models_dir=config["models_dir"],
                cache_dir=config["cache_dir"],
                verbose=config["verbose"],
                timeout_minutes=config["timeout_minutes"],
                enable_gpu=True,
                gpu_layers=config["gpu_layers"]
            )
        else:
            logger.warning("GPU requested but not available, falling back to CPU mode")
    
    # CPU mode
    logger.info("Using CPU mode")
    return CpuLlamaCpp(
        models_dir=config["models_dir"],
        cache_dir=config["cache_dir"],
        verbose=config["verbose"],
        timeout_minutes=config["timeout_minutes"]
    )

def run_server(config: Dict[str, Any] = None) -> None:
    """
    Run the llama-cpp-runner server.
    
    Args:
        config: Configuration dictionary (optional)
    """
    # Merge provided config with defaults
    if config is None:
        config = {}
    
    cfg = {**DEFAULT_CONFIG, **config}
    
    # Set up logging
    log_level = cfg.get("log_level", "info").upper()
    setup_logger(
        "llama_cpp_runner",
        level=log_level,
        console=True,
        file=cfg.get("verbose", False)
    )
    
    # Print banner
    print("\n" + "=" * 60)
    print("🦙 LlamaCpp Runner Server")
    print("=" * 60 + "\n")
    
    # Print configuration
    logger.info(f"Models directory: {cfg['models_dir']}")
    logger.info(f"Cache directory: {cfg['cache_dir']}")
    logger.info(f"Timeout: {cfg['timeout_minutes']} minutes")
    logger.info(f"GPU enabled: {cfg['enable_gpu']}")
    if cfg["enable_gpu"]:
        logger.info(f"GPU layers: {cfg['gpu_layers']}")
    
    # Create the runner
    try:
        logger.info("Initializing LlamaCpp runner")
        runner = create_runner(cfg)
        
        # List available models
        models = runner.list_models()
        if models:
            logger.info(f"Found {len(models)} models: {', '.join(models)}")
        else:
            logger.warning("No models found in the models directory")
        
        # Create the API
        logger.info("Creating API")
        title = "LlamaCpp GPU API" if cfg["enable_gpu"] else "LlamaCpp API"
        app = create_api(
            runner,
            title=title,
            enable_gpu=cfg["enable_gpu"],
            gpu_layers=cfg["gpu_layers"]
        )
        
        # Start the server
        logger.info(f"Starting server at http://{cfg['host']}:{cfg['port']}")
        uvicorn.run(
            app,
            host=cfg["host"],
            port=cfg["port"],
            log_level=cfg["log_level"].lower()
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="LlamaCpp Runner Server")
    
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_CONFIG["models_dir"],
        help="Directory containing GGUF models"
    )
    
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CONFIG["cache_dir"],
        help="Directory for caching llama.cpp binaries"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_CONFIG["port"],
        help="Port to run the server on"
    )
    
    parser.add_argument(
        "--host",
        default=DEFAULT_CONFIG["host"],
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONFIG["timeout_minutes"],
        help="Timeout in minutes for shutting down idle servers"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default=DEFAULT_CONFIG["log_level"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration"
    )
    
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=DEFAULT_CONFIG["gpu_layers"],
        help="Number of layers to offload to GPU (-1 for all)"
    )
    
    return parser

def main() -> None:
    """
    Main entry point for the command-line interface.
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert args to config dictionary
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
    
    # Run the server
    run_server(config)

if __name__ == "__main__":
    main()