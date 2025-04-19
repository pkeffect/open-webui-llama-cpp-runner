"""
GPU-accelerated implementation of llama-cpp-runner

This module provides classes optimized for GPU acceleration.
"""

import os
import subprocess
from typing import List, Dict, Any, Optional, Union, Generator

from llama_cpp_runner.base import BaseLlamaCpp, BaseLlamaCppServer
from llama_cpp_runner.logger import get_logger, log_method_call
from llama_cpp_runner.utils import detect_gpu, setup_gpu_environment, optimize_gpu_params

# Set up logger
logger = get_logger("llama_cpp_runner.gpu")

class GpuLlamaCppServer(BaseLlamaCppServer):
    """
    LlamaCppServer with GPU acceleration.
    
    This class extends the base server implementation with GPU-specific
    optimizations and configurations.
    """
    
    def __init__(
        self,
        llama_cpp_path: str = None,
        gguf_path: str = None,
        cache_dir: str = "./cache",
        hugging_face: bool = False,
        verbose: bool = False,
        timeout_minutes: int = 5,
        enable_gpu: bool = True,
        gpu_layers: int = -1,
    ):
        """
        Initialize the GPU-accelerated LlamaCppServer.
        
        Args:
            llama_cpp_path: Path to the llama.cpp binaries
            gguf_path: Path to the GGUF model file
            cache_dir: Directory to cache llama.cpp binaries and related files
            hugging_face: Whether the model is on Hugging Face
            verbose: Enable verbose logging
            timeout_minutes: Timeout in minutes for shutting down idle servers
            enable_gpu: Whether to enable GPU acceleration
            gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        super().__init__(
            llama_cpp_path=llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=cache_dir,
            hugging_face=hugging_face,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
        )
        
        # GPU-specific parameters
        self.enable_gpu = enable_gpu
        self.gpu_layers = gpu_layers
        
        # Detect GPU availability
        if self.enable_gpu:
            self.gpu_info = detect_gpu()
            if not self.gpu_info["available"]:
                logger.warning("GPU acceleration requested but no GPU detected. Using CPU only.")
                self.enable_gpu = False
        
        logger.info(f"Initialized GPU-accelerated server (GPU enabled: {self.enable_gpu})")
        if self.enable_gpu:
            logger.info(f"GPU layers: {self.gpu_layers}")
    
    def _get_server_args(self) -> List[str]:
        """
        Get GPU-specific server arguments.
        
        Returns:
            List of additional command-line arguments for GPU acceleration
        """
        args = [
            "--embedding",   # Enable embeddings
            "--ctx-size", "2048",  # Reasonable context size
        ]
        
        if self.verbose:
            args.append("--verbose")
        
        # Add GPU-specific args if GPU is enabled
        if self.enable_gpu:
            # Set up environment for GPU
            if self.gguf_path:
                try:
                    model_size_mb = os.path.getsize(self.gguf_path) / (1024 * 1024)
                    gpu_params = optimize_gpu_params(model_size_mb, self.gpu_info["gpus"])
                    setup_gpu_environment(gpu_params)
                    
                    # Add GPU-specific arguments
                    gpu_layers = self.gpu_layers if self.gpu_layers > 0 else gpu_params.get("gpu_layers", -1)
                    
                    args.extend([
                        "--n-gpu-layers", str(gpu_layers),
                        "--gpu-layers", str(gpu_layers),  # For older versions
                    ])
                    
                    logger.info(f"Using {gpu_layers} GPU layers")
                except Exception as e:
                    logger.error(f"Error configuring GPU: {e}")
                    logger.warning("Falling back to CPU only")
                    self.enable_gpu = False
        
        return args


class GpuLlamaCpp(BaseLlamaCpp):
    """
    LlamaCpp implementation with GPU acceleration.
    
    This class extends the base implementation with GPU-specific
    features and optimizations.
    """
    
    def __init__(
        self,
        models_dir: str,
        cache_dir: str = "~/.llama_cpp_runner",
        verbose: bool = False,
        timeout_minutes: int = 5,
        pinned_version: Optional[str] = None,
        enable_gpu: bool = True,
        gpu_layers: int = -1,
    ):
        """
        Initialize the GPU-accelerated LlamaCpp.
        
        Args:
            models_dir: Directory where GGUF models are stored
            cache_dir: Directory to cache llama.cpp binaries and related files
            verbose: Enable verbose logging
            timeout_minutes: Timeout for shutting down idle servers
            pinned_version: Specific version of llama.cpp to use
            enable_gpu: Whether to enable GPU acceleration
            gpu_layers: Number of layers to offload to GPU (-1 for all)
        """
        # GPU-specific parameters
        self.enable_gpu = enable_gpu
        self.gpu_layers = gpu_layers
        
        # Check GPU availability
        if self.enable_gpu:
            self.gpu_info = detect_gpu()
            if not self.gpu_info["available"]:
                logger.warning("GPU acceleration requested but no GPU detected. Using CPU only.")
                self.enable_gpu = False
            else:
                logger.info(f"Detected GPU: {self.gpu_info['gpus'][0]['name']}")
                if "cuda_version" in self.gpu_info:
                    logger.info(f"CUDA version: {self.gpu_info['cuda_version']}")
        
        # Initialize base class
        super().__init__(
            models_dir=models_dir,
            cache_dir=cache_dir,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
            pinned_version=pinned_version,
        )
        
        logger.info(f"Initialized GPU-accelerated LlamaCpp (GPU enabled: {self.enable_gpu})")
        if self.enable_gpu:
            logger.info(f"GPU layers: {self.gpu_layers}")
    
    @log_method_call(logger)
    def _get_appropriate_asset(self, assets: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Get the appropriate binary asset for the current system.
        Prefer CUDA-enabled binaries when GPU is enabled.
        
        Args:
            assets: List of asset metadata from the release
            
        Returns:
            Matching asset metadata, or None if no match found
        """
        # If GPU is enabled, prefer CUDA builds
        if self.enable_gpu:
            system = "linux"  # Currently CUDA builds are only available for Linux
            for asset in assets:
                name = asset["name"].lower()
                if ("cuda" in name or "gpu" in name) and name.endswith(".zip"):
                    logger.info(f"Found GPU-enabled asset: {asset['name']}")
                    return asset
        
        # Fall back to default implementation
        return super()._get_appropriate_asset(assets)
    
    @log_method_call(logger)
    def _create_server(self, gguf_path: str) -> GpuLlamaCppServer:
        """
        Create a new GPU-accelerated server instance for the given model.
        
        Args:
            gguf_path: Path to the GGUF model file
            
        Returns:
            A new GPU-accelerated server instance
        """
        return GpuLlamaCppServer(
            llama_cpp_path=self.llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
            timeout_minutes=self.timeout_minutes,
            enable_gpu=self.enable_gpu,
            gpu_layers=self.gpu_layers,
        )