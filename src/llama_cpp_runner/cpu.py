"""
CPU implementation of llama-cpp-runner

This module provides classes optimized for CPU usage without GPU acceleration.
"""

import os
from typing import List, Dict, Any, Optional, Union, Generator

from llama_cpp_runner.base import BaseLlamaCpp, BaseLlamaCppServer
from llama_cpp_runner.logger import get_logger, log_method_call

# Set up logger
logger = get_logger("llama_cpp_runner.cpu")

class LlamaCppServer(BaseLlamaCppServer):
    """
    LlamaCppServer optimized for CPU usage.
    
    This class extends the base server implementation with CPU-specific
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
        n_threads: Optional[int] = None,
    ):
        """
        Initialize the CPU-optimized LlamaCppServer.
        
        Args:
            llama_cpp_path: Path to the llama.cpp binaries
            gguf_path: Path to the GGUF model file
            cache_dir: Directory to cache llama.cpp binaries and related files
            hugging_face: Whether the model is on Hugging Face
            verbose: Enable verbose logging
            timeout_minutes: Timeout in minutes for shutting down idle servers
            n_threads: Number of CPU threads to use (defaults to auto-detect)
        """
        super().__init__(
            llama_cpp_path=llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=cache_dir,
            hugging_face=hugging_face,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
        )
        
        # CPU-specific parameters
        self.n_threads = n_threads or os.cpu_count() or 4
        
        logger.info(f"Initialized CPU-optimized server with {self.n_threads} threads")
    
    def _get_server_args(self) -> List[str]:
        """
        Get CPU-specific server arguments.
        
        Returns:
            List of additional command-line arguments for CPU optimization
        """
        args = [
            "--n-threads", str(self.n_threads),
            "--embedding",   # Enable embeddings
            "--ctx-size", "2048",  # Reasonable context size
            # Add other CPU-specific arguments here
        ]
        
        if self.verbose:
            args.append("--verbose")
        
        return args


class LlamaCpp(BaseLlamaCpp):
    """
    LlamaCpp implementation optimized for CPU usage.
    
    This class extends the base implementation with CPU-specific
    features and optimizations.
    """
    
    def __init__(
        self,
        models_dir: str,
        cache_dir: str = "~/.llama_cpp_runner",
        verbose: bool = False,
        timeout_minutes: int = 5,
        pinned_version: Optional[str] = None,
        n_threads: Optional[int] = None,
    ):
        """
        Initialize the CPU-optimized LlamaCpp.
        
        Args:
            models_dir: Directory where GGUF models are stored
            cache_dir: Directory to cache llama.cpp binaries and related files
            verbose: Enable verbose logging
            timeout_minutes: Timeout for shutting down idle servers
            pinned_version: Specific version of llama.cpp to use
            n_threads: Number of CPU threads to use (defaults to auto-detect)
        """
        # Initialize base class
        super().__init__(
            models_dir=models_dir,
            cache_dir=cache_dir,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
            pinned_version=pinned_version,
        )
        
        # CPU-specific parameters
        self.n_threads = n_threads or os.cpu_count() or 4
        
        logger.info(f"Initialized CPU-optimized LlamaCpp with {self.n_threads} threads")
    
    @log_method_call(logger)
    def _create_server(self, gguf_path: str) -> LlamaCppServer:
        """
        Create a new CPU-optimized server instance for the given model.
        
        Args:
            gguf_path: Path to the GGUF model file
            
        Returns:
            A new CPU-optimized server instance
        """
        return LlamaCppServer(
            llama_cpp_path=self.llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
            timeout_minutes=self.timeout_minutes,
            n_threads=self.n_threads,
        )