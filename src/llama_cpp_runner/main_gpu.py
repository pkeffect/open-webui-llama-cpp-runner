# Add these imports at the top of the file
"""
GPU-accelerated version of the LlamaCpp runner
This extends the original LlamaCpp class with GPU capabilities
"""

import os
import subprocess
import platform

# Import our new modules
from llama_cpp_runner.logger import get_logger, log_method_call
from llama_cpp_runner.gpu import detect_gpu, optimize_gpu_params, setup_gpu_environment, get_gpu_command_args

# Import the original LlamaCpp class
from llama_cpp_runner.main import LlamaCpp as OriginalLlamaCpp
from llama_cpp_runner.main import LlamaCppServer as OriginalLlamaCppServer

# Set up logger
logger = get_logger("llama_cpp_runner.gpu")

class LlamaCppServer(OriginalLlamaCppServer):
    """GPU-accelerated version of LlamaCppServer"""
    
    def __init__(
        self,
        llama_cpp_path=None,
        gguf_path=None,
        cache_dir="./cache",
        hugging_face=False,
        verbose=False,
        timeout_minutes=5,
        enable_gpu=True,
        gpu_layers=-1,
    ):
        """
        Initialize the GPU-accelerated LlamaCppServer.
        
        Args:
            llama_cpp_path (str): Path to the llama.cpp binaries.
            gguf_path (str): Path to the GGUF model file.
            cache_dir (str): Directory to store llama.cpp binaries and related files.
            hugging_face (bool): Whether the model is hosted on Hugging Face.
            verbose (bool): Enable verbose logging.
            timeout_minutes (int): Timeout duration for shutting down idle servers.
            enable_gpu (bool): Whether to enable GPU acceleration.
            gpu_layers (int): Number of layers to offload to GPU. -1 means all layers.
        """
        # Store GPU-specific attributes
        self.enable_gpu = enable_gpu
        self.gpu_layers = gpu_layers
        
        # Initialize the original server with standard parameters
        super().__init__(
            llama_cpp_path=llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=cache_dir,
            hugging_face=hugging_face,
            verbose=verbose,
            timeout_minutes=timeout_minutes
        )
        
        logger.info(f"GPU enabled: {self.enable_gpu}")
        if self.enable_gpu:
            logger.info(f"GPU layers: {self.gpu_layers}")
    
    @log_method_call(logger)
    def _start_server(self):
        """Start the llama-server with GPU support."""
        if not self.gguf_path or (
            not self.hugging_face and not os.path.exists(self.gguf_path)
        ):
            raise ValueError(
                f"GGUF model path is not specified or invalid: {self.gguf_path}"
            )

        server_binary = os.path.join(
            self.llama_cpp_path, "build", "bin", "llama-server"
        )
        if not os.path.exists(server_binary):
            raise FileNotFoundError(f"Server binary not found: {server_binary}")

        # Ensure the binary is executable
        self._set_executable(server_binary)

        # Find an available port
        self.port = self._find_available_port(start_port=10000)
        if self.port is None:
            raise RuntimeError("No available port found between 10000 and 11000.")

        logger.info(f"Starting server with binary: {server_binary}")
        logger.info(f"Using GGUF path: {self.gguf_path}")
        logger.info(f"Using port: {self.port}")
        logger.info(f"GPU enabled: {self.enable_gpu}")
        
        # Use our new GPU utilities to set up command arguments
        base_commands = [server_binary, "-m", self.gguf_path, "--port", str(self.port)]
        
        if self.enable_gpu:
            # Get GPU-specific command arguments
            gpu_args = get_gpu_command_args(self.gguf_path, self.enable_gpu, self.gpu_layers)
            commands = base_commands + gpu_args
            
            # Set up environment variables for GPU
            gpu_info = detect_gpu()
            if gpu_info["available"]:
                # Get model size to optimize GPU parameters
                model_size_mb = os.path.getsize(self.gguf_path) / (1024 * 1024)
                gpu_params = optimize_gpu_params(model_size_mb, gpu_info["gpus"])
                setup_gpu_environment(gpu_params)
                
                logger.info(f"GPU configuration: {gpu_params}")
            else:
                logger.warning("GPU requested but not available. Running in CPU mode.")
                self.enable_gpu = False
                commands = base_commands
        else:
            commands = base_commands

        # Run the server process
        env = os.environ.copy()
        self.server_process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env
        )

        # Capture and log all server output
        self._server_url = None
        for line in iter(self.server_process.stdout.readline, ""):
            line = line.strip()
            logger.debug(f"SERVER OUTPUT: {line}")
            
            # Look for specific GPU-related messages
            if any(gpu_keyword in line.lower() for gpu_keyword in [
                'gpu', 'cuda', 'device', 'layer', 'tensor', 'offload'
            ]):
                logger.info(f"GPU-RELATED MESSAGE: {line}")
            
            if "listening on" in line:
                self._server_url = f"http://localhost:{self.port}"
                logger.info(f"Server is now accessible at {self._server_url}")
                break

        # If no server URL was found, raise an error
        if not self._server_url:
            raise RuntimeError(f"Failed to start server")

class LlamaCpp(OriginalLlamaCpp):
    """GPU-accelerated version of LlamaCpp"""
    
    def __init__(
        self,
        models_dir,
        cache_dir="~/.llama_cpp_runner",
        verbose=False,
        timeout_minutes=5,
        pinned_version=None,
        enable_gpu=True,
        gpu_layers=-1,
    ):
        """
        Initialize the GPU-accelerated LlamaCpp class.
        
        Args:
            models_dir (str): Directory where GGUF models are stored.
            cache_dir (str): Directory to store llama.cpp binaries and related assets.
            verbose (bool): Whether to enable verbose logging.
            timeout_minutes (int): Timeout for shutting down idle servers.
            pinned_version (str or None): Pinned release version of llama.cpp binaries.
            enable_gpu (bool): Whether to enable GPU acceleration.
            gpu_layers (int): Number of layers to offload to GPU. -1 means all layers.
        """
        # Store GPU-specific attributes
        self.enable_gpu = enable_gpu
        self.gpu_layers = gpu_layers
        
        # Check GPU availability upfront
        if self.enable_gpu:
            gpu_info = detect_gpu()
            if not gpu_info["available"]:
                logger.warning("GPU support requested but no GPU detected. Falling back to CPU mode.")
                self.enable_gpu = False
            else:
                logger.info(f"GPU detected: {gpu_info.get('gpus', [{}])[0].get('name', 'Unknown')}")
                logger.info(f"CUDA version: {gpu_info.get('cuda_version', 'Unknown')}")
        
        # Initialize the original LlamaCpp with standard parameters
        super().__init__(
            models_dir=models_dir,
            cache_dir=cache_dir,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
            pinned_version=pinned_version
        )
    
    @log_method_call(logger)
    def _create_server(self, gguf_path):
        """
        Create a new GPU-enabled LlamaCppServer instance for the given model.
        
        Args:
            gguf_path (str): Path to the GGUF model file.
            
        Returns:
            LlamaCppServer: A new server instance.
        """
        return LlamaCppServer(
            llama_cpp_path=self.llama_cpp_path,
            gguf_path=gguf_path,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
            timeout_minutes=self.timeout_minutes,
            enable_gpu=self.enable_gpu,
            gpu_layers=self.gpu_layers,
        )
    
    @log_method_call(logger)
    def _get_appropriate_asset(self, assets):
        """
        Select the appropriate binary asset for the current system.
        Overridden to prefer CUDA-enabled binaries when GPU is enabled.
        
        Args:
            assets (list): List of asset metadata from the release.
            
        Returns:
            dict or None: Matching asset metadata, or None if no match found.
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        processor = platform.processor()
        
        # For Linux with NVIDIA GPU, prefer CUDA builds if GPU is enabled
        if system == "linux" and self.enable_gpu:
            return next((a for a in assets if "ubuntu-cuda" in a["name"]), None) or \
                   next((a for a in assets if "ubuntu-x64" in a["name"]), None)
        
        # Fall back to original implementation for other cases
        return super()._get_appropriate_asset(assets)