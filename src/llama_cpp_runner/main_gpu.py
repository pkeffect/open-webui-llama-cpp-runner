"""
GPU-accelerated version of the LlamaCpp runner
This extends the original LlamaCpp class with GPU capabilities
"""

import os
import subprocess
import platform

# Import the original LlamaCpp class
from llama_cpp_runner.main import LlamaCpp as OriginalLlamaCpp
from llama_cpp_runner.main import LlamaCppServer as OriginalLlamaCppServer


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
        
        if self.verbose:
            print(f"[LlamaCppServer] GPU enabled: {self.enable_gpu}")
            if self.enable_gpu:
                print(f"[LlamaCppServer] GPU layers: {self.gpu_layers}")
    
def _start_server(self):
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

    self._log(f"Starting server with binary: {server_binary}")
    self._log(f"Using GGUF path: {self.gguf_path}")
    self._log(f"Using port: {self.port}")
    self._log(f"GPU enabled: {self.enable_gpu}")
    
    # IMPORTANT: Force GPU usage with explicit parameters
    commands = [
        server_binary, 
        "-m", self.gguf_path, 
        "--port", str(self.port),
        "--n-gpu-layers", "-1",  # Force maximum GPU layers
        "-ngl", "999",            # Alternative flag for GPU layers
        "--tensor-split", "1.0",  # Force full GPU usage
        "--gpu-layers", "999"     # Additional flag for GPU layers
    ]

    # Additional diagnostic logging
    try:
        # Check GPU capabilities using subprocess
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=count,name,memory.total", "--format=csv,noheader,nounits"], universal_newlines=True).strip()
        self._log(f"GPU Info: {gpu_info}")
    except Exception as e:
        self._log(f"Error getting GPU info: {e}")

    # Run the server with verbose output to force GPU usage
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Explicitly set GPU device
    env['CUDA_DEVICE'] = '0'

    self.server_process = subprocess.Popen(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )

    # Capture and log all server output for GPU-related messages
    gpu_detection_log = []
    for line in iter(self.server_process.stdout.readline, ""):
        line = line.strip()
        self._log(f"SERVER OUTPUT: {line}")
        gpu_detection_log.append(line)
        
        # Look for specific GPU-related messages
        if any(gpu_keyword in line.lower() for gpu_keyword in [
            'gpu', 'cuda', 'device', 'layer', 'tensor', 'offload'
        ]):
            self._log(f"GPU-RELATED MESSAGE: {line}")
        
        if "listening on" in line:
            self._server_url = f"http://localhost:{self.port}"
            self._log(f"Server is now accessible at {self._server_url}")
            break

    # If no server URL was found, raise an error with diagnostic information
    if not self._server_url:
        raise RuntimeError(f"Failed to start server. Diagnostic log:\n{'\n'.join(gpu_detection_log)}")

def _get_gpu_capabilities(self):
    """Comprehensive GPU capability check."""
    gpu_info = {
        'cuda_available': False,
        'device_count': 0,
        'device_name': None,
        'memory_total': 0
    }
    
    try:
        # Use subprocess to get detailed GPU info
        nvidia_output = subprocess.check_output(["nvidia-smi", "--query-gpu=count,name,memory.total", "--format=csv,noheader,nounits"], universal_newlines=True)
        gpu_details = nvidia_output.strip().split(", ")
        
        if len(gpu_details) >= 3:
            gpu_info['cuda_available'] = True
            gpu_info['device_count'] = int(gpu_details[0])
            gpu_info['device_name'] = gpu_details[1]
            gpu_info['memory_total'] = int(gpu_details[2])
    except Exception as e:
        self._log(f"Error getting GPU capabilities: {e}")
    
    return gpu_info
            

        # Add GPU parameters if enabled
        if self.enable_gpu:
            # Check if CUDA is available
            self._log("Checking for CUDA availability...")
            cuda_available = self._check_gpu_available()
            if cuda_available:
                self._log("CUDA is available. Adding GPU parameters.")
                if self.gpu_layers == -1:
                    # Get the number of layers from the model metadata if available
                    # For now, use a reasonable default based on model size
                    model_size_mb = os.path.getsize(self.gguf_path) / (1024 * 1024)
                    estimated_layers = -1
                    # Adjust based on model size
                    if model_size_mb > 10000:  # Larger than 10GB
                        estimated_layers = 80  # Likely a 70B model
                    elif model_size_mb > 5000:  # Larger than 5GB
                        estimated_layers = 40  # Likely a 13B model
                    
                    self._log(f"Auto-detecting number of layers: {estimated_layers}")
                    commands.extend(["--n-gpu-layers", str(estimated_layers)])
                else:
                    commands.extend(["--n-gpu-layers", str(self.gpu_layers)])
            else:
                self._log("CUDA is not available. Continuing without GPU acceleration.")

        self.server_process = subprocess.Popen(
            commands,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Wait for the server to confirm it is ready by monitoring its output
        self._server_url = None
        for line in iter(self.server_process.stdout.readline, ""):
            self._log(line.strip())
            if "listening on" in line:
                self._server_url = f"http://localhost:{self.port}"
                self._log(f"Server is now accessible at {self._server_url}")
                break

        if not self._server_url:
            raise RuntimeError("Failed to confirm server is running.")
    
    def _setup_library_paths(self):
        """Set up shared library paths for llama.cpp."""
        if platform.system() == "Linux":
            # Check for libllama.so in build/lib directory
            lib_path = os.path.join(self.llama_cpp_path, "build", "lib", "libllama.so")
            if not os.path.exists(lib_path):
                self._log(f"Warning: Could not find {lib_path}")
                
                # Try to find libllama.so in other locations
                alternative_paths = [
                    os.path.join(self.llama_cpp_path, "build", "libllama.so"),
                    os.path.join(self.llama_cpp_path, "libllama.so"),
                    os.path.join(self.llama_cpp_path, "build", "bin", "libllama.so"),
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        self._log(f"Found libllama.so at {alt_path}")
                        
                        # Create lib directory if it doesn't exist
                        lib_dir = os.path.join(self.llama_cpp_path, "build", "lib")
                        os.makedirs(lib_dir, exist_ok=True)
                        
                        # Create symbolic link
                        try:
                            os.symlink(alt_path, lib_path)
                            self._log(f"Created symbolic link from {alt_path} to {lib_path}")
                            break
                        except OSError as e:
                            self._log(f"Failed to create symbolic link: {e}")
            
            # Add library path to environment
            lib_dir = os.path.dirname(lib_path)
            bin_dir = os.path.join(self.llama_cpp_path, "build", "bin")
            
            # Update LD_LIBRARY_PATH environment variable
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in current_ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{bin_dir}:{current_ld_path}"
                self._log(f"Updated LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    
    def _check_gpu_available(self):
        """Check if NVIDIA GPU is available for use."""
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError):
            return False


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
        
        # Initialize the original LlamaCpp with standard parameters
        super().__init__(
            models_dir=models_dir,
            cache_dir=cache_dir,
            verbose=verbose,
            timeout_minutes=timeout_minutes,
            pinned_version=pinned_version
        )
        
        if self.verbose:
            print(f"[LlamaCpp] GPU enabled: {self.enable_gpu}")
            if self.enable_gpu:
                print(f"[LlamaCpp] GPU layers: {self.gpu_layers}")
    
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