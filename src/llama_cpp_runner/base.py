"""
Base classes for llama-cpp-runner

This module provides the foundation classes for both CPU and GPU versions
of llama-cpp-runner with unified functionality and clear extension points.
"""

import os
import platform
import requests
import zipfile
import json
import subprocess
import threading
import time
import socket
import shutil
from typing import Dict, List, Any, Optional, Union, Generator

# Import utility modules
from llama_cpp_runner.logger import get_logger, log_method_call
from llama_cpp_runner.validation import (
    validate_system_compatibility, 
    validate_gguf_model, 
    validate_llama_cpp_binaries
)

# Set up logger
logger = get_logger("llama_cpp_runner.base")

class BaseLlamaCppServer:
    """
    Base class for managing a llama.cpp server process.
    
    This class handles the basic functionality of starting, stopping,
    and communicating with a llama.cpp server process.
    """
    
    def __init__(
        self,
        llama_cpp_path: str = None,
        gguf_path: str = None,
        cache_dir: str = "./cache",
        hugging_face: bool = False,
        verbose: bool = False,
        timeout_minutes: int = 5,
    ):
        """
        Initialize the BaseLlamaCppServer.
        
        Args:
            llama_cpp_path: Path to the llama.cpp binaries
            gguf_path: Path to the GGUF model file
            cache_dir: Directory to cache llama.cpp binaries and related files
            hugging_face: Whether the model is on Hugging Face
            verbose: Enable verbose logging
            timeout_minutes: Timeout in minutes for shutting down idle servers
        """
        self.llama_cpp_path = llama_cpp_path
        self.gguf_path = gguf_path
        self.cache_dir = os.path.expanduser(cache_dir)
        self.hugging_face = hugging_face
        self.verbose = verbose
        self.timeout_minutes = timeout_minutes
        
        # Initialize server properties
        self.server_process = None
        self._server_url = None
        self.port = None
        self.last_request_time = time.time()
        self.shutdown_timer = None
        
        # Log initialization
        logger.info(f"Initialized BaseLlamaCppServer with model: {gguf_path}")
    
    @log_method_call(logger)
    def _find_available_port(self, start_port: int = 8080, end_port: int = 9000) -> Optional[int]:
        """
        Find an available port within the given range.
        
        Args:
            start_port: Starting port number
            end_port: Ending port number
            
        Returns:
            Available port or None if none found
        """
        for port in range(start_port, end_port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    logger.debug(f"Found available port: {port}")
                    return port
        
        logger.error(f"No available ports found between {start_port} and {end_port}")
        return None
    
    @log_method_call(logger)
    def _set_executable(self, path: str) -> None:
        """
        Make a file executable.
        
        Args:
            path: Path to the file
        """
        try:
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | 0o111)  # Add executable bit
            logger.debug(f"Made file executable: {path}")
        except Exception as e:
            logger.error(f"Failed to make file executable: {path} - {e}")
            raise
    
    @log_method_call(logger)
    def _start_server(self) -> None:
        """
        Start the llama-server process.
        
        This is a base implementation that should be extended by subclasses.
        """
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
        self.port = self._find_available_port()
        if self.port is None:
            raise RuntimeError("No available port found between 8080 and 9000.")

        logger.info(f"Starting server with binary: {server_binary}")
        logger.info(f"Using GGUF path: {self.gguf_path}")
        logger.info(f"Using port: {self.port}")
        
        # Base command for starting the server
        base_commands = [
            server_binary, 
            "-m", self.gguf_path, 
            "--port", str(self.port)
        ]
        
        # Get additional command line arguments from subclasses
        additional_args = self._get_server_args()
        commands = base_commands + additional_args
        
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
            
            if "listening on" in line:
                self._server_url = f"http://localhost:{self.port}"
                logger.info(f"Server is now accessible at {self._server_url}")
                break

        # If no server URL was found, raise an error
        if not self._server_url:
            self.kill()
            raise RuntimeError(f"Failed to start server. Check llama.cpp server output.")
    
    def _get_server_args(self) -> List[str]:
        """
        Get additional server arguments.
        
        This method should be overridden by subclasses to provide
        additional arguments for the server.
        
        Returns:
            List of additional command-line arguments
        """
        return []
    
    @log_method_call(logger)
    def chat_completion(self, body: Dict[str, Any]) -> Union[Dict[str, Any], Generator]:
        """
        Handle chat completion requests.
        
        Args:
            body: Request body containing chat messages
            
        Returns:
            Response from the server (dict for non-streaming, generator for streaming)
        """
        # Update last request time
        self.last_request_time = time.time()
        
        # Start server if not already running
        if not self._server_url:
            self._start_server()
        
        # Schedule shutdown timer
        self._schedule_shutdown()
        
        # Forward request to server
        streaming = body.get("stream", False)
        
        try:
            if streaming:
                return self._handle_streaming_request(body)
            else:
                return self._handle_non_streaming_request(body)
        except Exception as e:
            logger.error(f"Error during chat completion: {e}")
            if hasattr(e, "response") and hasattr(e.response, "text"):
                logger.error(f"Server response: {e.response.text}")
            raise
    
    def _handle_streaming_request(self, body: Dict[str, Any]) -> Generator:
        """
        Handle a streaming chat completion request.
        
        Args:
            body: Request body
            
        Returns:
            Generator yielding streaming responses
        """
        import requests
        
        url = f"{self._server_url}/completion"
        headers = {"Content-Type": "application/json"}
        
        # Create a streaming request
        with requests.post(url, json=body, headers=headers, stream=True) as response:
            if response.status_code != 200:
                error_msg = f"Server error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    # Skip the "data: " prefix if present
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    # Skip empty lines or "[DONE]"
                    if line and line != "[DONE]":
                        try:
                            yield line
                        except Exception as e:
                            logger.error(f"Error yielding stream line: {e}")
                            raise
    
    def _handle_non_streaming_request(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a non-streaming chat completion request.
        
        Args:
            body: Request body
            
        Returns:
            Server response
        """
        import requests
        
        url = f"{self._server_url}/completion"
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(url, json=body, headers=headers)
        
        if response.status_code != 200:
            error_msg = f"Server error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return response.json()
    
    @log_method_call(logger)
    def _schedule_shutdown(self) -> None:
        """
        Schedule a shutdown after the timeout period.
        """
        # Cancel any existing timer
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        
        # Schedule a new timer
        timeout_seconds = self.timeout_minutes * 60
        self.shutdown_timer = threading.Timer(timeout_seconds, self._check_idle)
        self.shutdown_timer.daemon = True
        self.shutdown_timer.start()
        
        logger.debug(f"Scheduled shutdown timer for {self.timeout_minutes} minutes")
    
    @log_method_call(logger)
    def _check_idle(self) -> None:
        """
        Check if the server is idle and shut it down if necessary.
        """
        current_time = time.time()
        idle_time = current_time - self.last_request_time
        timeout_seconds = self.timeout_minutes * 60
        
        if idle_time >= timeout_seconds:
            logger.info(f"Server idle for {idle_time:.1f} seconds, shutting down")
            self.kill()
        else:
            # If not idle enough, reschedule
            remaining = timeout_seconds - idle_time
            self.shutdown_timer = threading.Timer(remaining, self._check_idle)
            self.shutdown_timer.daemon = True
            self.shutdown_timer.start()
            
            logger.debug(f"Server not idle enough, rescheduling ({remaining:.1f}s remaining)")
    
    @log_method_call(logger)
    def kill(self) -> None:
        """
        Kill the server process.
        """
        if self.server_process:
            try:
                if self.shutdown_timer:
                    self.shutdown_timer.cancel()
                    self.shutdown_timer = None
                
                # Try graceful termination first
                self.server_process.terminate()
                
                # Wait a bit for graceful shutdown
                for _ in range(5):
                    if self.server_process.poll() is not None:
                        break
                    time.sleep(0.5)
                
                # Force kill if still running
                if self.server_process.poll() is None:
                    self.server_process.kill()
                
                logger.info("Server process terminated")
            except Exception as e:
                logger.error(f"Error killing server process: {e}")
            finally:
                self.server_process = None
                self._server_url = None
                self.port = None


class BaseLlamaCpp:
    """
    Base class for managing llama.cpp instances.
    
    This class handles downloading binaries, managing models,
    and creating server instances.
    """
    
    def __init__(
        self,
        models_dir: str,
        cache_dir: str = "~/.llama_cpp_runner",
        verbose: bool = False,
        timeout_minutes: int = 5,
        pinned_version: Optional[str] = None,
    ):
        """
        Initialize the BaseLlamaCpp.
        
        Args:
            models_dir: Directory where GGUF models are stored
            cache_dir: Directory to cache llama.cpp binaries and related files
            verbose: Enable verbose logging
            timeout_minutes: Timeout for shutting down idle servers
            pinned_version: Specific version of llama.cpp to use
        """
        # Store configuration
        self.models_dir = os.path.expanduser(models_dir)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.verbose = verbose
        self.timeout_minutes = timeout_minutes
        self.pinned_version = pinned_version
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Dictionary to store server instances
        self.servers = {}
        
        # Install llama.cpp binaries
        self.llama_cpp_path = self._install_llama_cpp_binaries()
        
        # Validate system compatibility
        self._validate_system()
        
        # Log initialization
        logger.info(f"Initialized BaseLlamaCpp with models_dir: {self.models_dir}")
        logger.info(f"Using llama.cpp binaries at: {self.llama_cpp_path}")
    
    def _validate_system(self) -> None:
        """
        Validate system compatibility.
        """
        # Check system compatibility
        compat_result = validate_system_compatibility()
        if not compat_result["compatible"]:
            issues = ", ".join(compat_result["issues"])
            logger.warning(f"System compatibility issues: {issues}")
        
        # Validate llama.cpp binaries
        bin_result = validate_llama_cpp_binaries(self.llama_cpp_path)
        if not bin_result["valid"]:
            issues = ", ".join(bin_result["issues"])
            logger.warning(f"llama.cpp binary issues: {issues}")
    
    @log_method_call(logger)
    def _internet_available(self) -> bool:
        """
        Check if internet connection is available.
        
        Returns:
            True if internet is available, False otherwise
        """
        try:
            requests.get("https://www.github.com", timeout=5)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False
    
    @log_method_call(logger)
    def _get_release_info(self) -> Dict[str, Any]:
        """
        Get information about llama.cpp releases.
        
        Returns:
            Release information dictionary
        """
        # Use pinned version if provided
        if self.pinned_version:
            url = f"https://api.github.com/repos/ggerganov/llama.cpp/releases/tags/{self.pinned_version}"
        else:
            url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get release info: {response.status_code} - {response.text}")
                raise RuntimeError(f"Failed to get release info: {response.status_code}")
        except Exception as e:
            logger.error(f"Error getting release info: {e}")
            raise
    
    @log_method_call(logger)
    def _get_appropriate_asset(self, assets: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Get the appropriate asset for the current system.
        
        Args:
            assets: List of asset metadata from the release
            
        Returns:
            Matching asset metadata, or None if no match found
        """
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Define priorities based on system and architecture
        if system == "linux":
            # Linux priorities
            if "x86_64" in machine or "amd64" in machine:
                priorities = ["linux-x64", "ubuntu-x64", "ubuntu-latest"]
            elif "aarch64" in machine or "arm64" in machine:
                priorities = ["linux-arm64", "ubuntu-arm64"]
            else:
                priorities = ["linux", "ubuntu"]
        elif system == "darwin":
            # macOS priorities
            if "arm64" in machine:
                priorities = ["macos-arm64", "macos-m1", "macos-m2"]
            else:
                priorities = ["macos-x64", "macos"]
        elif system == "windows":
            # Windows priorities
            priorities = ["win-x64", "win64", "windows"]
        else:
            # Unknown system
            logger.warning(f"Unknown system: {system}-{machine}, trying generic asset")
            priorities = [system]
        
        # Try to find an asset matching our priorities
        for priority in priorities:
            for asset in assets:
                name = asset["name"].lower()
                if priority in name and name.endswith(".zip"):
                    logger.info(f"Found appropriate asset: {asset['name']}")
                    return asset
        
        # If no appropriate asset was found, return None
        logger.warning(f"No appropriate asset found for {system}-{machine}")
        return None
    
    @log_method_call(logger)
    def _check_cache(self, release_info: Dict[str, Any], asset: Dict[str, str]) -> bool:
        """
        Check if the appropriate binary is already cached.
        
        Args:
            release_info: Release information
            asset: Asset metadata
            
        Returns:
            True if the binary is cached and up to date, False otherwise
        """
        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        
        if not os.path.exists(cache_info_path):
            return False
        
        try:
            with open(cache_info_path, "r") as f:
                cache_info = json.load(f)
            
            # Check if the cached version matches
            if cache_info.get("tag_name") == release_info["tag_name"] and \
               cache_info.get("asset_name") == asset["name"]:
                # Check if the binary exists
                llama_cpp_dir = os.path.join(self.cache_dir, "llama_cpp")
                server_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-server")
                
                if os.path.exists(server_bin):
                    logger.info(f"Using cached binary for {release_info['tag_name']}")
                    return True
            
            logger.info("Cached binary is outdated or missing")
            return False
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    @log_method_call(logger)
    def _update_cache_info(self, release_info: Dict[str, Any], asset: Dict[str, str]) -> None:
        """
        Update the cache information.
        
        Args:
            release_info: Release information
            asset: Asset metadata
        """
        cache_info = {
            "tag_name": release_info["tag_name"],
            "asset_name": asset["name"],
            "installed_at": time.time()
        }
        
        cache_info_path = os.path.join(self.cache_dir, "cache_info.json")
        
        try:
            with open(cache_info_path, "w") as f:
                json.dump(cache_info, f, indent=2)
            
            logger.info(f"Updated cache info for {release_info['tag_name']}")
        except Exception as e:
            logger.error(f"Error updating cache info: {e}")
    
    @log_method_call(logger)
    def _download_and_unzip(self, url: str, filename: str) -> None:
        """
        Download and unzip a file.
        
        Args:
            url: URL to download from
            filename: Name of the file
        """
        download_path = os.path.join(self.cache_dir, filename)
        extract_dir = os.path.join(self.cache_dir, "llama_cpp")
        
        # Create extract directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            # Download the file
            logger.info(f"Downloading {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(download_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename}")
            
            # Unzip the file
            logger.info(f"Extracting {filename}")
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted {filename}")
            
            # Set permissions on binaries
            bin_dir = os.path.join(extract_dir, "build", "bin")
            if os.path.exists(bin_dir):
                for bin_file in os.listdir(bin_dir):
                    bin_path = os.path.join(bin_dir, bin_file)
                    if os.path.isfile(bin_path):
                        self._set_executable(bin_path)
            
            # Delete the zip file
            os.remove(download_path)
        except Exception as e:
            logger.error(f"Error downloading and extracting: {e}")
            raise
    
    @log_method_call(logger)
    def _set_executable(self, path: str) -> None:
        """
        Make a file executable.
        
        Args:
            path: Path to the file
        """
        try:
            current_mode = os.stat(path).st_mode
            os.chmod(path, current_mode | 0o111)  # Add executable bit
            logger.debug(f"Made file executable: {path}")
        except Exception as e:
            logger.error(f"Failed to make file executable: {path} - {e}")
    
    @log_method_call(logger)
    def _install_llama_cpp_binaries(self) -> str:
        """
        Download and install llama.cpp binaries.
        
        Returns:
            Path to the installed llama.cpp binaries
        """
        logger.info("Installing llama.cpp binaries...")
        try:
            # Use pinned version if provided, otherwise fetch the latest release
            release_info = self._get_release_info()
            assets = release_info["assets"]
            asset = self._get_appropriate_asset(assets)
            if not asset:
                raise RuntimeError("No appropriate binary found for your system.")
            asset_name = asset["name"]

            # Check if cached binaries match the required version
            if self._check_cache(release_info, asset):
                logger.info("Using cached llama.cpp binaries.")
            else:
                if not self._internet_available():
                    raise RuntimeError(
                        "No cached binary available and unable to fetch from the internet."
                    )
                self._download_and_unzip(asset["browser_download_url"], asset_name)
                self._update_cache_info(release_info, asset)

        except Exception as e:
            logger.error(f"Error during binary installation: {e}", exc_info=True)
            raise

        return os.path.join(self.cache_dir, "llama_cpp")
    
    @log_method_call(logger)
    def list_models(self) -> List[str]:
        """
        List all available GGUF models.
        
        Returns:
            List of model filenames
        """
        models = []
        
        try:
            if os.path.exists(self.models_dir) and os.path.isdir(self.models_dir):
                for file in os.listdir(self.models_dir):
                    if file.endswith(".gguf"):
                        models.append(file)
                
                logger.info(f"Found {len(models)} models")
            else:
                logger.warning(f"Models directory does not exist: {self.models_dir}")
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models
    
    @log_method_call(logger)
    def chat_completion(self, body: Dict[str, Any]) -> Union[Dict[str, Any], Generator]:
        """
        Handle chat completion requests.
        
        Args:
            body: The payload for the chat completion request. It must contain the "model" key.
            
        Returns:
            Response from the server (non-streaming or streaming mode).
        """
        if "model" not in body:
            logger.error("The request body must contain a 'model' key.")
            raise ValueError("The request body must contain a 'model' key.")
        
        model_name = body["model"]
        gguf_path = os.path.join(self.models_dir, model_name)
        
        # Add validation for the model file
        validation_result = validate_gguf_model(gguf_path)
        if not validation_result["valid"]:
            issues = "; ".join(validation_result["issues"])
            logger.error(f"Model validation failed: {issues}")
            raise FileNotFoundError(f"Model file invalid or not found: {gguf_path}. Issues: {issues}")
        
        # Check if the server for this model is already running
        if model_name not in self.servers or not self.servers[model_name]._server_url:
            logger.info(f"Initializing a new server for model: {model_name}")
            self.servers[model_name] = self._create_server(gguf_path)
        
        server = self.servers[model_name]
        return server.chat_completion(body)
    
    @log_method_call(logger)
    def _create_server(self, gguf_path: str):
        """
        Create a new server instance for the given model.
        
        This method should be overridden by subclasses to provide
        the appropriate server class.
        
        Args:
            gguf_path: Path to the GGUF model file
            
        Returns:
            A new server instance
        """
        raise NotImplementedError("Subclasses must implement _create_server")