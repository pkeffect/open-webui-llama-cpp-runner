# Add these imports at the top of the file
import os
import platform
import requests
import zipfile
import json
import subprocess
import threading
import stat
import time
import socket

# Import our new modules
from llama_cpp_runner.logger import get_logger, log_method_call
from llama_cpp_runner.validation import validate_gguf_model, validate_llama_cpp_binaries

# Replace the simple print logging with proper logging
logger = get_logger("llama_cpp_runner.main")

# Then replace all self._log() calls with logger calls
# For example, replace:
# self._log("Installing llama.cpp binaries...")
# With:
# logger.info("Installing llama.cpp binaries...")

# Add proper method logging with the decorator
# For example:

@log_method_call(logger)
def _install_llama_cpp_binaries(self):
    """
    Download and install llama.cpp binaries.
    Returns:
        str: Path to the installed llama.cpp binaries.
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

# Add validation after downloading and installing binaries
@log_method_call(logger)
def chat_completion(self, body):
    """
    Handle chat completion requests.
    Args:
        body (dict): The payload for the chat completion request. It must contain the "model" key.
    Returns:
        dict or generator: Response from the server (non-streaming or streaming mode).
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

# Add validation when creating a server
@log_method_call(logger)
def _create_server(self, gguf_path):
    """
    Create a new LlamaCppServer instance for the given model.
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
    )

# Similarly, update other methods with proper logging and validation