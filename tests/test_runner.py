"""
Test suite for llama-cpp-runner

This module provides comprehensive tests for the llama-cpp-runner package.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import time
import platform
from unittest import mock
import pytest

# Add project root to path to allow importing llama_cpp_runner
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock data for tests
MOCK_RELEASE_INFO = {
    "tag_name": "v1.0.0",
    "assets": [
        {
            "name": "llama-linux-x64.zip",
            "browser_download_url": "https://example.com/llama-linux-x64.zip"
        },
        {
            "name": "llama-win-x64.zip",
            "browser_download_url": "https://example.com/llama-win-x64.zip"
        },
        {
            "name": "llama-macos-arm64.zip",
            "browser_download_url": "https://example.com/llama-macos-arm64.zip"
        }
    ]
}

class TestBase(unittest.TestCase):
    """Base class for llama-cpp-runner tests"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.temp_dir, "models")
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a mock GGUF model file
        self.model_path = os.path.join(self.models_dir, "test_model.gguf")
        with open(self.model_path, "wb") as f:
            # Write the GGUF magic bytes
            f.write(b"GGUF")
            # Add some dummy content
            f.write(b"\x00" * 1024)
        
        # Create a mock llama.cpp directory structure
        llama_cpp_dir = os.path.join(self.cache_dir, "llama_cpp")
        bin_dir = os.path.join(llama_cpp_dir, "build", "bin")
        lib_dir = os.path.join(llama_cpp_dir, "build", "lib")
        
        os.makedirs(bin_dir, exist_ok=True)
        os.makedirs(lib_dir, exist_ok=True)
        
        # Create a mock server binary
        server_bin = os.path.join(bin_dir, "llama-server")
        with open(server_bin, "wb") as f:
            f.write(b"#!/bin/sh\necho 'Server is listening on http://localhost:8080'\nsleep 10\n")
        
        # Make it executable
        os.chmod(server_bin, 0o755)
        
        # Create a mock library
        if platform.system() != "Windows":
            lib_path = os.path.join(lib_dir, "libllama.so")
            with open(lib_path, "wb") as f:
                f.write(b"\x00" * 1024)
        
        # Create a mock cache info file
        cache_info = {
            "tag_name": "v1.0.0",
            "asset_name": "llama-linux-x64.zip" if platform.system() == "Linux" else
                         "llama-win-x64.zip" if platform.system() == "Windows" else
                         "llama-macos-arm64.zip"
        }
        
        with open(os.path.join(self.cache_dir, "cache_info.json"), "w") as f:
            json.dump(cache_info, f)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

class TestLlamaCpp(TestBase):
    """Tests for the LlamaCpp class"""
    
    @mock.patch("llama_cpp_runner.main.requests.get")
    def test_get_release_info(self, mock_get):
        """Test fetching release information"""
        from llama_cpp_runner.main import LlamaCpp
        
        # Set up mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = MOCK_RELEASE_INFO
        mock_get.return_value = mock_response
        
        # Create an instance of LlamaCpp
        llama = LlamaCpp(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            verbose=True
        )
        
        # Patch internet_available to return True
        with mock.patch.object(llama, "_internet_available", return_value=True):
            # Call the method
            result = llama._get_release_info()
        
        # Verify the result
        self.assertEqual(result["tag_name"], "v1.0.0")
        self.assertEqual(len(result["assets"]), 3)
    
    def test_list_models(self):
        """Test listing available models"""
        from llama_cpp_runner.main import LlamaCpp
        
        # Create an instance of LlamaCpp
        llama = LlamaCpp(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            verbose=True
        )
        
        # Call the method
        models = llama.list_models()
        
        # Verify the result
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0], "test_model.gguf")
    
    @mock.patch("llama_cpp_runner.main.LlamaCppServer")
    def test_chat_completion(self, mock_server_class):
        """Test chat completion functionality"""
        from llama_cpp_runner.main import LlamaCpp
        
        # Set up mock server
        mock_server = mock.Mock()
        mock_server.chat_completion.return_value = {"generated_text": "Test response"}
        mock_server_class.return_value = mock_server
        
        # Create an instance of LlamaCpp
        llama = LlamaCpp(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            verbose=True
        )
        
        # Call the method
        result = llama.chat_completion({
            "model": "test_model.gguf",
            "messages": [{"role": "user", "content": "Hello"}]
        })
        
        # Verify the result
        self.assertEqual(result, {"generated_text": "Test response"})
        mock_server.chat_completion.assert_called_once()

class TestLlamaCppServer(TestBase):
    """Tests for the LlamaCppServer class"""
    
    @mock.patch("llama_cpp_runner.main.subprocess.Popen")
    def test_start_server(self, mock_popen):
        """Test starting the server"""
        from llama_cpp_runner.main import LlamaCppServer
        
        # Set up mock process
        mock_process = mock.Mock()
        mock_process.stdout.readline.side_effect = [
            "Loading model test_model.gguf",
            "Server is listening on http://localhost:8080",
            ""  # End of readline loop
        ]
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Create a server instance
        server = LlamaCppServer(
            llama_cpp_path=os.path.join(self.cache_dir, "llama_cpp"),
            gguf_path=self.model_path,
            cache_dir=self.cache_dir,
            verbose=True
        )
        
        # Verify the server URL
        self.assertEqual(server._server_url, "http://localhost:8080")
        
        # Clean up
        server.kill()
    
    @mock.patch("llama_cpp_runner.main.requests.post")
    @mock.patch("llama_cpp_runner.main.subprocess.Popen")
    def test_chat_completion(self, mock_popen, mock_post):
        """Test chat completion functionality"""
        from llama_cpp_runner.main import LlamaCppServer
        
        # Set up mock process
        mock_process = mock.Mock()
        mock_process.stdout.readline.side_effect = [
            "Loading model test_model.gguf",
            "Server is listening on http://localhost:8080",
            ""  # End of readline loop
        ]
        mock_process.poll.return_value = None  # Process is running
        mock_popen.return_value = mock_process
        
        # Set up mock response
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"generated_text": "Test response"}
        mock_post.return_value = mock_response
        
        # Create a server instance
        server = LlamaCppServer(
            llama_cpp_path=os.path.join(self.cache_dir, "llama_cpp"),
            gguf_path=self.model_path,
            cache_dir=self.cache_dir,
            verbose=True
        )
        
        # Call the method
        result = server.chat_completion({
            "messages": [{"role": "user", "content": "Hello"}]
        })
        
        # Verify the result
        self.assertEqual(result, {"generated_text": "Test response"})
        
        # Clean up
        server.kill()

@pytest.fixture
def llama_env():
    """Pytest fixture for llama-cpp-runner tests"""
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    models_dir = os.path.join(temp_dir, "models")
    cache_dir = os.path.join(temp_dir, "cache")
    
    # Create directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a mock GGUF model file
    model_path = os.path.join(models_dir, "test_model.gguf")
    with open(model_path, "wb") as f:
        f.write(b"GGUF" + b"\x00" * 1024)
    
    # Create a mock llama.cpp directory structure
    llama_cpp_dir = os.path.join(cache_dir, "llama_cpp")
    bin_dir = os.path.join(llama_cpp_dir, "build", "bin")
    lib_dir = os.path.join(llama_cpp_dir, "build", "lib")
    
    os.makedirs(bin_dir, exist_ok=True)
    os.makedirs(lib_dir, exist_ok=True)
    
    # Create a mock server binary
    server_bin = os.path.join(bin_dir, "llama-server")
    with open(server_bin, "wb") as f:
        f.write(b"#!/bin/sh\necho 'Server is listening on http://localhost:8080'\nsleep 10\n")
    
    # Make it executable
    os.chmod(server_bin, 0o755)
    
    # Yield the test paths
    yield {
        "temp_dir": temp_dir,
        "models_dir": models_dir,
        "cache_dir": cache_dir,
        "model_path": model_path
    }
    
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.mark.parametrize("gpu_enabled", [True, False])
def test_api_initialization(llama_env, gpu_enabled):
    """Test API initialization with and without GPU"""
    import llama_cpp_runner.api
    from llama_cpp_runner.main import LlamaCpp
    
    with mock.patch("llama_cpp_runner.api.subprocess.run") as mock_run:
        # Set up mock subprocess for GPU check
        mock_process = mock.Mock()
        mock_process.returncode = 0 if gpu_enabled else 1
        mock_process.stdout = "CUDA Version: 11.2" if gpu_enabled else ""
        mock_run.return_value = mock_process
        
        # Create a LlamaCpp instance
        llama = LlamaCpp(
            models_dir=llama_env["models_dir"],
            cache_dir=llama_env["cache_dir"],
            verbose=True
        )
        
        # Create the API
        if gpu_enabled:
            app = llama_cpp_runner.api.create_gpu_api(llama, gpu_layers=4)
        else:
            app = llama_cpp_runner.api.create_cpu_api(llama)
        
        # Check API routes
        routes = {route.path for route in app.routes}
        
        # Common routes
        assert "/" in routes
        assert "/v1/chat/completions" in routes
        assert "/models" in routes
        assert "/diagnostics" in routes
        
        # GPU-specific routes
        if gpu_enabled:
            assert "/gpu/info" in routes

if __name__ == "__main__":
    unittest.main()