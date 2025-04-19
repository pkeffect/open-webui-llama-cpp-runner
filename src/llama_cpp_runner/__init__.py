"""
llama-cpp-runner package

A Python library for running llama.cpp with zero hassle, providing
automated binary downloads and an easy-to-use API.
"""

# Import main components for easy access
from llama_cpp_runner.cpu import LlamaCpp
from llama_cpp_runner.gpu import GpuLlamaCpp
from llama_cpp_runner.main import run_server, create_runner
from llama_cpp_runner.api import create_api
from llama_cpp_runner.logger import get_logger, setup_logger
from llama_cpp_runner.utils import detect_gpu, get_system_info

__version__ = "0.0.1"

def hello() -> str:
    """
    Simple function to check if the library is working.
    
    Returns:
        Greeting message
    """
    return "Hello from llama-cpp-runner! 🦙"