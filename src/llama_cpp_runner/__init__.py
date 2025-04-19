from llama_cpp_runner.main import LlamaCpp, LlamaCppServer
from llama_cpp_runner.logger import get_logger, setup_logger
from llama_cpp_runner.api import create_api


def hello() -> str:
    return "Hello from llama-cpp-runner! 🦙"


__version__ = "0.0.1"