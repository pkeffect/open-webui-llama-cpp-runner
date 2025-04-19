"""
Validation utilities for llama-cpp-runner

This module provides validation functions to ensure system compatibility,
file integrity, and proper configuration for llama-cpp-runner.
"""

import os
import sys
import platform
import logging
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logger = logging.getLogger("llama_cpp_runner.validation")

def validate_system_compatibility() -> Dict[str, Any]:
    """
    Check if the system meets the requirements for running llama-cpp-runner.
    
    Returns:
        Dictionary containing validation results
    """
    results = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "compatible": True,
        "issues": []
    }
    
    # Check Python version (3.11+ required)
    py_major, py_minor = sys.version_info[:2]
    if py_major < 3 or (py_major == 3 and py_minor < 11):
        results["compatible"] = False
        results["issues"].append(f"Python 3.11+ required, found {py_major}.{py_minor}")
    
    # Check for required executables
    for exe in ["curl", "wget"]:
        if shutil.which(exe) is None:
            results["issues"].append(f"Required executable '{exe}' not found in PATH")
    
    return results

def validate_gguf_model(model_path: str) -> Dict[str, Any]:
    """
    Validate a GGUF model file.
    
    Args:
        model_path: Path to the GGUF model file
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        "path": model_path,
        "exists": os.path.exists(model_path),
        "valid": False,
        "issues": []
    }
    
    if not results["exists"]:
        results["issues"].append(f"Model file does not exist: {model_path}")
        return results
    
    # Check file size
    try:
        file_size = os.path.getsize(model_path)
        results["size"] = file_size
        
        if file_size < 1_000_000:  # Less than 1MB is probably not a valid model
            results["issues"].append(f"Model file is suspiciously small: {file_size} bytes")
    except OSError as e:
        results["issues"].append(f"Failed to get file size: {str(e)}")
    
    # Check file permissions
    try:
        if not os.access(model_path, os.R_OK):
            results["issues"].append("Model file is not readable")
    except OSError as e:
        results["issues"].append(f"Failed to check file permissions: {str(e)}")
    
    # Check GGUF header
    try:
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                results["issues"].append("Invalid GGUF file format (incorrect magic bytes)")
    except Exception as e:
        results["issues"].append(f"Failed to check GGUF header: {str(e)}")
    
    # Set validity flag
    results["valid"] = len(results["issues"]) == 0
    
    return results

def validate_llama_cpp_binaries(llama_cpp_path: str) -> Dict[str, Any]:
    """
    Validate llama.cpp binaries.
    
    Args:
        llama_cpp_path: Path to the llama.cpp directory
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        "path": llama_cpp_path,
        "exists": os.path.exists(llama_cpp_path),
        "valid": False,
        "issues": [],
        "binaries": {}
    }
    
    if not results["exists"]:
        results["issues"].append(f"llama.cpp directory does not exist: {llama_cpp_path}")
        return results
    
    # Check for server binary
    server_bin_path = os.path.join(llama_cpp_path, "build", "bin", "llama-server")
    if os.name == "nt":  # Windows
        server_bin_path += ".exe"
    
    results["binaries"]["server"] = {
        "path": server_bin_path,
        "exists": os.path.exists(server_bin_path),
        "executable": False
    }
    
    if not results["binaries"]["server"]["exists"]:
        results["issues"].append(f"Server binary not found: {server_bin_path}")
    else:
        # Check if executable
        try:
            executable = os.access(server_bin_path, os.X_OK)
            results["binaries"]["server"]["executable"] = executable
            
            if not executable:
                results["issues"].append(f"Server binary is not executable: {server_bin_path}")
        except OSError as e:
            results["issues"].append(f"Failed to check if server binary is executable: {str(e)}")
    
    # Check for shared libraries on Linux/macOS
    if os.name != "nt":  # Not Windows
        lib_paths = [
            os.path.join(llama_cpp_path, "build", "lib", "libllama.so"),
            os.path.join(llama_cpp_path, "build", "libllama.so"),
            "/usr/lib/libllama.so"
        ]
        
        lib_found = False
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                lib_found = True
                results["binaries"]["library"] = {
                    "path": lib_path,
                    "exists": True
                }
                break
        
        if not lib_found:
            results["issues"].append("libllama.so not found in any expected location")
            results["binaries"]["library"] = {
                "exists": False,
                "paths_checked": lib_paths
            }
    
    # Set validity flag
    results["valid"] = len(results["issues"]) == 0
    
    return results

def validate_gpu_support() -> Dict[str, Any]:
    """
    Validate GPU support for llama-cpp-runner.
    
    Returns:
        Dictionary containing validation results
    """
    results = {
        "gpu_available": False,
        "cuda_available": False,
        "issues": []
    }
    
    # Check for NVIDIA GPU using nvidia-smi
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if nvidia_smi.returncode == 0:
            results["gpu_available"] = True
            
            # Extract CUDA version if available
            for line in nvidia_smi.stdout.split('\n'):
                if "CUDA Version:" in line:
                    cuda_version = line.split("CUDA Version:")[-1].strip()
                    results["cuda_version"] = cuda_version
                    results["cuda_available"] = True
        else:
            results["issues"].append("NVIDIA GPU not detected or driver issue")
    except FileNotFoundError:
        results["issues"].append("nvidia-smi not found, NVIDIA driver likely not installed")
    except Exception as e:
        results["issues"].append(f"Error checking GPU support: {str(e)}")
    
    # Check for Docker GPU support if in Docker
    if os.path.exists("/.dockerenv"):
        try:
            # Try to get Docker GPU information
            docker_gpu = subprocess.run(
                ["docker", "info"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )
            
            if docker_gpu.returncode == 0:
                gpu_runtime = False
                for line in docker_gpu.stdout.split('\n'):
                    if "Runtimes:" in line and "nvidia" in line:
                        gpu_runtime = True
                        break
                
                results["docker_gpu_runtime"] = gpu_runtime
                
                if not gpu_runtime and results["gpu_available"]:
                    results["issues"].append("NVIDIA GPU detected but Docker NVIDIA runtime not configured")
        except FileNotFoundError:
            results["issues"].append("Docker command not found")
        except Exception as e:
            results["issues"].append(f"Error checking Docker GPU support: {str(e)}")
    
    return results

def comprehensive_validation(
    models_dir: str,
    cache_dir: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a comprehensive validation of the llama-cpp-runner setup.
    
    Args:
        models_dir: Path to the models directory
        cache_dir: Path to the cache directory
        verbose: Whether to log detailed validation information
        
    Returns:
        Dictionary containing all validation results
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    
    logger.debug("Starting comprehensive validation")
    
    results = {
        "system": validate_system_compatibility(),
        "gpu": validate_gpu_support(),
        "directories": {
            "models": {
                "path": models_dir,
                "exists": os.path.exists(models_dir),
                "issues": []
            },
            "cache": {
                "path": cache_dir,
                "exists": os.path.exists(cache_dir),
                "issues": []
            }
        },
        "llama_cpp": None,
        "models": {}
    }
    
    # Check directories
    for dir_name, dir_info in results["directories"].items():
        if not dir_info["exists"]:
            dir_info["issues"].append(f"Directory does not exist: {dir_info['path']}")
        elif not os.path.isdir(dir_info["path"]):
            dir_info["exists"] = False
            dir_info["issues"].append(f"Path exists but is not a directory: {dir_info['path']}")
        elif not os.access(dir_info["path"], os.R_OK | os.W_OK):
            dir_info["issues"].append(f"Directory is not readable and writable: {dir_info['path']}")
    
    # Check llama.cpp binaries if cache directory exists
    llama_cpp_path = os.path.join(cache_dir, "llama_cpp")
    if os.path.exists(llama_cpp_path):
        results["llama_cpp"] = validate_llama_cpp_binaries(llama_cpp_path)
    
    # Check models if models directory exists
    if os.path.exists(models_dir) and os.path.isdir(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]
        
        if not model_files:
            results["directories"]["models"]["issues"].append("No GGUF model files found")
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            results["models"][model_file] = validate_gguf_model(model_path)
    
    # Check overall validity
    all_valid = (
        results["system"]["compatible"] and
        not results["directories"]["models"]["issues"] and
        not results["directories"]["cache"]["issues"] and
        (results["llama_cpp"] is None or results["llama_cpp"]["valid"]) and
        all(model["valid"] for model in results["models"].values())
    )
    
    results["all_valid"] = all_valid
    
    logger.debug(f"Comprehensive validation completed. All valid: {all_valid}")
    
    return results