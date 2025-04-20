"""
GPU utilities for llama-cpp-runner

This module provides utilities for GPU detection, configuration,
and optimization for llama-cpp-runner.
"""

import os
import sys
import subprocess
import platform
import logging
import re
from typing import Dict, List, Tuple, Optional, Union, Any

# Import custom logging
from llama_cpp_runner.logger import get_logger

# Set up logger
logger = get_logger("llama_cpp_runner.gpu")

class GpuDetectionError(Exception):
    """Exception raised for GPU detection errors."""
    pass

def detect_gpu() -> Dict[str, Any]:
    """
    Detect GPU capabilities and configuration.
    
    Returns:
        Dictionary containing GPU information
    """
    gpu_info = {
        "available": False,
        "cuda_version": None,
        "driver_version": None,
        "gpus": [],
        "gpu_count": 0
    }
    
    try:
        # Try running nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,compute_cap", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            logger.warning("nvidia-smi failed: %s", result.stderr.strip())
            return gpu_info
        
        # Parse the output
        gpu_info["available"] = True
        
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                try:
                    idx, name, total_mem, free_mem, compute_cap = parts
                    gpu_info["gpus"].append({
                        "index": int(idx),
                        "name": name,
                        "memory_total_mb": _parse_memory(total_mem),
                        "memory_free_mb": _parse_memory(free_mem),
                        "compute_capability": compute_cap
                    })
                except (ValueError, IndexError) as e:
                    logger.warning("Failed to parse GPU information: %s", str(e))
        
        gpu_info["gpu_count"] = len(gpu_info["gpus"])
        
        # Get CUDA and driver version
        version_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if version_result.returncode == 0 and version_result.stdout.strip():
            versions = version_result.stdout.strip().split("\n")[0].split(",")
            if len(versions) >= 2:
                gpu_info["driver_version"] = versions[0].strip()
                gpu_info["cuda_version"] = versions[1].strip()
        
        # Check if Docker GPU runtime is available
        if os.path.exists("/.dockerenv"):
            docker_info = _check_docker_gpu_support()
            gpu_info.update(docker_info)
        
        return gpu_info
    
    except FileNotFoundError:
        logger.warning("nvidia-smi not found, GPU support not available")
        return gpu_info
    except Exception as e:
        logger.exception("Error detecting GPU: %s", str(e))
        return gpu_info

def _parse_memory(mem_str: str) -> int:
    """
    Parse memory string from nvidia-smi (e.g., '16384 MiB').
    
    Args:
        mem_str: Memory string from nvidia-smi
        
    Returns:
        Memory in MB
    """
    match = re.match(r"(\d+)\s*(\w+)", mem_str)
    if not match:
        return 0
    
    value, unit = match.groups()
    value = int(value)
    
    if unit.lower() in ("mib", "mb"):
        return value
    elif unit.lower() in ("gib", "gb"):
        return value * 1024
    else:
        return value

def _check_docker_gpu_support() -> Dict[str, Any]:
    """
    Check if Docker GPU support is available.
    
    Returns:
        Dictionary containing Docker GPU information
    """
    docker_info = {
        "docker": {
            "detected": False,
            "gpu_runtime": False
        }
    }
    
    try:
        # Check if NVIDIA Container Toolkit is available
        result = subprocess.run(
            ["docker", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            return docker_info
        
        docker_info["docker"]["detected"] = True
        
        # Check for NVIDIA runtime
        for line in result.stdout.split("\n"):
            if "Runtimes:" in line and "nvidia" in line:
                docker_info["docker"]["gpu_runtime"] = True
                break
        
        return docker_info
    except FileNotFoundError:
        return docker_info
    except Exception as e:
        logger.warning("Error checking Docker GPU support: %s", str(e))
        return docker_info

def optimize_gpu_params(
    model_size_mb: int,
    available_gpus: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Optimize GPU parameters based on model size and available GPU memory.
    
    Args:
        model_size_mb: Model size in MB
        available_gpus: List of available GPUs with their information
        
    Returns:
        Dictionary containing optimized GPU parameters
    """
    if not available_gpus:
        return {
            "use_gpu": False,
            "reason": "No GPUs available"
        }
    
    # Sort GPUs by available memory (descending)
    gpus = sorted(available_gpus, key=lambda g: g.get("memory_free_mb", 0), reverse=True)
    best_gpu = gpus[0]
    
    # Check if there's enough memory on the best GPU
    if model_size_mb > best_gpu.get("memory_free_mb", 0):
        return {
            "use_gpu": False,
            "reason": f"Model size ({model_size_mb}MB) exceeds available GPU memory ({best_gpu.get('memory_free_mb', 0)}MB)"
        }
    
    # Determine optimal number of GPU layers
    # A rough heuristic: estimate 30-40 layers for 7B models, 60-80 for 13B models
    estimated_layers = -1  # Use all layers by default
    
    # Estimate the number of layers based on model size
    if model_size_mb > 20000:  # Larger than 20GB (likely 70B model)
        estimated_layers = 80
    elif model_size_mb > 10000:  # Larger than 10GB (likely 30-40B model)
        estimated_layers = 60
    elif model_size_mb > 5000:  # Larger than 5GB (likely 13B model)
        estimated_layers = 40
    elif model_size_mb > 2000:  # Larger than 2GB (likely 7B model)
        estimated_layers = 32
    
    # If there's limited memory, reduce the number of layers
    memory_ratio = best_gpu.get("memory_free_mb", 0) / model_size_mb
    if memory_ratio < 1.2 and estimated_layers > 0:
        # If memory is tight, reduce layers by 25%
        estimated_layers = max(1, int(estimated_layers * 0.75))
    
    return {
        "use_gpu": True,
        "gpu_index": best_gpu.get("index", 0),
        "gpu_name": best_gpu.get("name", "Unknown"),
        "gpu_layers": estimated_layers,
        "compute_capability": best_gpu.get("compute_capability", "Unknown"),
        "memory_free_mb": best_gpu.get("memory_free_mb", 0),
        "memory_total_mb": best_gpu.get("memory_total_mb", 0),
        "memory_ratio": memory_ratio
    }

def setup_gpu_environment(gpu_params: Dict[str, Any]) -> None:
    """
    Set up environment variables for GPU usage.
    
    Args:
        gpu_params: GPU parameters from optimize_gpu_params
    """
    if not gpu_params.get("use_gpu", False):
        # Clear GPU environment variables
        for var in ["CUDA_VISIBLE_DEVICES", "GPU_DEVICE"]:
            if var in os.environ:
                del os.environ[var]
        return
    
    # Set GPU environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_params.get("gpu_index", 0))
    
    # Set additional variables for debugging
    os.environ["LLAMA_GPU_LAYERS"] = str(gpu_params.get("gpu_layers", -1))
    
    logger.info(
        "GPU environment configured: CUDA_VISIBLE_DEVICES=%s, GPU_LAYERS=%s",
        os.environ["CUDA_VISIBLE_DEVICES"],
        os.environ["LLAMA_GPU_LAYERS"]
    )

def estimate_gpu_memory_usage(
    model_path: str,
    n_gpu_layers: int = -1
) -> Dict[str, Any]:
    """
    Estimate GPU memory usage for a given model.
    
    Args:
        model_path: Path to the model file
        n_gpu_layers: Number of layers to offload to GPU (-1 for all)
        
    Returns:
        Dictionary containing memory usage estimates
    """
    try:
        # Get model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Estimate the number of layers in the model based on size
        total_layers = 32  # Default for 7B models
        
        if model_size_mb > 20000:  # Larger than 20GB (likely 70B model)
            total_layers = 80
        elif model_size_mb > 10000:  # Larger than 10GB (likely 30-40B model)
            total_layers = 60
        elif model_size_mb > 5000:  # Larger than 5GB (likely 13B model)
            total_layers = 40
        
        # If n_gpu_layers is -1, use all layers
        gpu_layers = total_layers if n_gpu_layers == -1 else min(n_gpu_layers, total_layers)
        
        # Rough estimation: each layer is approximately equal in size
        layer_size_mb = model_size_mb / total_layers
        
        # Estimate GPU memory usage (with some overhead)
        gpu_memory_mb = layer_size_mb * gpu_layers * 1.2  # 20% overhead
        
        return {
            "model_size_mb": model_size_mb,
            "estimated_total_layers": total_layers,
            "gpu_layers": gpu_layers,
            "estimated_gpu_memory_mb": gpu_memory_mb,
            "estimated_layer_size_mb": layer_size_mb
        }
    except Exception as e:
        logger.exception("Error estimating GPU memory usage: %s", str(e))
        
        return {
            "error": str(e),
            "model_size_mb": 0,
            "estimated_gpu_memory_mb": 0
        }

def get_gpu_command_args(
    model_path: str,
    enable_gpu: bool = True,
    gpu_layers: int = -1
) -> List[str]:
    """
    Get command-line arguments for GPU usage with llama-server.
    
    Args:
        model_path: Path to the model file
        enable_gpu: Whether to enable GPU acceleration
        gpu_layers: Number of layers to offload to GPU
        
    Returns:
        List of command-line arguments
    """
    args = []
    
    if not enable_gpu:
        return args
    
    # Detect GPU and optimize parameters
    gpu_info = detect_gpu()
    
    if not gpu_info["available"]:
        logger.warning("GPU requested but not available")
        return args
    
    # Add GPU-specific arguments
    if gpu_layers <= 0:
        # Auto-determine GPU layers based on model size
        memory_estimate = estimate_gpu_memory_usage(model_path)
        gpu_layers = memory_estimate["gpu_layers"]
    
    # Add GPU arguments
    args.extend(["--n-gpu-layers", str(gpu_layers)])
    
    # For older versions of llama.cpp that might use different flags
    args.extend(["--gpu-layers", str(gpu_layers)])
    
    logger.info("GPU command args: %s", args)
    
    return args

def get_gpu_model_info() -> str:
    """
    Get a human-readable string with GPU model information.
    
    Returns:
        String with GPU information
    """
    gpu_info = detect_gpu()
    
    if not gpu_info["available"]:
        return "No GPU detected"
    
    gpu_str = f"Detected {gpu_info['gpu_count']} GPU(s):\n"
    
    for i, gpu in enumerate(gpu_info["gpus"]):
        gpu_str += f"  GPU {i}: {gpu['name']} ({gpu['memory_total_mb']}MB total, {gpu['memory_free_mb']}MB free)\n"
    
    gpu_str += f"CUDA Version: {gpu_info.get('cuda_version', 'Unknown')}\n"
    gpu_str += f"Driver Version: {gpu_info.get('driver_version', 'Unknown')}"
    
    return gpu_str