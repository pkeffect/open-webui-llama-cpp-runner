"""
Utility functions for llama-cpp-runner

This module provides helper functions for GPU detection,
optimization, and configuration.
"""

import os
import subprocess
import platform
import re
from typing import Dict, List, Any, Optional, Union

from llama_cpp_runner.logger import get_logger

# Set up logger
logger = get_logger("llama_cpp_runner.utils")

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
            logger.warning(f"nvidia-smi failed: {result.stderr.strip()}")
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
                    logger.warning(f"Failed to parse GPU information: {e}")
        
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
        logger.exception(f"Error detecting GPU: {e}")
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
        logger.warning(f"Error checking Docker GPU support: {e}")
        return docker_info

def optimize_gpu_params(model_size_mb: int, available_gpus: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        f"GPU environment configured: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, "
        f"GPU_LAYERS={os.environ['LLAMA_GPU_LAYERS']}"
    )

def estimate_model_size(model_path: str) -> Dict[str, Any]:
    """
    Estimate model size and characteristics based on file size.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary containing model size estimation
    """
    try:
        # Get model size
        model_size_bytes = os.path.getsize(model_path)
        model_size_mb = model_size_bytes / (1024 * 1024)
        model_size_gb = model_size_mb / 1024
        
        # Estimate model parameters based on size
        params_billion = 0
        model_type = "Unknown"
        
        if model_size_mb < 1000:
            params_billion = 0.1
            model_type = "Tiny"
        elif model_size_mb < 2000:
            params_billion = 1
            model_type = "1B class"
        elif model_size_mb < 4000:
            params_billion = 3
            model_type = "3B class"
        elif model_size_mb < 6000:
            params_billion = 7
            model_type = "7B class"
        elif model_size_mb < 12000:
            params_billion = 13
            model_type = "13B class"
        elif model_size_mb < 25000:
            params_billion = 30
            model_type = "30B class"
        elif model_size_mb < 40000:
            params_billion = 65
            model_type = "65B class"
        else:
            params_billion = 70
            model_type = "70B+ class"
        
        return {
            "path": model_path,
            "size_bytes": model_size_bytes,
            "size_mb": model_size_mb,
            "size_gb": model_size_gb,
            "estimated_params_b": params_billion,
            "model_type": model_type
        }
    except Exception as e:
        logger.error(f"Error estimating model size: {e}")
        return {
            "path": model_path,
            "error": str(e)
        }

def get_system_info() -> Dict[str, Any]:
    """
    Get detailed system information.
    
    Returns:
        Dictionary containing system information
    """
    info = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation()
        }
    }
    
    # Get CPU information
    try:
        cpu_count = os.cpu_count()
        info["cpu"] = {
            "count": cpu_count
        }
        
        # Get more detailed CPU info on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    cpuinfo = f.read()
                
                # Extract model name
                model_match = re.search(r"model name\s+:\s+(.*)", cpuinfo)
                if model_match:
                    info["cpu"]["model"] = model_match.group(1)
                
                # Extract CPU frequency
                freq_match = re.search(r"cpu MHz\s+:\s+(.*)", cpuinfo)
                if freq_match:
                    info["cpu"]["mhz"] = float(freq_match.group(1))
            except Exception as e:
                logger.debug(f"Error getting detailed CPU info: {e}")
    except Exception as e:
        logger.debug(f"Error getting CPU info: {e}")
    
    # Get memory information
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            # Extract total memory
            mem_match = re.search(r"MemTotal:\s+(\d+)", meminfo)
            if mem_match:
                total_kb = int(mem_match.group(1))
                info["memory"] = {
                    "total_kb": total_kb,
                    "total_mb": total_kb / 1024,
                    "total_gb": total_kb / (1024 * 1024)
                }
    except Exception as e:
        logger.debug(f"Error getting memory info: {e}")
    
    # Get GPU information if available
    try:
        gpu_info = detect_gpu()
        if gpu_info["available"]:
            info["gpu"] = gpu_info
    except Exception as e:
        logger.debug(f"Error getting GPU info: {e}")
    
    return info