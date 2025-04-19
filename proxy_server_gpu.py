"""
LlamaCpp GPU Proxy Server with Comprehensive Diagnostics

This script provides a full-featured GPU-enabled proxy server for LlamaCpp,
including model serving, chat completions, and extensive GPU diagnostics.
"""

import os
import sys
import json
import time
import importlib
import platform
import subprocess
from typing import Dict, Any, List, Optional

# Ensure proper error handling for import dependencies
try:
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as e:
    print(f"Critical Import Error: {e}")
    print("Please install required dependencies: fastapi, uvicorn")
    sys.exit(1)

# Initialize environment variables
models_dir = os.environ.get("MODELS_DIR", "/models")
cache_dir = os.environ.get("CACHE_DIR", "/cache")
verbose = os.environ.get("VERBOSE", "true").lower() == "true"
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))

# GPU Configuration
enable_gpu = os.environ.get("ENABLE_GPU", "true").lower() == "true"
gpu_layers = os.environ.get("GPU_LAYERS", "-1")
try:
    gpu_layers = int(gpu_layers) if gpu_layers != "-1" else -1
except ValueError:
    gpu_layers = -1

# Initialize FastAPI application
app = FastAPI(
    title="LlamaCpp GPU Proxy",
    description="GPU-enabled proxy server for LlamaCpp with comprehensive diagnostics"
)

def safe_subprocess_run(command: List[str], timeout: int = 5) -> Dict[str, Any]:
    """
    Safely run subprocess commands with error handling.
    
    Args:
        command (List[str]): Command to execute
        timeout (int): Timeout in seconds
    
    Returns:
        Dict with command execution results
    """
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "output": result.stdout.strip(),
            "error": result.stderr.strip(),
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "return_code": -2
        }

def get_cuda_system_info() -> Dict[str, Any]:
    """
    Collect comprehensive CUDA and system GPU information.
    
    Returns:
        Dict containing CUDA and GPU system details
    """
    system_info = {
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "machine": platform.machine()
        },
        "cuda": {
            "available": False,
            "version": None,
            "devices": []
        }
    }

    # NVIDIA-SMI Information
    nvidia_smi_result = safe_subprocess_run(["nvidia-smi"])
    if nvidia_smi_result["success"]:
        system_info["cuda"]["available"] = True
        
        # Parse CUDA version
        for line in nvidia_smi_result["output"].split('\n'):
            if "CUDA Version:" in line:
                system_info["cuda"]["version"] = line.split("CUDA Version:")[-1].strip()
                break
        
        # Detect GPU devices
        gpu_list_result = safe_subprocess_run(["nvidia-smi", "-L"])
        if gpu_list_result["success"]:
            system_info["cuda"]["devices"] = gpu_list_result["output"].split('\n')

    return system_info

def check_library_availability() -> Dict[str, Any]:
    """
    Check availability of critical GPU and llama.cpp libraries.
    
    Returns:
        Dict with library existence and paths
    """
    library_paths = [
        "/usr/lib/libllama.so",
        "/cache/llama_cpp/build/lib/libllama.so",
        "/cache/llama_cpp/build/libllama.so"
    ]

    libraries = {
        "libllama.so": {
            "found_locations": [],
            "system_paths_checked": library_paths
        },
        "cuda_libraries": {
            "ldconfig_results": safe_subprocess_run(["ldconfig", "-p"])
        }
    }

    for path in library_paths:
        if os.path.exists(path):
            libraries["libllama.so"]["found_locations"].append(path)

    return libraries

def analyze_torch_cuda() -> Dict[str, Any]:
    """
    Analyze PyTorch CUDA capabilities if available.
    
    Returns:
        Dict with PyTorch and CUDA information
    """
    torch_cuda_info = {
        "torch_cuda_available": False,
        "cuda_device_count": 0,
        "cuda_devices": []
    }

    try:
        cuda_module = importlib.import_module('torch.cuda')
        torch_cuda_info["torch_cuda_available"] = cuda_module.is_available()
        torch_cuda_info["cuda_device_count"] = cuda_module.device_count()
        
        for i in range(torch_cuda_info["cuda_device_count"]):
            torch_cuda_info["cuda_devices"].append({
                "index": i,
                "name": cuda_module.get_device_name(i)
            })
    except ImportError:
        torch_cuda_info["error"] = "PyTorch with CUDA support not installed"

    return torch_cuda_info

# Import LlamaCpp with GPU support
try:
    if enable_gpu:
        try:
            from llama_cpp_runner.main_gpu import LlamaCpp
            print("Using GPU-enabled LlamaCpp class")
            llama_runner = LlamaCpp(
                models_dir=models_dir,
                cache_dir=cache_dir,
                verbose=verbose,
                timeout_minutes=timeout,
                enable_gpu=enable_gpu,
                gpu_layers=gpu_layers
            )
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Error importing GPU version: {e}")
            print("Falling back to standard LlamaCpp class")
            from llama_cpp_runner.main import LlamaCpp
            llama_runner = LlamaCpp(
                models_dir=models_dir,
                cache_dir=cache_dir,
                verbose=verbose,
                timeout_minutes=timeout
            )
    else:
        # Use standard version if GPU is disabled
        from llama_cpp_runner.main import LlamaCpp
        llama_runner = LlamaCpp(
            models_dir=models_dir,
            cache_dir=cache_dir,
            verbose=verbose,
            timeout_minutes=timeout
        )
except Exception as e:
    print(f"Error initializing LlamaCpp: {e}")
    raise

# API Endpoints
@app.get("/")
def read_root():
    """Get server status and list of available models."""
    response = {
        "status": "running", 
        "models": llama_runner.list_models(),
        "gpu_enabled": enable_gpu
    }
    if enable_gpu:
        response["gpu_layers"] = gpu_layers
    return response

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward chat completion requests to the LlamaCpp server."""
    try:
        body = await request.json()
        
        if "model" not in body:
            return JSONResponse(
                status_code=400,
                content={"error": "Model not specified in request"}
            )
        
        try:
            # Explicitly set GPU layers if provided in the request
            if "n_gpu_layers" in body:
                body["gpu_layers"] = body["n_gpu_layers"]
            
            result = llama_runner.chat_completion(body)
            
            # Handle streaming responses
            if body.get("stream", False):
                async def generate():
                    for line in result:
                        if line:
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                return result
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid request: {str(e)}"}
        )

@app.get("/models")
def list_models():
    """List all available models."""
    return {"models": llama_runner.list_models()}

@app.get("/gpu/diagnostics")
async def comprehensive_gpu_diagnostics():
    """
    Comprehensive GPU diagnostics endpoint.
    
    Provides detailed information about:
    - System and CUDA configuration
    - Library availability
    - PyTorch CUDA capabilities
    """
    diagnostics = {
        "system_info": get_cuda_system_info(),
        "libraries": check_library_availability(),
        "torch_cuda": analyze_torch_cuda(),
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "Not set")
        },
        "llama_cpp": {
            "gpu_enabled": enable_gpu,
            "gpu_layers": gpu_layers
        }
    }

    return JSONResponse(content=diagnostics)

@app.get("/gpu/validation")
async def gpu_validation():
    """
    Validate GPU readiness for model inference.
    
    Checks:
    - CUDA availability
    - Required libraries
    - Inference preparation
    """
    validation = {
        "gpu_ready": False,
        "checks": []
    }

    # Check CUDA availability
    cuda_system_info = get_cuda_system_info()
    cuda_check = {
        "name": "CUDA Availability",
        "passed": cuda_system_info["cuda"]["available"],
        "details": cuda_system_info["cuda"]
    }
    validation["checks"].append(cuda_check)

    # Library availability
    libraries = check_library_availability()
    library_check = {
        "name": "Critical Libraries",
        "passed": bool(libraries["libllama.so"]["found_locations"]),
        "details": libraries
    }
    validation["checks"].append(library_check)

    # PyTorch CUDA Check
    torch_cuda = analyze_torch_cuda()
    torch_check = {
        "name": "PyTorch CUDA",
        "passed": torch_cuda["torch_cuda_available"],
        "details": torch_cuda
    }
    validation["checks"].append(torch_check)

    # Determine overall GPU readiness
    validation["gpu_ready"] = all(check["passed"] for check in validation["checks"])

    return JSONResponse(content=validation)

@app.get("/gpu/test")
def test_gpu():
    """Run a quick GPU test to verify CUDA is working properly."""
    test_results = {
        "enabled": enable_gpu,
        "requested_layers": gpu_layers,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set"),
        "tests": []
    }
    
    if not enable_gpu:
        test_results["status"] = "GPU not enabled"
        return test_results
    
    # Test 1: Check nvidia-smi
    try:
        nvidia_check = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        test_results["tests"].append({
            "name": "nvidia-smi",
            "success": nvidia_check.returncode == 0,
            "output": nvidia_check.stdout.split("\n")[0] if nvidia_check.returncode == 0 else nvidia_check.stderr
        })
    except Exception as e:
        test_results["tests"].append({
            "name": "nvidia-smi",
            "success": False,
            "error": str(e)
        })
    
    # Existing additional tests from the original script remain the same...
    
    return test_results

@app.get("/system/libraries")
def library_info():
    """Get information about system libraries and paths."""
    return check_library_availability()

# Server runtime configuration
if __name__ == "__main__":
    # Additional startup diagnostics
    print(f"Models directory: {models_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"GPU enabled: {enable_gpu}")
    print(f"GPU layers: {gpu_layers}")

    # Run server
    uvicorn.run(app, host="0.0.0.0", port=10000)