import os
import uvicorn
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="LlamaCpp Proxy")

# Initialize environment variables
models_dir = os.environ.get("MODELS_DIR", "/models")
cache_dir = os.environ.get("CACHE_DIR", "/cache")
verbose = os.environ.get("VERBOSE", "true").lower() == "true"
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))
# Check for GPU environment variables but set better defaults
enable_gpu = os.environ.get("ENABLE_GPU", "true").lower() == "true"
gpu_layers = os.environ.get("GPU_LAYERS", "32")  # Default to 32 layers instead of -1
# If GPU_LAYERS is -1, calculate a reasonable value based on the model
if gpu_layers == "-1":
    gpu_layers = "32"  # Default for most models
try:
    gpu_layers = int(gpu_layers)
except ValueError:
    gpu_layers = 32  # Fall back to default if not a valid integer

print(f"Models directory: {models_dir}")
print(f"Cache directory: {cache_dir}")
print(f"GPU enabled: {enable_gpu}")
print(f"GPU layers: {gpu_layers}")

# Additional GPU diagnostics if enabled
if enable_gpu:
    # Check for CUDA environment
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    # Try to detect CUDA capabilities in llama-cpp
    server_binary = os.path.join(cache_dir, "llama_cpp/build/bin/llama-server")
    if os.path.exists(server_binary):
        try:
            cuda_check = subprocess.run(
                [server_binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if "n-gpu-layers" in cuda_check.stdout or "n-gpu-layers" in cuda_check.stderr:
                print("✅ CUDA support detected in llama-server binary")
            else:
                print("⚠️ WARNING: llama-server binary does not appear to have CUDA support")
                print("GPU acceleration will not work even if enabled")
        except Exception as e:
            print(f"Error checking CUDA support in binary: {e}")
            
    # Check nvidia-smi
    try:
        nvidia_check = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if nvidia_check.returncode == 0:
            print("✅ NVIDIA GPU detected via nvidia-smi")
            print(nvidia_check.stdout.split("\n")[0])  # Print the first line with CUDA version
        else:
            print("⚠️ WARNING: nvidia-smi command failed, GPU may not be accessible")
    except Exception as e:
        print(f"Error running nvidia-smi: {e}")
        
    # Explicitly set CUDA device if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"Set CUDA_VISIBLE_DEVICES to 0")

# Check for libllama.so
try:
    lib_paths = [
        "/usr/lib/libllama.so",
        "/cache/llama_cpp/build/lib/libllama.so",
        "/cache/llama_cpp/build/libllama.so"
    ]
    found_lib = False
    for path in lib_paths:
        if os.path.exists(path):
            print(f"Found libllama.so at: {path}")
            found_lib = True
            break
    
    if not found_lib:
        print("WARNING: libllama.so not found in expected locations")
        # Run find command to look for it
        try:
            result = subprocess.run(
                ["find", "/cache", "-name", "libllama.so"], 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.stdout:
                print(f"Found libllama.so locations: {result.stdout}")
            else:
                print("No libllama.so found in cache directory")
        except Exception as e:
            print(f"Error searching for libllama.so: {e}")
except Exception as e:
    print(f"Error checking for libllama.so: {e}")

# Create the LlamaCpp instance with conditional import
try:
    # First try to import the GPU version
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
    
    # Test 2: Check CUDA libraries
    try:
        libs = subprocess.run(
            ["ldconfig", "-p"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        cuda_libs = [line for line in libs.stdout.split("\n") if "cuda" in line.lower()]
        test_results["tests"].append({
            "name": "cuda-libraries",
            "success": len(cuda_libs) > 0,
            "found_libs": len(cuda_libs),
            "sample": cuda_libs[:5] if cuda_libs else []
        })
    except Exception as e:
        test_results["tests"].append({
            "name": "cuda-libraries",
            "success": False,
            "error": str(e)
        })
    
    # Test 3: Check if server binary has CUDA capabilities
    server_binary = os.path.join(cache_dir, "llama_cpp/build/bin/llama-server")
    if os.path.exists(server_binary):
        try:
            cuda_check = subprocess.run(
                [server_binary, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            has_gpu_params = "n-gpu-layers" in cuda_check.stdout or "n-gpu-layers" in cuda_check.stderr
            test_results["tests"].append({
                "name": "llama-server-cuda-support",
                "success": has_gpu_params,
                "details": "GPU parameters detected in help output" if has_gpu_params else "No GPU parameters found in help output"
            })
        except Exception as e:
            test_results["tests"].append({
                "name": "llama-server-cuda-support",
                "success": False,
                "error": str(e)
            })
    
    # Test 4: Run a mini inference to check GPU memory usage before and after
    try:
        # Get GPU memory usage before
        before_mem = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        before_usage = int(before_mem.stdout.strip()) if before_mem.returncode == 0 else None
        
        # Run a minimal inference with explicit GPU layers
        if os.path.exists(server_binary):
            model_path = None
            # Find a model to test with
            if models := llama_runner.list_models():
                model_path = os.path.join(models_dir, models[0])
            
            if model_path:
                # Just start and immediately stop to see if CUDA initializes
                test_proc = subprocess.Popen(
                    [server_binary, "-m", model_path, "--n-gpu-layers", str(gpu_layers)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Give it a moment to initialize CUDA
                time.sleep(5)
                test_proc.terminate()
                test_proc.wait()
                
                # Get GPU memory usage after
                after_mem = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                after_usage = int(after_mem.stdout.strip()) if after_mem.returncode == 0 else None
                
                if before_usage is not None and after_usage is not None:
                    memory_diff = after_usage - before_usage
                    test_results["tests"].append({
                        "name": "gpu-memory-usage-test",
                        "success": memory_diff > 10,  # If GPU memory usage increased significantly
                        "before_usage_mb": before_usage,
                        "after_usage_mb": after_usage,
                        "diff_mb": memory_diff,
                        "using_gpu": memory_diff > 10
                    })
                else:
                    test_results["tests"].append({
                        "name": "gpu-memory-usage-test",
                        "success": False,
                        "error": "Could not measure GPU memory usage"
                    })
        
    except Exception as e:
        test_results["tests"].append({
            "name": "gpu-memory-usage-test",
            "success": False,
            "error": str(e)
        })
    
    # Overall assessment
    successful_tests = sum(1 for test in test_results["tests"] if test.get("success", False))
    test_results["status"] = "OK" if successful_tests >= 3 else "ISSUES DETECTED"
    test_results["success_rate"] = f"{successful_tests}/{len(test_results['tests'])}"
    
    return test_results

@app.get("/system/libraries")
def library_info():
    """Get information about system libraries and paths."""
    lib_info = {
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", "Not set"),
        "library_check": {}
    }
    
    # Check for important libraries
    lib_paths = [
        "/usr/lib/libllama.so",
        "/cache/llama_cpp/build/lib/libllama.so",
        "/cache/llama_cpp/build/libllama.so"
    ]
    
    for path in lib_paths:
        lib_info["library_check"][path] = os.path.exists(path)
    
    # Add search results
    try:
        result = subprocess.run(
            ["find", "/cache", "-name", "libllama.so"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        lib_info["search_results"] = result.stdout.strip().split("\n") if result.stdout else []
    except Exception as e:
        lib_info["search_error"] = str(e)
    
    return lib_info

if __name__ == "__main__":
    print("Starting LlamaCpp Proxy Server on port 3636")
    models = llama_runner.list_models()
    print(f"Available models: {models}")
    if not models:
        print("WARNING: No models found in the models directory.")
    uvicorn.run(app, host="0.0.0.0", port=3636)