"""
Unified API for llama-cpp-runner

This module provides a centralized API for both CPU and GPU modes
with consistent endpoints and error handling.
"""

import os
import platform
import time
import json
import subprocess
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from llama_cpp_runner.logger import get_logger
from llama_cpp_runner.utils import get_system_info, detect_gpu, estimate_model_size

# Set up logger
logger = get_logger("llama_cpp_runner.api")

def create_api(
    llama_runner,
    title: str = "LlamaCpp API",
    version: str = "0.0.1",
    enable_gpu: bool = False,
    gpu_layers: int = -1
) -> FastAPI:
    """
    Create a FastAPI application for llama-cpp-runner.
    
    Args:
        llama_runner: Instance of LlamaCpp or GpuLlamaCpp
        title: API title
        version: API version
        enable_gpu: Whether GPU is enabled
        gpu_layers: Number of GPU layers (for GPU mode)
        
    Returns:
        FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title=title,
        version=version,
        description="API for running llama.cpp models",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store configuration
    config = {
        "enable_gpu": enable_gpu,
        "gpu_layers": gpu_layers,
        "models_dir": llama_runner.models_dir,
        "cache_dir": llama_runner.cache_dir,
        "verbose": llama_runner.verbose,
        "timeout_minutes": llama_runner.timeout_minutes,
        "system_info": get_system_info()
    }
    
    # Helper for error handling
    def handle_exceptions(func):
        """Decorator for consistent error handling"""
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                status_code = 500
                
                if isinstance(e, HTTPException):
                    status_code = e.status_code
                elif isinstance(e, FileNotFoundError):
                    status_code = 404
                elif isinstance(e, ValueError):
                    status_code = 400
                
                return JSONResponse(
                    status_code=status_code,
                    content={
                        "error": str(e),
                        "type": e.__class__.__name__
                    }
                )
        
        return wrapper
    
    # Root endpoint (status)
    @app.get("/")
    @handle_exceptions
    async def read_root():
        """Get server status and available models"""
        models = llama_runner.list_models()
        
        start_time = getattr(app, "start_time", time.time())
        uptime_seconds = time.time() - start_time
        
        response = {
            "status": "running",
            "models": models,
            "count": len(models),
            "version": version,
            "uptime_seconds": uptime_seconds,
            "config": {
                "gpu_enabled": enable_gpu,
                "models_dir": llama_runner.models_dir,
                "system": platform.system(),
                "platform": platform.platform()
            }
        }
        
        if enable_gpu:
            response["config"]["gpu_layers"] = gpu_layers
        
        return response
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "ok", "timestamp": time.time()}
    
    # Models endpoint
    @app.get("/models")
    @handle_exceptions
    async def list_models():
        """List all available models"""
        models = llama_runner.list_models()
        
        # Add model details
        models_info = []
        for model_name in models:
            model_path = os.path.join(llama_runner.models_dir, model_name)
            model_info = estimate_model_size(model_path)
            model_info["name"] = model_name
            models_info.append(model_info)
        
        return {
            "models": models,
            "count": len(models),
            "details": models_info
        }
    
    # Chat completions endpoint (OpenAI-compatible)
    @app.post("/v1/chat/completions")
    @handle_exceptions
    async def chat_completions(request: Request):
        """
        Process chat completion requests (OpenAI-compatible).
        
        Expected format:
        {
            "model": "model_name.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            ...other parameters
        }
        """
        try:
            body = await request.json()
            
            # Validate required fields
            if "model" not in body:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Model not specified in request"}
                )
            
            if "messages" not in body:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Messages not specified in request"}
                )
            
            # Add GPU layers parameter if GPU is enabled
            if enable_gpu and "n_gpu_layers" not in body and "gpu_layers" not in body:
                body["n_gpu_layers"] = gpu_layers
            
            # Process request
            try:
                # Log request (excluding sensitive content)
                safe_log = {k: v for k, v in body.items() if k != "messages"}
                logger.info(f"Processing chat completion request: {json.dumps(safe_log)}")
                
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
                logger.error(f"Error in chat completion: {e}")
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e)}
                )
        except Exception as e:
            logger.error(f"Invalid request: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid request: {str(e)}"}
            )
    
    # Compatibility endpoint for non-OpenAI format
    @app.post("/completion")
    @handle_exceptions
    async def completion(request: Request):
        """Legacy completion endpoint for backward compatibility"""
        try:
            body = await request.json()
            return await chat_completions(request)
        except Exception as e:
            logger.error(f"Error in completion endpoint: {e}")
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid request: {str(e)}"}
            )
    
    # System information endpoint
    @app.get("/system/info")
    @handle_exceptions
    async def system_info():
        """Get detailed system information"""
        return get_system_info()
    
    # GPU information endpoint (only if GPU is enabled)
    if enable_gpu:
        @app.get("/gpu/info")
        @handle_exceptions
        async def gpu_info():
            """Get GPU information"""
            gpu_status = {"enabled": enable_gpu, "layers": gpu_layers}
            
            try:
                # Get detailed GPU information
                gpu_status.update(detect_gpu())
                
                # Add GPU utilization if available
                try:
                    nvidia_smi = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu", "--format=csv,noheader"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5
                    )
                    
                    if nvidia_smi.returncode == 0:
                        gpu_metrics = []
                        for i, line in enumerate(nvidia_smi.stdout.strip().split("\n")):
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 3:
                                gpu_metrics.append({
                                    "index": i,
                                    "utilization_gpu": parts[0],
                                    "utilization_memory": parts[1],
                                    "temperature": parts[2]
                                })
                        
                        gpu_status["metrics"] = gpu_metrics
                except Exception as e:
                    gpu_status["metrics_error"] = str(e)
            except Exception as e:
                gpu_status["error"] = str(e)
                
            return gpu_status
    
    # Model information endpoint
    @app.get("/models/{model_name}")
    @handle_exceptions
    async def model_info(model_name: str):
        """Get detailed information about a specific model"""
        model_path = os.path.join(llama_runner.models_dir, model_name)
        
        if not os.path.exists(model_path):
            return JSONResponse(
                status_code=404,
                content={"error": f"Model not found: {model_name}"}
            )
        
        # Get model details
        model_info = estimate_model_size(model_path)
        model_info["name"] = model_name
        
        # Add file information
        try:
            stat_info = os.stat(model_path)
            model_info["created"] = stat_info.st_ctime
            model_info["modified"] = stat_info.st_mtime
            model_info["accessed"] = stat_info.st_atime
        except Exception as e:
            model_info["stat_error"] = str(e)
        
        return model_info
    
    # Configure startup event
    @app.on_event("startup")
    async def startup_event():
        """Initialize startup time for uptime tracking"""
        app.start_time = time.time()
        logger.info("API server started")
    
    # Return the configured app
    return app