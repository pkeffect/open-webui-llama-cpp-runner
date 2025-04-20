"""
Simplified API module for llama-cpp-runner

This module provides a centralized definition of all API endpoints
for both CPU and GPU modes with pretty formatting.
"""

import os
import platform
import subprocess
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

def create_api(llama_runner, enable_gpu: bool = False, gpu_layers: int = -1) -> FastAPI:
    """
    Create a FastAPI application with all required endpoints.
    
    Args:
        llama_runner: Instance of LlamaCpp
        enable_gpu: Whether GPU is enabled
        gpu_layers: Number of GPU layers
        
    Returns:
        FastAPI application with all endpoints registered
    """
    # Create a new FastAPI application
    app = FastAPI(title="LlamaCpp API", version="0.0.1")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ROOT endpoint (status)
    @app.get("/")
    async def read_root():
        """Get server status and list of available models."""
        models = llama_runner.list_models()
        
        response = {
            "status": "running",
            "models": models,
            "config": {
                "gpu_enabled": enable_gpu,
                "models_dir": llama_runner.models_dir,
                "system": f"{platform.system()} {platform.release()}"
            }
        }
        
        if enable_gpu:
            response["config"]["gpu_layers"] = gpu_layers
        
        return response
    
    # HEALTH CHECK endpoint
    @app.get("/health")
    async def health_check():
        """Simple health check endpoint."""
        return {"status": "ok"}
    
    # MODELS endpoint
    @app.get("/models")
    async def list_models():
        """List all available models."""
        models = llama_runner.list_models()
        return {"models": models}
    
    # CHAT COMPLETIONS endpoint
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
                # Add GPU layers parameter if GPU is enabled
                if enable_gpu and "n_gpu_layers" not in body and "gpu_layers" not in body:
                    body["n_gpu_layers"] = gpu_layers
                
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
    
    # GPU INFO endpoint (only added when GPU is enabled)
    if enable_gpu:
        @app.get("/gpu/info")
        async def gpu_info():
            """Get GPU information."""
            gpu_status = {"enabled": enable_gpu, "layers": gpu_layers}
            
            try:
                # Run nvidia-smi to get GPU information
                nvidia_smi = subprocess.run(
                    ["nvidia-smi"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                
                if nvidia_smi.returncode == 0:
                    gpu_status["nvidia_smi"] = nvidia_smi.stdout
                else:
                    gpu_status["error"] = nvidia_smi.stderr
            except Exception as e:
                gpu_status["error"] = str(e)
                
            return gpu_status
    
    return app