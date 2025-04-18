FROM python:3.11-slim

WORKDIR /app

# Install only essential packages and clean up in one layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/

# Install the package in development mode and required dependencies
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir requests fastapi uvicorn

# Create volume mount points
VOLUME /models
VOLUME /cache

# Create proxy server script directly in the Dockerfile
RUN echo 'import os\n\
import uvicorn\n\
from fastapi import FastAPI, Request\n\
from fastapi.responses import StreamingResponse, JSONResponse\n\
from llama_cpp_runner.main import LlamaCpp\n\
\n\
app = FastAPI(title="LlamaCpp Proxy")\n\
\n\
# Initialize the LlamaCpp class\n\
models_dir = os.environ.get("MODELS_DIR", "/models")\n\
cache_dir = os.environ.get("CACHE_DIR", "/cache")\n\
verbose = os.environ.get("VERBOSE", "true").lower() == "true"\n\
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))\n\
\n\
print(f"Models directory: {models_dir}")\n\
print(f"Cache directory: {cache_dir}")\n\
\n\
# Create the LlamaCpp instance\n\
llama_runner = LlamaCpp(\n\
    models_dir=models_dir,\n\
    cache_dir=cache_dir, \n\
    verbose=verbose, \n\
    timeout_minutes=timeout\n\
)\n\
\n\
@app.get("/")\n\
def read_root():\n\
    """Get server status and list of available models."""\n\
    return {"status": "running", "models": llama_runner.list_models()}\n\
\n\
@app.post("/v1/chat/completions")\n\
async def chat_completions(request: Request):\n\
    """Forward chat completion requests to the LlamaCpp server."""\n\
    try:\n\
        body = await request.json()\n\
        \n\
        if "model" not in body:\n\
            return JSONResponse(\n\
                status_code=400,\n\
                content={"error": "Model not specified in request"}\n\
            )\n\
        \n\
        try:\n\
            result = llama_runner.chat_completion(body)\n\
            \n\
            # Handle streaming responses\n\
            if body.get("stream", False):\n\
                async def generate():\n\
                    for line in result:\n\
                        if line:\n\
                            yield f"data: {line}\\n\\n"\n\
                    yield "data: [DONE]\\n\\n"\n\
                \n\
                return StreamingResponse(generate(), media_type="text/event-stream")\n\
            else:\n\
                return result\n\
        except Exception as e:\n\
            return JSONResponse(\n\
                status_code=500,\n\
                content={"error": str(e)}\n\
            )\n\
    except Exception as e:\n\
        return JSONResponse(\n\
            status_code=400,\n\
            content={"error": f"Invalid request: {str(e)}"}\n\
        )\n\
\n\
@app.get("/models")\n\
def list_models():\n\
    """List all available models."""\n\
    return {"models": llama_runner.list_models()}\n\
\n\
if __name__ == "__main__":\n\
    print("Starting LlamaCpp Proxy Server on port 3636")\n\
    models = llama_runner.list_models()\n\
    print(f"Available models: {models}")\n\
    if not models:\n\
        print("WARNING: No models found in the models directory.")\n\
    uvicorn.run(app, host="0.0.0.0", port=3636)' > /app/proxy_server.py

# Expose the proxy server port
EXPOSE 3636

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/models
ENV CACHE_DIR=/cache
ENV VERBOSE=true
ENV TIMEOUT_MINUTES=30

# Command to run when the container starts
CMD ["python", "/app/proxy_server.py"]