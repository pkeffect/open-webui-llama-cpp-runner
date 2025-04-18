import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from llama_cpp_runner.main import LlamaCpp

app = FastAPI(title="LlamaCpp Proxy")

# Initialize the LlamaCpp class
models_dir = os.environ.get("MODELS_DIR", "/models")
cache_dir = os.environ.get("CACHE_DIR", "/cache")
verbose = os.environ.get("VERBOSE", "true").lower() == "true"
timeout = int(os.environ.get("TIMEOUT_MINUTES", "30"))

print(f"Models directory: {models_dir}")
print(f"Cache directory: {cache_dir}")

# Create the LlamaCpp instance
llama_runner = LlamaCpp(
    models_dir=models_dir,
    cache_dir=cache_dir, 
    verbose=verbose, 
    timeout_minutes=timeout
)

@app.get("/")
def read_root():
    """Get server status and list of available models."""
    return {"status": "running", "models": llama_runner.list_models()}

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

if __name__ == "__main__":
    print("Starting LlamaCpp Proxy Server on port 3636")
    models = llama_runner.list_models()
    print(f"Available models: {models}")
    if not models:
        print("WARNING: No models found in the models directory.")
    uvicorn.run(app, host="0.0.0.0", port=3636)