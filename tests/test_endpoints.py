#!/usr/bin/env python3
"""
Script to test all API endpoints of the llama-cpp-runner
"""

import requests
import sys
import time
import json

# Default settings
CPU_PORT = 3636
GPU_PORT = 10000

def test_endpoint(base_url, endpoint, method="GET", data=None):
    """Test a specific endpoint and print the result"""
    url = f"{base_url}{endpoint}"
    
    print(f"\n🔍 Testing: {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=5)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"📡 Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Success!")
            try:
                print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"📄 Response: {response.text[:200]}...")
            return True
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            try:
                print(f"📄 Response: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"📄 Response: {response.text[:200]}...")
            return False
    except requests.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_all_endpoints(port):
    """Test all endpoints on the given port"""
    base_url = f"http://localhost:{port}"
    
    print(f"\n{'='*60}")
    print(f"🧪 Testing all endpoints on {base_url}")
    print(f"{'='*60}")
    
    # Test root endpoint
    success = test_endpoint(base_url, "/")
    
    # Test health endpoint
    test_endpoint(base_url, "/health")
    
    # Test models endpoint
    test_endpoint(base_url, "/models")
    
    # If GPU mode (port 10000)
    if port == GPU_PORT:
        # Test GPU info endpoint
        test_endpoint(base_url, "/gpu/info")
    
    # Only test chat completion if we have models
    if success:
        try:
            response = requests.get(f"{base_url}/models")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    # Test chat completion with the first model
                    test_data = {
                        "model": models[0],
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": False,
                        "max_tokens": 10
                    }
                    test_endpoint(base_url, "/v1/chat/completions", method="POST", data=test_data)
        except Exception as e:
            print(f"❌ Error testing chat completion: {e}")

def main():
    """Main function to test endpoints"""
    if len(sys.argv) > 1 and sys.argv[1] == "gpu":
        # Test GPU endpoints
        test_all_endpoints(GPU_PORT)
    else:
        # Test CPU endpoints
        test_all_endpoints(CPU_PORT)

if __name__ == "__main__":
    main()