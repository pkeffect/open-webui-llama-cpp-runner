#!/bin/bash
# Script to set up library paths for llama-cpp-runner with GPU support

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up library paths for llama-cpp-runner with GPU support${NC}"

# Check if we're in a Docker container
if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}Running inside Docker container${NC}"
    CACHE_DIR=${CACHE_DIR:-/cache}
else
    echo -e "${YELLOW}Running on host system${NC}"
    CACHE_DIR=${CACHE_DIR:-./cache}
fi

# Create cache directory if it doesn't exist
mkdir -p "${CACHE_DIR}"

LLAMA_CPP_DIR="${CACHE_DIR}/llama_cpp"
LIB_PATH="${LLAMA_CPP_DIR}/build/lib/libllama.so"
LIB_DIR="${LLAMA_CPP_DIR}/build/lib"
BIN_DIR="${LLAMA_CPP_DIR}/build/bin"

echo "Looking for llama.cpp libraries in: ${LLAMA_CPP_DIR}"

# Create lib directory if it doesn't exist
mkdir -p "${LIB_DIR}"
mkdir -p "${BIN_DIR}"

# Look for libllama.so
if [ -f "${LIB_PATH}" ]; then
    echo -e "${GREEN}Found library at: ${LIB_PATH}${NC}"
else
    echo -e "${YELLOW}Library not found at expected location. Searching for it...${NC}"
    
    # Search for libllama.so in various locations
    FOUND=false
    
    # Possible locations
    POSSIBLE_LOCATIONS=(
        "${LLAMA_CPP_DIR}/build/libllama.so"
        "${LLAMA_CPP_DIR}/libllama.so"
        "${BIN_DIR}/libllama.so"
        "/usr/lib/libllama.so"
        "/usr/local/lib/libllama.so"
    )
    
    for loc in "${POSSIBLE_LOCATIONS[@]}"; do
        if [ -f "$loc" ]; then
            echo -e "${GREEN}Found library at: $loc${NC}"
            echo -e "${YELLOW}Creating symbolic link to ${LIB_PATH}${NC}"
            ln -sf "$loc" "${LIB_PATH}"
            FOUND=true
            break
        fi
    done
    
    if [ "$FOUND" = false ]; then
        echo -e "${RED}Could not find libllama.so in any expected location${NC}"
        
        # Check if we can find it anywhere in the build directory
        echo -e "${YELLOW}Searching entire cache directory...${NC}"
        SEARCH_RESULT=$(find "${CACHE_DIR}" -name "libllama.so" 2>/dev/null || echo "")
        
        if [ -n "$SEARCH_RESULT" ]; then
            echo -e "${GREEN}Found library at: $SEARCH_RESULT${NC}"
            echo -e "${YELLOW}Creating symbolic link to ${LIB_PATH}${NC}"
            mkdir -p $(dirname "${LIB_PATH}")
            ln -sf "$SEARCH_RESULT" "${LIB_PATH}"
            FOUND=true
        else
            echo -e "${YELLOW}No existing library found. The library will be downloaded or compiled during first run.${NC}"
        fi
    fi
fi

# Update LD_LIBRARY_PATH
if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="${LIB_DIR}:${BIN_DIR}"
else
    # Check if paths are already in LD_LIBRARY_PATH
    if [[ ":$LD_LIBRARY_PATH:" != *":${LIB_DIR}:"* ]]; then
        export LD_LIBRARY_PATH="${LIB_DIR}:$LD_LIBRARY_PATH"
    fi
    if [[ ":$LD_LIBRARY_PATH:" != *":${BIN_DIR}:"* ]]; then
        export LD_LIBRARY_PATH="${BIN_DIR}:$LD_LIBRARY_PATH"
    fi
fi

echo -e "${GREEN}LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH${NC}"

# Add CUDA libraries to LD_LIBRARY_PATH if they exist
CUDA_LIBS="/usr/local/cuda/lib64"
if [ -d "$CUDA_LIBS" ]; then
    if [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_LIBS}:"* ]]; then
        export LD_LIBRARY_PATH="${CUDA_LIBS}:$LD_LIBRARY_PATH"
        echo -e "${GREEN}Added CUDA libraries to LD_LIBRARY_PATH${NC}"
    fi
fi

# Create system-wide link if we have permissions (Docker container)
if [ -f /.dockerenv ]; then
    if [ -f "${LIB_PATH}" ] && [ ! -f "/usr/lib/libllama.so" ]; then
        echo -e "${YELLOW}Creating system-wide symbolic link to /usr/lib/libllama.so${NC}"
        if ln -sf "${LIB_PATH}" "/usr/lib/libllama.so" 2>/dev/null; then
            echo -e "${GREEN}Created system-wide link successfully${NC}"
        else
            echo -e "${RED}Failed to create system-wide link (permission denied)${NC}"
        fi
    elif [ -f "/usr/lib/libllama.so" ]; then
        echo -e "${GREEN}System-wide link already exists${NC}"
    fi
fi

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi

    # Check CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "${GREEN}CUDA driver version: ${CUDA_VERSION}${NC}"

    # Check for compatible CUDA libraries
    if [ -d "/usr/local/cuda" ]; then
        echo -e "${GREEN}CUDA installation found at /usr/local/cuda${NC}"
    else
        echo -e "${YELLOW}CUDA libraries not found at expected location. GPU acceleration may not work properly.${NC}"
    fi
else
    echo -e "${YELLOW}NVIDIA GPU not detected or drivers not installed.${NC}"
    echo -e "${YELLOW}GPU acceleration will not be available.${NC}"
fi

# Print success message
echo -e "${BLUE}=====================================${NC}"
echo -e "${GREEN}Library setup completed${NC}"
echo -e "${BLUE}=====================================${NC}"

if [ -f /.dockerenv ]; then
    echo -e "${YELLOW}For Docker environments, add this to your compose file:${NC}"
    echo -e "  environment:"
    echo -e "    - LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
else
    echo -e "${YELLOW}To use in your current shell:${NC}"
    echo -e "  export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    echo -e "${YELLOW}To make permanent, add to your ~/.bashrc or ~/.zshrc:${NC}"
    echo -e "  echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' >> ~/.bashrc"
fi