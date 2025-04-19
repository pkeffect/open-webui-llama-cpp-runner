#!/bin/bash

# Helper script to easily switch between CPU and GPU versions of llama-cpp-runner

set -e  # Exit on any error

# Color outputs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}"
echo "====================================================="
echo "  llama-cpp-runner: CPU/GPU Version Switcher"
echo "====================================================="
echo -e "${NC}"

# Check if both compose files exist
if [ ! -f "compose.yaml" ]; then
    echo -e "${RED}Error: compose.yaml not found. Make sure you're in the project root directory.${NC}"
    exit 1
fi

if [ ! -f "compose-gpu.yaml" ]; then
    echo -e "${RED}Error: compose-gpu.yaml not found. Make sure you're in the project root directory.${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in the PATH.${NC}"
    exit 1
fi

# Function to check for the NVIDIA Container Toolkit
check_nvidia() {
    echo -e "${YELLOW}Checking for NVIDIA GPU support...${NC}"
    
    # Try to run a simple NVIDIA container
    if docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU support is available${NC}"
        return 0
    else
        echo -e "${RED}✗ NVIDIA GPU support is not available${NC}"
        echo -e "${YELLOW}To enable GPU support, please install the NVIDIA Container Toolkit:${NC}"
        echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
}

# Function to start the CPU version
start_cpu() {
    echo -e "${YELLOW}Starting CPU version...${NC}"
    docker compose -f compose.yaml up -d
    echo -e "${GREEN}CPU version started! Available at http://localhost:3636${NC}"
}

# Function to start the GPU version
start_gpu() {
    echo -e "${YELLOW}Starting GPU version...${NC}"
    docker compose -f compose-gpu.yaml up -d
    echo -e "${GREEN}GPU version started! Available at http://localhost:3637${NC}"
}

# Function to stop all versions
stop_all() {
    echo -e "${YELLOW}Stopping all running versions...${NC}"
    docker compose -f compose.yaml down 2> /dev/null || true
    docker compose -f compose-gpu.yaml down 2> /dev/null || true
    echo -e "${GREEN}All versions stopped.${NC}"
}

# Main menu
while true; do
    echo -e "\n${BLUE}Choose an option:${NC}"
    echo "1) Start CPU version (port 3636)"
    echo "2) Start GPU version (port 3637) - requires NVIDIA GPU"
    echo "3) Stop all running versions"
    echo "4) Check GPU support"
    echo "5) Exit"
    
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            stop_all
            start_cpu
            ;;
        2)
            if check_nvidia; then
                stop_all
                start_gpu
            else
                echo -e "${YELLOW}Starting GPU version without GPU support (will run in CPU-only mode)...${NC}"
                read -p "Continue anyway? (y/n): " continue_anyway
                if [[ $continue_anyway == "y" || $continue_anyway == "Y" ]]; then
                    stop_all
                    start_gpu
                fi
            fi
            ;;
        3)
            stop_all
            ;;
        4)
            check_nvidia
            ;;
        5)
            echo -e "${GREEN}Exiting. Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
done