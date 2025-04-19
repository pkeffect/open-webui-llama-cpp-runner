#!/bin/bash
# Interactive script to switch between CPU and GPU versions of llama-cpp-runner

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Display banner
echo -e "${BLUE}"
echo "====================================================="
echo -e "${BOLD} 🦙 llama-cpp-runner: Control Panel ${NC}${BLUE}"
echo "====================================================="
echo -e "${NC}"

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo -e "Please install Docker to use this script: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for docker-compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: docker compose not available${NC}"
    echo -e "Please install Docker Compose or update your Docker installation"
    exit 1
fi

# Check for configuration files
if [ ! -f "compose.yaml" ]; then
    echo -e "${RED}Error: compose.yaml not found${NC}"
    echo -e "Make sure you're running this script from the project root directory"
    exit 1
fi

if [ ! -f "compose-gpu.yaml" ]; then
    echo -e "${RED}Error: compose-gpu.yaml not found${NC}"
    echo -e "Make sure you're running this script from the project root directory"
    exit 1
fi

# Function to check for NVIDIA GPU support
check_nvidia() {
    echo -e "${YELLOW}Checking for NVIDIA GPU support...${NC}"
    
    # Check for nvidia-smi
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}✗ nvidia-smi command not found${NC}"
        echo -e "${YELLOW}This indicates NVIDIA drivers are not installed${NC}"
        return 1
    fi
    
    # Check GPU detection
    if ! nvidia-smi &> /dev/null; then
        echo -e "${RED}✗ No NVIDIA GPU detected${NC}"
        return 1
    fi
    
    # Check for Docker NVIDIA runtime
    if docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${GREEN}✓ NVIDIA Container Runtime is available${NC}"
    else
        echo -e "${RED}✗ NVIDIA Container Runtime not found${NC}"
        echo -e "${YELLOW}To enable GPU support, please install NVIDIA Container Runtime:${NC}"
        echo -e "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        return 1
    fi
    
    # Get GPU information
    echo -e "${GREEN}✓ NVIDIA GPU support is available:${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    
    return 0
}

# Function to get container status
get_status() {
    cpu_running=$(docker ps --filter "name=llama-cpp-runner$" --format "{{.Status}}" | grep -v "llama-cpp-runner-gpu" || echo "")
    gpu_running=$(docker ps --filter "name=llama-cpp-runner-gpu" --format "{{.Status}}" || echo "")
    
    echo -e "\n${YELLOW}Current Status:${NC}"
    
    if [ -n "$cpu_running" ]; then
        echo -e "${GREEN}✓ CPU version is running: ${CYAN}$cpu_running${NC}"
        echo -e "${GREEN}✓ Available at: ${BOLD}http://localhost:10000${NC}"
    else
        echo -e "${YELLOW}✗ CPU version is not running${NC}"
    fi
    
    if [ -n "$gpu_running" ]; then
        echo -e "${GREEN}✓ GPU version is running: ${CYAN}$gpu_running${NC}"
        echo -e "${GREEN}✓ Available at: ${BOLD}http://localhost:10000${NC}"
    else
        echo -e "${YELLOW}✗ GPU version is not running${NC}"
    fi
}

# Function to start CPU version
start_cpu() {
    echo -e "${YELLOW}Starting CPU version...${NC}"
    docker compose -f compose.yaml up -d
    echo -e "${GREEN}CPU version started!${NC}"
    echo -e "${GREEN}Available at: ${BOLD}http://localhost:10000${NC}"
}

# Function to start GPU version
start_gpu() {
    echo -e "${YELLOW}Starting GPU version...${NC}"
    docker compose -f compose-gpu.yaml up -d
    echo -e "${GREEN}GPU version started!${NC}"
    echo -e "${GREEN}Available at: ${BOLD}http://localhost:10000${NC}"
}

# Function to stop all versions
stop_all() {
    echo -e "${YELLOW}Stopping all running versions...${NC}"
    docker compose -f compose.yaml down 2> /dev/null || true
    docker compose -f compose-gpu.yaml down 2> /dev/null || true
    echo -e "${GREEN}All versions stopped${NC}"
}

# Function to show logs
show_logs() {
    local container=$1
    if [ -z "$container" ]; then
        # Try to find a running container
        if docker ps --filter "name=llama-cpp-runner" --format "{{.Names}}" | grep -q .; then
            container=$(docker ps --filter "name=llama-cpp-runner" --format "{{.Names}}" | head -n 1)
        else
            echo -e "${RED}No running containers found${NC}"
            return 1
        fi
    fi
    
    echo -e "${YELLOW}Showing logs for $container (press Ctrl+C to exit):${NC}"
    docker logs -f "$container"
}

# Function to rebuild containers
rebuild() {
    local mode=$1
    
    if [ "$mode" = "cpu" ] || [ "$mode" = "all" ]; then
        echo -e "${YELLOW}Rebuilding CPU version...${NC}"
        docker compose -f compose.yaml build --no-cache
        echo -e "${GREEN}CPU version rebuilt${NC}"
    fi
    
    if [ "$mode" = "gpu" ] || [ "$mode" = "all" ]; then
        echo -e "${YELLOW}Rebuilding GPU version...${NC}"
        docker compose -f compose-gpu.yaml build --no-cache
        echo -e "${GREEN}GPU version rebuilt${NC}"
    fi
}

# Function to check server health
check_health() {
    cpu_running=$(docker ps --filter "name=llama-cpp-runner$" --format "{{.Names}}" | grep -v "llama-cpp-runner-gpu" || echo "")
    gpu_running=$(docker ps --filter "name=llama-cpp-runner-gpu" --format "{{.Names}}" || echo "")
    
    if [ -n "$cpu_running" ]; then
        echo -e "${YELLOW}Checking CPU server health...${NC}"
        if curl -s -f http://localhost:10000/health > /dev/null; then
            echo -e "${GREEN}✓ CPU server is healthy${NC}"
            
            # Get additional info
            models=$(curl -s http://localhost:10000/models | jq -r '.models | join(", ")' 2>/dev/null || echo "Error getting models")
            echo -e "${GREEN}✓ Available models: ${CYAN}$models${NC}"
        else
            echo -e "${RED}✗ CPU server is not responding${NC}"
        fi
    fi
    
    if [ -n "$gpu_running" ]; then
        echo -e "${YELLOW}Checking GPU server health...${NC}"
        if curl -s -f http://localhost:10000/health > /dev/null; then
            echo -e "${GREEN}✓ GPU server is healthy${NC}"
            
            # Get additional info
            models=$(curl -s http://localhost:10000/models | jq -r '.models | join(", ")' 2>/dev/null || echo "Error getting models")
            echo -e "${GREEN}✓ Available models: ${CYAN}$models${NC}"
            
            # Get GPU info if available
            if curl -s -f http://localhost:10000/gpu/info > /dev/null; then
                gpu_status=$(curl -s http://localhost:10000/gpu/info | jq -r '.available' 2>/dev/null || echo "unknown")
                if [ "$gpu_status" = "true" ]; then
                    echo -e "${GREEN}✓ GPU acceleration is active${NC}"
                else
                    echo -e "${YELLOW}✗ GPU acceleration is not active${NC}"
                fi
            fi
        else
            echo -e "${RED}✗ GPU server is not responding${NC}"
        fi
    fi
    
    if [ -z "$cpu_running" ] && [ -z "$gpu_running" ]; then
        echo -e "${YELLOW}No servers are currently running${NC}"
    fi
}

# Main menu
while true; do
    # Show current status
    get_status
    
    echo -e "\n${BLUE}Choose an option:${NC}"
    echo -e "${BOLD}1)${NC} Start CPU version"
    echo -e "${BOLD}2)${NC} Start GPU version"
    echo -e "${BOLD}3)${NC} Stop all running versions"
    echo -e "${BOLD}4)${NC} Show logs"
    echo -e "${BOLD}5)${NC} Check server health"
    echo -e "${BOLD}6)${NC} Check GPU support"
    echo -e "${BOLD}7)${NC} Rebuild containers"
    echo -e "${BOLD}q)${NC} Exit"
    
    read -p "Enter your choice: " choice
    
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
            # List containers
            running_containers=$(docker ps --filter "name=llama-cpp-runner" --format "{{.Names}}")
            
            if [ -z "$running_containers" ]; then
                echo -e "${RED}No running containers found${NC}"
            else
                echo -e "${YELLOW}Select container to view logs:${NC}"
                select container in $running_containers "Cancel"; do
                    if [ "$container" != "Cancel" ]; then
                        show_logs "$container"
                    fi
                    break
                done
            fi
            ;;
        5)
            check_health
            ;;
        6)
            check_nvidia
            ;;
        7)
            echo -e "${YELLOW}Select which container to rebuild:${NC}"
            select rebuild_option in "CPU" "GPU" "Both" "Cancel"; do
                case $rebuild_option in
                    CPU)
                        rebuild "cpu"
                        ;;
                    GPU)
                        rebuild "gpu"
                        ;;
                    Both)
                        rebuild "all"
                        ;;
                    Cancel)
                        ;;
                esac
                break
            done
            ;;
        q|Q)
            echo -e "${GREEN}Exiting. Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
    
    # Add a small delay to make the output more readable
    sleep 1
done
