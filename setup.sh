#!/usr/bin/env bash
set -e

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version > /dev/null 2>&1; then
    echo "Error: docker compose is not available."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    if [ ! -f .env.sample ]; then
        echo "Error: .env.sample not found. Please create it first."
        exit 1
    fi
    
    cp .env.sample .env
    echo "Created .env from .env.sample."

    # Setup Hugging Face Token
    read -rsp "Enter your Hugging Face token: " HF_TOKEN
    echo

    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: No token provided. Model download may fail if the repo is private."
    else
        sed -i.bak "s/^HF_TOKEN=.*/HF_TOKEN=$HF_TOKEN/" .env
        rm -f .env.bak
    fi

    echo ""
    
    # Setup GPU / CPU Support
    read -p "Do you want to build with GPU (CUDA) support? (y/N): " USE_GPU
    if [[ "$USE_GPU" =~ ^[Yy]$ ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        echo "Configuring for GPU (CUDA 12.1)..."
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        echo "Configuring for CPU..."
    fi

    if grep -q "^TORCH_INDEX=" .env; then
        sed -i.bak "s|^TORCH_INDEX=.*|TORCH_INDEX=$TORCH_INDEX|" .env
        rm -f .env.bak
    else
        echo "TORCH_INDEX=$TORCH_INDEX" >> .env
    fi
fi

# Create necessary directories
mkdir -p checkpoints/convnext
mkdir -p outputs/checkpoints

echo ""
echo "Building and starting containers..."
docker compose up --build -d

echo ""
echo "=========================================="
echo "  Deepfake Detection is starting..."
echo "=========================================="
echo ""
echo "  Frontend: http://localhost:7860"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "  View logs: docker compose logs -f"
echo "  Stop:      docker compose down"
echo "=========================================="

# Show logs
docker compose logs -f --tail=50