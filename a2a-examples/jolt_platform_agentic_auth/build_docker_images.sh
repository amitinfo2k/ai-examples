#!/bin/bash
# Script to build and load JOLT Platform Docker images
echo "📦 Building Docker images..."
docker build -t jolt-orchestrator:latest -f Dockerfile.orchestrator .
docker build -t jolt-creator:latest -f Dockerfile.creator .
docker build -t jolt-validator:latest -f Dockerfile.validator .
docker build -t mcp-server:latest -f Dockerfile.mcp .

# Load images into Kind/Minikube
if command -v kind &> /dev/null; then
    # Detect Kind cluster name
    KIND_CLUSTER=$(kind get clusters | head -n 1)
    if [ ! -z "$KIND_CLUSTER" ]; then
        echo "🔄 Loading images into Kind cluster '$KIND_CLUSTER'..."
        kind load docker-image jolt-orchestrator:latest --name "$KIND_CLUSTER"
        kind load docker-image jolt-creator:latest --name "$KIND_CLUSTER"
        kind load docker-image jolt-validator:latest --name "$KIND_CLUSTER"
        kind load docker-image mcp-server:latest --name "$KIND_CLUSTER"
    fi
elif command -v minikube &> /dev/null; then
    if minikube status | grep -q "Running"; then
        echo "🔄 Loading images into Minikube..."
        minikube image load jolt-orchestrator:latest
        minikube image load jolt-creator:latest
        minikube image load jolt-validator:latest
        minikube image load mcp-server:latest
    fi
fi