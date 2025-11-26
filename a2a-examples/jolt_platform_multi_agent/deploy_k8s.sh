#!/bin/bash
# Script to build and deploy JOLT Platform to Kubernetes

set -e

# Check for API Key
if [ -z "$1" ] && [ -z "$GOOGLE_API_KEY" ]; then
  echo "❌ Error: Google API Key is required."
  echo "Usage: ./deploy_k8s.sh <YOUR_GOOGLE_API_KEY>"
  echo "   OR: export GOOGLE_API_KEY=... && ./deploy_k8s.sh"
  exit 1
fi

API_KEY="${1:-$GOOGLE_API_KEY}"

echo "🚀 Starting JOLT Platform Clean Deployment..."

# Cleanup existing resources
echo "🧹 Cleaning up existing resources..."
kubectl delete -f k8s/validator.yaml --ignore-not-found=true
kubectl delete -f k8s/creator.yaml --ignore-not-found=true
kubectl delete -f k8s/orchestrator.yaml --ignore-not-found=true
kubectl delete -f k8s/kafka.yaml --ignore-not-found=true
kubectl delete configmap jolt-config --ignore-not-found=true
kubectl delete secret jolt-secrets --ignore-not-found=true
kubectl delete job kafka-topic-creator --ignore-not-found=true

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

echo "☸️  Applying Kubernetes manifests..."

# Create Secret
echo "🔐 Creating Secrets..."
kubectl create secret generic jolt-secrets \
  --from-literal=GOOGLE_API_KEY="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply Config
kubectl apply -f k8s/config.yaml

# Deploy Kafka
echo "🐘 Deploying Kafka..."
kubectl apply -f k8s/kafka.yaml

echo "⏳ Waiting for Kafka to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/kafka || echo "⚠️ Kafka wait timed out, proceeding..."

# Create Topics
echo "📋 Creating Kafka topics..."
kubectl apply -f k8s/kafka-topics-job.yaml
kubectl wait --for=condition=complete --timeout=60s job/kafka-topic-creator || echo "⚠️ Topic creation wait timed out"

# Deploy Apps
echo "🚀 Deploying application components..."
kubectl apply -f k8s/mcp-server.yaml
kubectl apply -f k8s/orchestrator.yaml
kubectl apply -f k8s/creator.yaml
kubectl apply -f k8s/validator.yaml

echo "✅ Deployment initiated!"
echo "Run 'kubectl get pods' to check status."
echo "Use 'kubectl port-forward svc/jolt-orchestrator 8000:8000' to access the API locally."
