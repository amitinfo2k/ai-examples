#!/bin/bash
# Setup script for Kubernetes CrashLoopBackOff Diagnosis System

set -e

echo "Setting up Kubernetes CrashLoopBackOff Diagnosis System..."

# Check if Python 3.10.12+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.10.12 or higher is required. Found: $python_version"
    echo "Please install Python 3.10.12 or higher and try again."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed."
    echo "Please install kubectl and configure access to a Kubernetes cluster."
    exit 1
fi

# Check kubectl version
kubectl_version=$(kubectl version --client -o json | grep -o '"gitVersion": "v[0-9.]*"' | head -1 | cut -d'"' -f4 | cut -d'v' -f2)
required_k8s_version="1.30.0"

if [ "$(printf '%s\n' "$required_k8s_version" "$kubectl_version" | sort -V | head -n1)" != "$required_k8s_version" ]; then
    echo "Warning: Kubernetes v1.30.4 or higher is recommended. Found: $kubectl_version"
    echo "The system may not work correctly with older versions."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv sync

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file and add your Gemini API key."
fi

echo "Setup complete!"
echo
echo "To run the system:"
echo "1. Make sure your .env file is configured with your Gemini API key"
echo "2. Run all services: python main.py run-all"
echo "3. In another terminal, troubleshoot a pod: python main.py troubleshoot -n <namespace> -p <pod_name>"
echo
echo "To run a full test with a simulated CrashLoopBackOff pod:"
echo "python test/run_full_test.py"
