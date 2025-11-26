#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./update_api_key.sh <YOUR_GOOGLE_API_KEY>"
  echo "Example: ./update_api_key.sh AIzaSy..."
  exit 1
fi

API_KEY=$1

echo "🔐 Updating Kubernetes Secret..."
# Create/Update the secret securely
kubectl create secret generic jolt-secrets \
  --from-literal=GOOGLE_API_KEY="$API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "🔄 Restarting pods to pick up new key..."
# Delete pods to force them to restart and pull the new secret
kubectl delete pod -l app=jolt-creator
kubectl delete pod -l app=jolt-validator
kubectl delete pod -l app=jolt-orchestrator

echo "✅ API Key updated! Pods are restarting with the new key."
echo "Wait a few seconds and then check status with: kubectl get pods"
