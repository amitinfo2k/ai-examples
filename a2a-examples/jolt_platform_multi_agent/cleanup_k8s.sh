#!/bin/bash
# Script to clean up JOLT Platform from Kubernetes

echo "🧹 Cleaning up JOLT Platform from Kubernetes..."

echo "🗑️  Deleting application deployments..."
kubectl delete -f k8s/validator.yaml --ignore-not-found=true
kubectl delete -f k8s/creator.yaml --ignore-not-found=true
kubectl delete -f k8s/orchestrator.yaml --ignore-not-found=true

echo "🗑️  Deleting Kafka and Zookeeper..."
kubectl delete -f k8s/kafka.yaml --ignore-not-found=true

echo "🗑️  Deleting Kafka topic creation job..."
kubectl delete job kafka-topic-creator --ignore-not-found=true

echo "🗑️  Deleting ConfigMap and Secret..."
kubectl delete configmap jolt-config --ignore-not-found=true
kubectl delete secret jolt-secrets --ignore-not-found=true

echo "🗑️  Deleting any remaining pods..."
kubectl delete pods -l app=jolt-orchestrator --ignore-not-found=true
kubectl delete pods -l app=jolt-creator --ignore-not-found=true
kubectl delete pods -l app=jolt-validator --ignore-not-found=true
kubectl delete pods -l app=kafka --ignore-not-found=true
kubectl delete pods -l app=zookeeper --ignore-not-found=true

echo "✅ Cleanup completed!"
echo "Run 'kubectl get all' to verify all resources are deleted."
