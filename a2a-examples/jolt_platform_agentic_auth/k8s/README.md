# Kubernetes Deployment Guide

## 🎯 Overview

This guide covers deploying the JOLT Multi-Agent Platform to Kubernetes with:
- **Loose Coupling**: Each agent runs in a separate container
- **Kafka Message Bus**: External Kafka for reliable A2A communication
- **Scalable Architecture**: Horizontally scalable agent workers

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  ┌──────────────┐    ┌──────────────┐   ┌────────────┐ │
│  │ Orchestrator │◄───┤    Kafka     │───►│  Creator   │ │
│  │  (API Pod)   │    │  (Message    │   │   Agent    │ │
│  └──────────────┘    │    Bus)      │   │    Pod     │ │
│                      └──────────────┘   └────────────┘ │
│                             ▲                           │
│                             │                           │
│                             ▼                           │
│                      ┌────────────┐                     │
│                      │ Validator  │                     │
│                      │   Agent    │                     │
│                      │    Pod     │                     │
│                      └────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## 📋 Prerequisites

1. **Kubernetes Cluster**
   - Minikube (local development)
   - Kind (local development)
   - GKE/EKS/AKS (production)

2. **Tools**
   - `kubectl` installed and configured
   - `docker` installed
   - (Optional) `minikube` or `kind` for local cluster

3. **Google API Key**
   - Get from: https://makersuite.google.com/app/apikey

## 🚀 Quick Start

### 1. Configure Secrets

Edit `k8s/config.yaml` and replace the placeholder API key:

```yaml
stringData:
  GOOGLE_API_KEY: "your-actual-google-api-key"
```

### 2. Deploy

Run the deployment script:

```bash
./deploy_k8s.sh
```

This will:
- Build Docker images for all components
- Load images into your cluster (if using Minikube/Kind)
- Deploy Kafka, Orchestrator, Creator, and Validator

### 3. Verify Deployment

```bash
# Check all pods are running
kubectl get pods

# Expected output:
# NAME                                READY   STATUS    RESTARTS   AGE
# jolt-orchestrator-xxx               1/1     Running   0          1m
# jolt-creator-xxx                    1/1     Running   0          1m
# jolt-validator-xxx                  1/1     Running   0          1m
# kafka-xxx                           1/1     Running   0          2m
# zookeeper-xxx                       1/1     Running   0          2m
```

### 4. Access the API

```bash
# Port forward to access locally
kubectl port-forward svc/jolt-orchestrator 8000:8000

# In another terminal, test the API
curl http://localhost:8000/health
```

## 📦 Components

### Orchestrator (API Server)
- **Image**: `jolt-orchestrator:latest`
- **Port**: 8000
- **Service**: LoadBalancer
- **Replicas**: 1

### Creator Agent
- **Image**: `jolt-creator:latest`
- **Role**: Listens to `START_WORKFLOW` messages
- **Replicas**: 1 (can scale up)

### Validator Agent
- **Image**: `jolt-validator:latest`
- **Role**: Listens to `SPEC_CREATED` messages
- **Replicas**: 1 (can scale up)

### Kafka
- **Image**: `wurstmeister/kafka`
- **Port**: 9092
- **Topics**: Auto-created (START_WORKFLOW, SPEC_CREATED, etc.)

## 🔧 Configuration

### Environment Variables

Set in `k8s/config.yaml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | - | Google Gemini API key (Secret) |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Gemini model to use |
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka connection string |
| `KAFKA_GROUP_ID` | `jolt-group` | Kafka consumer group |

### Scaling Agents

Scale Creator agents:
```bash
kubectl scale deployment jolt-creator --replicas=3
```

Scale Validator agents:
```bash
kubectl scale deployment jolt-validator --replicas=3
```

## 🧪 Testing

### Test via API

```bash
# Create and validate a JOLT spec
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"firstName": "John", "lastName": "Doe"}},
    "expected_output": {"fullName": "John Doe"},
    "execution_mode": "a2a"
  }'
```

### View Logs

```bash
# Orchestrator logs
kubectl logs -f deployment/jolt-orchestrator

# Creator logs
kubectl logs -f deployment/jolt-creator

# Validator logs
kubectl logs -f deployment/jolt-validator

# Kafka logs
kubectl logs -f deployment/kafka
```

## 🐛 Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Agent Not Receiving Messages

```bash
# Check Kafka is running
kubectl get pods -l app=kafka

# Check Kafka logs
kubectl logs deployment/kafka

# Verify topics exist
kubectl exec -it deployment/kafka -- kafka-topics.sh \
  --list --bootstrap-server localhost:9092
```

### API Key Issues

```bash
# Verify secret is created
kubectl get secret jolt-secrets

# Check if env vars are set
kubectl exec deployment/jolt-creator -- env | grep GOOGLE_API_KEY
```

## 🧹 Cleanup

Remove all resources:

```bash
./cleanup_k8s.sh
```

Or manually:

```bash
kubectl delete -f k8s/
```

## 🔄 Local Testing with Docker Compose

Before deploying to K8s, test locally:

```bash
# Set your API key
export GOOGLE_API_KEY="your-key-here"

# Start all services
docker-compose up

# Access API at http://localhost:8000
```

## 🚀 Production Considerations

### 1. Use Managed Kafka
Replace the simple Kafka deployment with:
- **Confluent Cloud**
- **AWS MSK**
- **Google Cloud Pub/Sub** (with adapter)

### 2. Secrets Management
Use Kubernetes Secrets or external secret managers:
- **Google Secret Manager**
- **AWS Secrets Manager**
- **HashiCorp Vault**

### 3. Resource Limits

Add resource requests/limits to deployments:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

### 4. Health Checks

Add liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 5. Monitoring

Deploy:
- **Prometheus** for metrics
- **Grafana** for dashboards
- **ELK Stack** for logging

## 📊 Architecture Benefits

✅ **Loose Coupling**: Agents are independent services  
✅ **Scalability**: Scale each agent independently  
✅ **Reliability**: Kafka provides message persistence  
✅ **Fault Tolerance**: Agents can fail and restart  
✅ **Observability**: Centralized logging and monitoring  

## 🔗 Related Documentation

- [MCP Integration](../MCP_INTEGRATION.md)
- [A2A Guide](../A2A_GUIDE.md)
- [Architecture](../ARCHITECTURE.md)

---

**Ready for production deployment!** 🚀
