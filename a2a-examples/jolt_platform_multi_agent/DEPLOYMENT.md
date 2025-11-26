# JOLT Platform Deployment Guide

Complete guide for deploying the JOLT Multi-Agent Platform to various environments.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Compose](#docker-compose)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Production Checklist](#production-checklist)

## Prerequisites

### Software Requirements

- **Python**: 3.10 or higher
- **Docker**: 20.10+ (for containerization)
- **kubectl**: 1.24+ (for Kubernetes)
- **Kind** or **Minikube**: For local Kubernetes cluster
- **curl**: For API testing

### API Keys

- **Google Gemini API Key**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Local Development

### 1. Set Environment Variables

```bash
export GOOGLE_API_KEY="your-google-api-key"
export GEMINI_MODEL="gemini-2.0-flash"
```

### 2. Run Quickstart

```bash
python quickstart.py
```

### 3. Run API Server

```bash
uvicorn jolt_platform.api_server:app --host 0.0.0.0 --port 8000
```

### 4. Test API

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"name": "Test"}},
    "expected_output": {"fullName": "Test"},
    "execution_mode": "traditional"
  }'
```

## Docker Compose

### 1. Update API Key

Edit `docker-compose.yaml` and set your API key:

```yaml
environment:
  - GOOGLE_API_KEY=your-actual-api-key
```

### 2. Start Services

```bash
docker-compose up
```

This starts:
- Zookeeper
- Kafka
- Orchestrator (API server)
- Creator Agent
- Validator Agent

### 3. Test Workflow

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"name": "Alice"}},
    "expected_output": {"fullName": "Alice"},
    "execution_mode": "a2a"
  }'
```

### 4. View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f orchestrator
docker-compose logs -f creator
docker-compose logs -f validator
```

### 5. Stop Services

```bash
docker-compose down -v
```

## Kubernetes Deployment

### Setup Local Cluster

#### Option 1: Kind

```bash
kind create cluster --name jolt-platform
```

#### Option 2: Minikube

```bash
minikube start --cpus=4 --memory=8192
```

### Deploy Using Script

The deployment script handles everything:

```bash
./deploy_k8s.sh YOUR_GOOGLE_API_KEY
```

**What it does**:
1. ✅ Cleans up existing resources
2. ✅ Builds Docker images
3. ✅ Loads images into cluster
4. ✅ Creates Kubernetes Secret with API key
5. ✅ Deploys ConfigMap
6. ✅ Deploys Kafka & Zookeeper
7. ✅ Creates Kafka topics
8. ✅ Deploys Orchestrator, Creator, and Validator

### Manual Deployment

If you prefer manual steps:

#### 1. Create Secret

```bash
kubectl create secret generic jolt-secrets \
  --from-literal=GOOGLE_API_KEY='your-api-key'
```

#### 2. Apply ConfigMap

```bash
kubectl apply -f k8s/config.yaml
```

#### 3. Deploy Kafka

```bash
kubectl apply -f k8s/kafka.yaml
kubectl wait --for=condition=available --timeout=120s deployment/kafka
```

#### 4. Create Kafka Topics

```bash
kubectl apply -f k8s/kafka-topics-job.yaml
kubectl wait --for=condition=complete --timeout=60s job/kafka-topic-creator
```

#### 5. Deploy Application

```bash
kubectl apply -f k8s/orchestrator.yaml
kubectl apply -f k8s/creator.yaml
kubectl apply -f k8s/validator.yaml
```

### Verify Deployment

#### Check Pods

```bash
kubectl get pods
```

Expected output:
```
NAME                                READY   STATUS      RESTARTS   AGE
jolt-creator-xxx                    1/1     Running     0          2m
jolt-orchestrator-xxx               1/1     Running     0          2m
jolt-validator-xxx                  1/1     Running     0          2m
kafka-xxx                           1/1     Running     0          3m
kafka-topic-creator-xxx             0/1     Completed   0          3m
zookeeper-xxx                       1/1     Running     0          3m
```

#### Check Logs

```bash
# Orchestrator
kubectl logs -l app=jolt-orchestrator --tail=20

# Creator
kubectl logs -l app=jolt-creator --tail=20

# Validator
kubectl logs -l app=jolt-validator --tail=20

# Kafka
kubectl logs -l app=kafka --tail=20
```

#### Verify Topics

```bash
kubectl exec -it deployment/kafka -- kafka-topics \
  --list --bootstrap-server localhost:9092
```

Expected topics:
- START_WORKFLOW
- SPEC_CREATED
- VALIDATION_COMPLETED
- WORKFLOW_COMPLETE
- WORKFLOW_ERROR

### Access API

#### Port Forward

```bash
kubectl port-forward svc/jolt-orchestrator 8000:8000
```

#### Test Workflow

```bash
# Test health
curl http://localhost:8000/health

# Run workflow
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"firstName": "John", "lastName": "Doe"}},
    "expected_output": {"fullName": "John Doe"},
    "execution_mode": "a2a"
  }'

# Get job_id from response, then check status
curl http://localhost:8000/workflow/{job_id}
```

### Scale Deployment

```bash
# Scale Creator agents
kubectl scale deployment jolt-creator --replicas=3

# Scale Validator agents
kubectl scale deployment jolt-validator --replicas=2

# Verify
kubectl get pods
```

### Update Configuration

#### Update Model

```bash
kubectl patch configmap jolt-config \
  -p '{"data":{"GEMINI_MODEL":"gemini-2.0-flash"}}'
  
# Restart pods to pick up change
kubectl rollout restart deployment/jolt-creator
kubectl rollout restart deployment/jolt-validator
kubectl rollout restart deployment/jolt-orchestrator
```

#### Update API Key

```bash
./update_api_key.sh NEW_API_KEY
```

Or manually:

```bash
kubectl create secret generic jolt-secrets \
  --from-literal=GOOGLE_API_KEY='new-api-key' \
  --dry-run=client -o yaml | kubectl apply -f -
  
kubectl rollout restart deployment/jolt-orchestrator
kubectl rollout restart deployment/jolt-creator
kubectl rollout restart deployment/jolt-validator
```

## Configuration

### ConfigMap (`k8s/config.yaml`)

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: jolt-config
data:
  GEMINI_MODEL: "gemini-2.0-flash"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
  KAFKA_GROUP_ID: "jolt-group"
```

### Secret (Created via script or manually)

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jolt-secrets
type: Opaque
stringData:
  GOOGLE_API_KEY: "your-api-key"
```

### Resource Limits

Edit deployments to adjust resources:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl get pods

# Describe pod
kubectl describe pod <pod-name>

# Check logs
kubectl logs <pod-name>
```

Common issues:
- **ImagePullBackOff**: Image not loaded into cluster
- **CrashLoopBackOff**: Check logs for errors
- **Pending**: Insufficient resources

### Kafka Issues

```bash
# Check Kafka logs
kubectl logs -l app=kafka

# Verify topics
kubectl exec -it deployment/kafka -- kafka-topics \
  --list --bootstrap-server localhost:9092

# Test topic creation manually
kubectl exec -it deployment/kafka -- kafka-topics \
  --create --if-not-exists \
  --bootstrap-server localhost:9092 \
  --topic test-topic \
  --partitions 1 \
  --replication-factor 1
```

### API Errors

```bash
# Check orchestrator logs
kubectl logs -l app=jolt-orchestrator --tail=50

# Port forward and test
kubectl port-forward svc/jolt-orchestrator 8000:8000
curl http://localhost:8000/health
```

### Agent Not Responding

```bash
# Check creator
kubectl logs -l app=jolt-creator --tail=50

# Check validator
kubectl logs -l app=jolt-validator --tail=50

# Restart agents
kubectl rollout restart deployment/jolt-creator
kubectl rollout restart deployment/jolt-validator
```

### Job Status Not Updating

The orchestrator runs a background Kafka consumer. Check:

```bash
kubectl logs -l app=jolt-orchestrator | grep "background"
```

Should see:
```
🚀 Starting background status consumer...
🔄 Updating status for job {job_id}: WORKFLOW_COMPLETE
```

## Production Checklist

### Security

- [ ] Use managed Kubernetes (GKE, EKS, AKS)
- [ ] Enable TLS for Kafka
- [ ] Use Kubernetes Secrets for API keys
- [ ] Implement network policies
- [ ] Enable RBAC
- [ ] Use service accounts
- [ ] Scan images for vulnerabilities

### Scalability

- [ ] Configure Horizontal Pod Autoscaling
- [ ] Use managed Kafka (Confluent Cloud, AWS MSK)
- [ ] Set appropriate resource limits
- [ ] Enable pod disruption budgets
- [ ] Configure multiple Kafka partitions

### Reliability

- [ ] Set up health checks
- [ ] Configure liveness/readiness probes
- [ ] Use persistent volumes for Kafka
- [ ] Implement backup strategies
- [ ] Set up disaster recovery

### Monitoring

- [ ] Deploy Prometheus
- [ ] Set up Grafana dashboards
- [ ] Configure alerting
- [ ] Enable distributed tracing
- [ ] Set up logging (ELK/Loki)

### Performance

- [ ] Tune Kafka settings
- [ ] Optimize JVM for Kafka
- [ ] Configure message batching
- [ ] Enable compression
- [ ] Use connection pooling

### Deployment

- [ ] Use GitOps (ArgoCD/Flux)
- [ ] Implement CI/CD pipelines
- [ ] Version Docker images
- [ ] Use Helm charts
- [ ] Implement blue-green deployment

## Cleanup

### Kubernetes

```bash
./cleanup_k8s.sh
```

Or manually:

```bash
kubectl delete -f k8s/validator.yaml
kubectl delete -f k8s/creator.yaml
kubectl delete -f k8s/orchestrator.yaml
kubectl delete -f k8s/kafka.yaml
kubectl delete job kafka-topic-creator
kubectl delete configmap jolt-config
kubectl delete secret jolt-secrets
```

### Docker Compose

```bash
docker-compose down -v
```

### Delete Cluster

```bash
# Kind
kind delete cluster --name jolt-platform

# Minikube
minikube delete
```

## Next Steps

- **Monitoring**: Set up Prometheus and Grafana
- **Logging**: Deploy ELK or Loki stack
- **Security**: Implement mTLS and auth
- **Scaling**: Configure HPA and cluster autoscaling
- **CI/CD**: Automate deployments

---

**Need Help?** Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design details or [USER_GUIDE.md](USER_GUIDE.md) for usage examples.
