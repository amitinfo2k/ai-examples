# JOLT Platform User Guide

Complete guide for using the JOLT Multi-Agent Platform.

## Table of Contents

- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Execution Modes](#execution-modes)
- [Best Practices](#best-practices)
- [FAQ](#faq)

## Getting Started

### Quick Test

```bash
# Start API server (local)
uvicorn jolt_platform.api_server:app --port 8000

# Or use Kubernetes
kubectl port-forward svc/jolt-orchestrator 8000:8000

# Test health
curl http://localhost:8000/health
```

### Your First Workflow

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {
      "user": {
        "firstName": "Alice",
        "lastName": "Smith"
      }
    },
    "expected_output": {
      "fullName": "Alice Smith"
    },
    "execution_mode": "traditional"
  }'
```

## API Reference

### Base URL

- **Local**: `http://localhost:8000`
- **Kubernetes**: `http://localhost:8000` (via port-forward)

### Endpoints

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "ok"
}
```

#### POST /create

Create JOLT specification only.

**Request**:
```json
{
  "input_json": {...},
  "expected_output": {...}
}
```

**Response**:
```json
{
  "status": "success",
  "agent": "CrewAI",
  "timestamp": "2025-11-25T10:00:00",
  "jolt_spec": [...]
}
```

#### POST /validate

Validate existing JOLT specification.

**Request**:
```json
{
  "jolt_spec": [...],
  "input_json": {...},
  "expected_output": {...}
}
```

**Response**:
```json
{
  "status": "success",
  "agent": "LangChain",
  "timestamp": "2025-11-25T10:00:00",
  "validation_report": {
    "is_valid": true,
    "matches": true,
    "actual_output": {...},
    "expected_output": {...},
    "differences": []
  }
}
```

#### POST /workflow

Complete workflow: create and validate.

**Request**:
```json
{
  "input_json": {...},
  "expected_output": {...},
  "execution_mode": "a2a",  // "traditional" or "a2a"
  "async_mode": false       // true for background execution
}
```

**Response (A2A mode)**:
```json
{
  "status": "success",
  "execution_mode": "a2a",
  "job_id": "uuid-here",
  "agents": {
    "creation": "CrewAI",
    "validation": "LangChain"
  },
  "timestamp": "2025-11-25T10:00:00",
  "result": {
    "status": "initiated",
    "job_id": "uuid-here",
    "message": "Workflow started successfully"
  }
}
```

#### GET /workflow/{job_id}

Check workflow status.

**Response**:
```json
{
  "job_id": "uuid-here",
  "status": "completed",  // or "running", "failed"
  "created_at": "2025-11-25T10:00:00",
  "completed_at": "2025-11-25T10:00:05",
  "execution_mode": "a2a",
  "result": {
    "jolt_spec": [...],
    "validation_report": {...}
  }
}
```

## Examples

### Example 1: Simple Name Transformation

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {
      "user": {
        "name": "John Doe"
      }
    },
    "expected_output": {
      "fullName": "John Doe"
    },
    "execution_mode": "traditional"
  }'
```

### Example 2: Nested Object Transformation

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {
      "customer": {
        "profile": {
          "firstName": "Alice",
          "lastName": "Smith"
        },
        "contact": {
          "email": "alice@example.com"
        }
      }
    },
    "expected_output": {
      "name": "Alice Smith",
      "email": "alice@example.com"
    }
  }'
```

### Example 3: Array Transformation

```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {
      "orders": [
        {"id": 1, "amount": 100},
        {"id": 2, "amount": 200}
      ]
    },
    "expected_output": {
      "orderIds": [1, 2]
    }
  }'
```

### Example 4: Async Workflow (A2A Mode)

```bash
# Submit workflow
RESPONSE=$(curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"name": "Test"}},
    "expected_output": {"fullName": "Test"},
    "execution_mode": "a2a"
  }')

# Extract job_id
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

# Check status
curl http://localhost:8000/workflow/$JOB_ID

# Poll until complete
while true; do
  STATUS=$(curl -s http://localhost:8000/workflow/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 2
done

# Get final result
curl http://localhost:8000/workflow/$JOB_ID | jq '.'
```

### Example 5: Using Python

```python
import requests
import json
import time

API_URL = "http://localhost:8000"

# Submit workflow
response = requests.post(
    f"{API_URL}/workflow",
    json={
        "input_json": {"user": {"name": "Python Test"}},
        "expected_output": {"fullName": "Python Test"},
        "execution_mode": "a2a"
    }
)

result = response.json()
job_id = result["job_id"]
print(f"Job ID: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f"{API_URL}/workflow/{job_id}")
    status = status_response.json()
    
    print(f"Status: {status['status']}")
    
    if status['status'] in ['completed', 'failed']:
        print(json.dumps(status, indent=2))
        break
    
    time.sleep(2)
```

## Execution Modes

### Traditional Mode

**Use when**:
- Single, synchronous request
- Immediate response needed
- Simple workflows
- Local development

**Characteristics**:
- Blocking API call
- Direct agent invocation
- Lower latency
- No job tracking

**Example**:
```json
{
  "execution_mode": "traditional"
}
```

### A2A Mode (Event-Driven)

**Use when**:
- Distributed deployment
- High throughput
- Fault tolerance needed
- Async processing

**Characteristics**:
- Non-blocking API call
- Kafka message bus
- Job tracking
- Scalable agents

**Example**:
```json
{
  "execution_mode": "a2a"
}
```

## Best Practices

### 1. Input Data

- **Use valid JSON**: Ensure input is well-formed
- **Simplify structure**: Avoid unnecessary nesting
- **Test incrementally**: Start simple, add complexity

### 2. Expected Output

- **Be specific**: Define exact structure wanted
- **Match data types**: Ensure types align (string vs number)
- **Test edge cases**: Empty arrays, null values, etc.

### 3. Error Handling

- **Check status codes**: 200 = success, 4xx = client error, 5xx = server error
- **Read error messages**: API provides detailed error info
- **Retry on failure**: Implement retry logic for A2A mode

### 4. Performance

- **Use A2A for scale**: Event-driven handles load better
- **Batch requests**: Group similar transformations
- **Cache results**: Reuse JOLT specs when possible

### 5. Production

- **Monitor job queue**: Track pending jobs
- **Set timeouts**: Don't wait indefinitely
- **Log workflows**: Keep audit trail
- **Version specs**: Track JOLT spec versions

## FAQ

### Q: How long does a workflow take?

**A**: Typically 5-15 seconds depending on complexity and AI model response time.

### Q: Can I reuse a JOLT spec?

**A**: Yes! Save the `jolt_spec` from the response and use the `/validate` endpoint directly.

### Q: What if the AI generates an incorrect spec?

**A**: The validator will detect mismatches. You can then manually adjust the spec or provide more specific examples.

### Q: How many agents can I run?

**A**: In Kubernetes, scale as needed. Each agent replica can process messages independently.

### Q: Can I use my own Kafka?

**A**: Yes! Set `KAFKA_BOOTSTRAP_SERVERS` to your Kafka cluster.

### Q: How do I debug issues?

**A**: Check logs:
```bash
kubectl logs -l app=jolt-creator --tail=50
kubectl logs -l app=jolt-validator --tail=50
kubectl logs -l app=jolt-orchestrator --tail=50
```

### Q: What Gemini models are supported?

**A**: Check available models:
```bash
kubectl exec deployment/jolt-creator -- python list_models.py
```

Common models:
- `gemini-2.0-flash` (fast, recommended)
- `gemini-2.5-pro` (higher quality)
- `gemini-2.0-flash-exp` (experimental)

### Q: How do I change the model?

**A**:
```bash
kubectl patch configmap jolt-config \
  -p '{"data":{"GEMINI_MODEL":"gemini-2.5-pro"}}'
kubectl rollout restart deployment/jolt-creator
kubectl rollout restart deployment/jolt-validator
```

### Q: Can I run without Kubernetes?

**A**: Yes! Use Docker Compose for local testing or run directly with Python.

### Q: How do I monitor the system?

**A**: Use Prometheus + Grafana. See [DEPLOYMENT.md](DEPLOYMENT.md#monitoring--observability).

### Q: What's the difference between `/create` and `/workflow`?

**A**:
- `/create`: Only generates JOLT spec (no validation)
- `/workflow`: Generates and validates (complete pipeline)

### Q: How do I handle rate limits?

**A**: Implement client-side rate limiting or use A2A mode with multiple agents to distribute load.

## Troubleshooting

### API Returns 500 Error

Check orchestrator logs:
```bash
kubectl logs -l app=jolt-orchestrator --tail=50
```

Common causes:
- Invalid API key
- Model not available
- Kafka connection error

### Job Stuck in "running" Status

Check agent logs:
```bash
kubectl logs -l app=jolt-creator --tail=30
kubectl logs -l app=jolt-validator --tail=30
```

Common causes:
- Agent crashed
- Kafka topic doesn't exist
- Message not consumed

### Validation Always Fails

- Check if expected output matches actual structure
- Verify JOLT spec is correct
- Try `/create` endpoint alone to see generated spec

### No Response from API

- Verify pod is running: `kubectl get pods`
- Check port-forward is active: `kubectl port-forward svc/jolt-orchestrator 8000:8000`
- Test health endpoint: `curl http://localhost:8000/health`

## Next Steps

- **Explore Examples**: Try different JSON transformations
- **Scale Agents**: Test with multiple replicas
- **Monitor Performance**: Set up Grafana dashboards
- **Customize Agents**: Modify agent behavior for your use case

---

**Questions?** Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details or [DEPLOYMENT.md](DEPLOYMENT.md) for deployment help.
