# JOLT Multi-Agent Platform Architecture

## System Overview

The JOLT Multi-Agent Platform is a production-ready, Kubernetes-native system for automated JOLT specification creation and validation using multiple AI agents.

## High-Level Architecture

```mermaid
graph TB
    User[User/API Client] --> API[Agentic Orchestrator<br/>FastAPI Server]
    API --> MB{Message Bus}
    MB --> |START_WORKFLOW| Creator[Creator Agent<br/>CrewAI]
    Creator --> |SPEC_CREATED| MB
    MB --> Validator[Validator Agent<br/>LangChain]
    Validator --> |VALIDATION_COMPLETED| MB
    MB --> API
    API --> |HUMAN_REVIEW_REQUESTED| User
    User --> |FEEDBACK_RECEIVED| API
    API --> |FEEDBACK_RECEIVED| MB
    MB --> |FEEDBACK_RECEIVED| Creator
    MB --> |WORKFLOW_COMPLETE| API
    Validator --> MCP[MCP Server<br/>JOLT Transforms]
    
    style API fill:#4CAF50
    style Creator fill:#2196F3
    style Validator fill:#FF9800
    style MB fill:#9C27B0
    style MCP fill:#F44336
```

## Execution Modes

### 1. Traditional Mode (Synchronous)

```
User → API → Platform.create_and_validate()
         ↓
    Creator Agent → JOLT Spec
         ↓
    Validator Agent → Validation Report
         ↓
    Response to User
```

**Characteristics**:
- Synchronous execution
- Direct method calls
- Simpler debugging
- Lower latency for single requests

### 2. A2A Mode (Event-Driven)

```
User → API → Kafka[START_WORKFLOW] → Creator Agent
                ↓                          ↓
         Job ID returned            Kafka[SPEC_CREATED]
                                           ↓
         Background Consumer ← Validator Agent
                ↓                          ↓
         Updates Job Status      Kafka[WORKFLOW_COMPLETE]
                ↓
         User polls /workflow/{job_id}
```

**Characteristics**:
- Asynchronous execution
- Event-driven messaging
- Horizontal scaling
- Fault tolerance

## Component Architecture

### 1. API Server (Orchestrator)

**File**: `jolt_platform/api_server.py`

**Responsibilities**:
- REST API endpoints
- Job management
- Message bus initialization
- Background Kafka consumer for status updates

**Key Features**:
- FastAPI framework
- Async/sync workflow support
- Job tracking with in-memory store
- Background thread for Kafka consumption

### 2. Creator Agent (CrewAI)

**File**: `agents/crewai_jolt_agent.py`

**Responsibilities**:
- Generate JOLT specifications
- Collaborate with AI models
- Publish results to message bus

**Workflow**:
1. Receive `START_WORKFLOW` message
2. Analyze input/output JSON
3. Generate JOLT spec using CrewAI
4. Publish `SPEC_CREATED` message

### 3. Validator Agent (LangChain)

**File**: `agents/langchain_validation_agent.py`

**Responsibilities**:
- Validate JOLT specifications
- Transform data via MCP server
- Compare actual vs expected output
- Generate validation reports

**Workflow**:
1. Receive `SPEC_CREATED` message
2. Call MCP server for transformation
3. Validate output against expected
4. Publish `WORKFLOW_COMPLETE` message

### 4. Message Bus

**File**: `jolt_platform/messaging.py`

**Implementations**:

#### InMemoryMessageBus
- For local development
- Synchronous callbacks
- No external dependencies

#### KafkaMessageBus
- For distributed deployment
- Asynchronous messaging
- Kafka consumer/producer
- Topic-based routing

**Factory Pattern**:
```python
def get_message_bus() -> MessageBus:
    if os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        return KafkaMessageBus(...)
    return InMemoryMessageBus()
```

### 5. MCP Server (Standalone HTTP Service)

**File**: `mcp_servers/mcp_http_server.py`

**Deployment**: Standalone Kubernetes service (2 replicas)

**Responsibilities**:
- JOLT transformations via HTTP API
- Load-balanced request handling
- Stateless transformation processing

**Architecture**:
```
Validator → HTTP Request → MCP Server Service (LoadBalancer)
                                    ↓
                          ┌─────────┴──────────┐
                          ↓                    ↓
                    MCP Server Pod 1    MCP Server Pod 2
                    (Port 8080)         (Port 8080)
```

**Communication**:
- **Protocol**: HTTP/REST
- **Endpoint**: `http://mcp-server:8080/transform`
- **Request Format**:
  ```json
  {
    "jolt_spec": [...],
    "input_json": {...}
  }
  ```
- **Response Format**:
  ```json
  {
    "success": true,
    "result": {...},
    "error": null
  }
  ```

**Benefits**:
- **Scalability**: Independent horizontal scaling (currently 2 replicas)
- **Performance**: No subprocess overhead, persistent process
- **Observability**: Dedicated logs and metrics
- **Reliability**: Load balanced, automatic pod restart
- **Maintainability**: Clear service boundary

**Debug Logging**:
- Shows received requests with full JOLT spec and input JSON
- Shows transformation results
- All logs formatted with JSON indentation for readability
- Uses emoji prefixes (🔄, 📥, 📋, 🔧, ✅, ❌) for easy scanning

## Message Flow

### Kafka Topics

| Topic | Producer | Consumer | Purpose |
|-------|----------|----------|---------|
| `START_WORKFLOW` | Orchestrator | Creator | Initiate workflow |
| `SPEC_CREATED` | Creator | Validator | Pass JOLT spec |
| `VALIDATION_COMPLETED` | Validator | - | Intermediate result |
| `WORKFLOW_COMPLETE` | Validator | Orchestrator | Final success |
| `WORKFLOW_ERROR` | Any | Orchestrator | Error handling |

### Message Schema

#### START_WORKFLOW
```json
{
  "job_id": "uuid",
  "input_json": {...},
  "expected_output": {...}
}
```

#### SPEC_CREATED
```json
{
  "job_id": "uuid",
  "jolt_spec": [...],
  "input_json": {...},
  "expected_output": {...}
}
```

#### WORKFLOW_COMPLETE
```json
{
  "job_id": "uuid",
  "status": "success",
  "result": {
    "jolt_spec": [...],
    "validation_report": {...}
  }
}
```

## Kubernetes Architecture

### Pod Structure

```
┌─────────────────────────────────────┐
│         Kubernetes Cluster          │
│                                     │
│  ┌────────────────────────────────┐│
│  │  jolt-orchestrator (Deployment)││
│  │  - FastAPI server              ││
│  │  - Background Kafka consumer   ││
│  │  - Port 8000                   ││
│  └────────────────────────────────┘│
│                                     │
│  ┌────────────────────────────────┐│
│  │  jolt-creator (Deployment)     ││
│  │  - CrewAI agent                ││
│  │  - Kafka consumer              ││
│  │  - Scalable (replicas: N)     ││
│  └────────────────────────────────┘│
│                                     │
│  ┌────────────────────────────────┐│
│  │  jolt-validator (Deployment)   ││
│  │  - LangChain agent             ││
│  │  - HTTP client to MCP Server  ││
│  │  - Scalable (replicas: M)     ││
│  └────────────────────────────────┘│
│                                     │
│  ┌────────────────────────────────┐│
│  │  mcp-server (Deployment)       ││
│  │  - FastAPI HTTP server         ││
│  │  - JOLT transformation engine  ││
│  │  - Replicas: 2 (default)      ││
│  │  - Port 8080                   ││
│  │  - ClusterIP Service           ││
│  └────────────────────────────────┘│
│                                     │
│  ┌────────────────────────────────┐│
│  │  kafka (Deployment)            ││
│  │  - Confluent Kafka 7.5.0       ││
│  │  - Port 9092                   ││
│  └────────────────────────────────┘│
│                                     │
│  ┌────────────────────────────────┐│
│  │  zookeeper (Deployment)        ││
│  │  - Kafka coordination          ││
│  └────────────────────────────────┘│
└─────────────────────────────────────┘
```

### Scaling Strategy

**Horizontal Pod Autoscaling**:
- Creator agents: Scale based on `START_WORKFLOW` topic lag
- Validator agents: Scale based on `SPEC_CREATED` topic lag
- **MCP Server**: Scale based on CPU/memory usage or request rate (2 replicas default)
- Orchestrator: Fixed (1 replica typically)

**Resource Allocation**:
```yaml
# Agents
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "2000m"

# MCP Server (lighter)
resources:
  requests:
    memory: "256Mi"
    cpu: "250m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

## Data Flow

### Complete Workflow

1. **API Request**
   ```
   POST /workflow → Orchestrator
   ```

2. **Job Creation**
   ```
   Job ID generated → Stored in memory → Returned to user
   ```

3. **Workflow Initiation**
   ```
   Orchestrator → Kafka[START_WORKFLOW] → Creator
   ```

4. **Spec Creation**
   ```
   Creator → CrewAI → JOLT Spec → Kafka[SPEC_CREATED]
   ```

### 5. Validation
   ```
   Validator ← Kafka[SPEC_CREATED]
   Validator → HTTP POST http://mcp-server:8080/transform (JOLT Spec + Input JSON)
   MCP Server → Performs transformation
   MCP Server → HTTP 200 (Transformed JSON)
   Validator → Compare output with expected
   Validator → Kafka[VALIDATION_COMPLETED]
   ```

6. **Agentic Decision**
   ```
   Orchestrator ← Kafka[VALIDATION_COMPLETED]
   If Valid:
     Orchestrator → Kafka[WORKFLOW_COMPLETE]
   If Invalid:
     Orchestrator → Kafka[HUMAN_REVIEW_REQUESTED]
   ```

7. **Human-in-the-Loop (If Invalid)**
   ```
   User → GET /reviews (List pending)
   User → POST /reviews/{id}/feedback (Submit feedback)
   Orchestrator → Kafka[FEEDBACK_RECEIVED]
   Creator ← Kafka[FEEDBACK_RECEIVED]
   Creator → Refine Spec → Kafka[SPEC_CREATED] (Loop continues)
   ```

8. **Status Update**
   ```
   Orchestrator ← Kafka[WORKFLOW_COMPLETE]
   Background Consumer → Updates job status
   ```

9. **Status Query**
   ```
   GET /workflow/{job_id} → Returns result
   ```

## Design Patterns

### 1. Factory Pattern
**Usage**: Message bus creation
```python
bus = get_message_bus()  # Returns InMemory or Kafka
```

### 2. Wrapper Pattern
**Usage**: Agent abstraction
```python
class AgentWrapper:
    def __init__(self, agent, bus, name):
        self.agent = agent
        self.bus = bus
        self.setup_subscriptions()
```

### 3. Pub/Sub Pattern
**Usage**: Event-driven messaging
```python
bus.subscribe("START_WORKFLOW", handler)
bus.publish(Message(type="SPEC_CREATED", payload={...}))
```

### 4. Background Jobs Pattern
**Usage**: Async workflow execution
```python
jobs[job_id] = {"status": "running"}
background_tasks.add_task(run_workflow_async, job_id, ...)
```

## Security Considerations

### 1. API Key Management
- Stored in Kubernetes Secrets
- Never logged or exposed
- Environment variable injection

### 2. Network Policies
- Pod-to-pod communication restrictions
- Kafka access control
- API rate limiting

### 3. Data Privacy
- No persistent storage of sensitive data
- In-memory job storage (ephemeral)
- Secure Kafka communication (TLS in production)

## Fault Tolerance

### 1. Kafka Consumer Groups
- Multiple agent replicas in same consumer group
- Automatic load balancing
- Message replay on failure

### 2. Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
readinessProbe:
  httpGet:
    path: /health
    port: 8000
```

### 3. Restart Policies
```yaml
restartPolicy: Always
```

### 4. Message Retry
- Kafka message retention
- Consumer offset management
- Error topic for failed messages

## Performance Optimization

### 1. Caching
- Agent model caching
- Kafka producer pooling
- Connection re-use

### 2. Batch Processing
- Kafka message batching
- Bulk API operations

### 3. Resource Limits
- Memory limits prevent OOM
- CPU limits prevent resource starvation
- Request/limit ratios for bursting

## Monitoring & Observability

### Recommended Tools
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Kafka Manager**: Topic monitoring
- **K8s Dashboard**: Cluster monitoring

### Key Metrics
- Message throughput (messages/sec)
- Processing latency (ms)
- Queue depth (messages pending)
- Error rate (errors/min)
- Resource utilization (CPU/Memory)

## Future Enhancements

1. **Database Integration**: Persistent job storage (PostgreSQL/MongoDB)
2. **Authentication**: JWT-based API auth
3. **Rate Limiting**: Per-user request limits
4. **Advanced Routing**: Conditional message routing
5. **Observability**: Distributed tracing (Jaeger/Zipkin)
6. **Multi-Tenancy**: Namespace isolation
7. **GitOps**: ArgoCD/Flux deployment
8. **Service Mesh**: Istio for advanced networking

---

**Architecture Version**: 2.0 | **Last Updated**: 2025-11-25
