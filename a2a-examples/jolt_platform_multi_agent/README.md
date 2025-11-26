# JOLT Multi-Agent Platform

A production-ready multi-agent platform for automated JOLT specification creation and validation using CrewAI and LangChain, with Kubernetes deployment and Kafka-based event-driven architecture.

## 🌟 Overview

This platform combines the strengths of two leading AI frameworks:
- **CrewAI**: For collaborative JOLT spec generation
- **LangChain**: For intelligent spec validation
- **Kafka**: For distributed event-driven communication
- **MCP Server**: For JOLT transformations
- **Kubernetes**: For scalable deployment

## 🏗️ Architecture

The platform supports two execution modes:

### Traditional Mode (Synchronous)
Direct orchestration where the platform calls agents sequentially.

### A2A Mode (Event-Driven)
Agents communicate via Kafka message bus:
```
API Request → Orchestrator → [START_WORKFLOW] → Creator Agent
                                     ↓
                              [SPEC_CREATED]
                                     ↓
                              Validator Agent → [WORKFLOW_COMPLETE]
                                     ↓
                              Orchestrator (updates job status)
```

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
export GOOGLE_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-2.0-flash"

# Run traditional mode
python quickstart.py

# Run API server locally
uvicorn jolt_platform.api_server:app --host 0.0.0.0 --port 8000
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
./deploy_k8s.sh YOUR_GOOGLE_API_KEY

# Check status
kubectl get pods

# Port forward API
kubectl port-forward svc/jolt-orchestrator 8000:8000

# Test workflow
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"name": "Alice"}},
    "expected_output": {"fullName": "Alice"},
    "execution_mode": "a2a"
  }'
```

### Docker Compose (Local Testing)

```bash
# Start all services
docker-compose up

# Test API
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "input_json": {"user": {"name": "Test"}},
    "expected_output": {"fullName": "Test"},
    "execution_mode": "a2a"
  }'
```

## 📁 Project Structure

```
.
├── agents/                      # Agent implementations
│   ├── crewai_jolt_agent.py    # CrewAI JOLT creator
│   ├── langchain_validation_agent.py  # LangChain validator
│   ├── creator_worker.py        # Creator worker (K8s)
│   └── validator_worker.py      # Validator worker (K8s)
├── jolt_platform/              # Core platform
│   ├── unified_platform.py     # Main orchestrator
│   ├── api_server.py           # FastAPI server
│   ├── messaging.py            # Message bus (In-Memory/Kafka)
│   └── agent_wrappers.py       # Agent-to-Agent wrappers
├── mcp_servers/                # MCP servers
│   └── jolt_server.py          # JOLT transformation server
├── k8s/                        # Kubernetes manifests
│   ├── config.yaml             # ConfigMap
│   ├── kafka.yaml              # Kafka & Zookeeper
│   ├── kafka-topics-job.yaml   # Topic creation
│   ├── orchestrator.yaml       # API server
│   ├── creator.yaml            # Creator agent
│   └── validator.yaml          # Validator agent
├── Dockerfile.*                # Docker images
├── docker-compose.yaml         # Local testing
├── deploy_k8s.sh              # Deployment script
└── cleanup_k8s.sh             # Cleanup script
```

## 🔧 Key Features

### Multi-Agent Collaboration
- **Creator Agent (CrewAI)**: Generates JOLT specs using collaborative AI
- **Validator Agent (LangChain)**: Validates specs with intelligent analysis

### Event-Driven Architecture
- Kafka-based message bus for distributed communication
- Asynchronous workflow execution
- Job tracking and status updates

### MCP Integration
- Model Context Protocol server for JOLT transformations
- Clean separation of concerns
- Reusable transformation service

### Kubernetes Native
- Horizontal scaling of agents
- High availability
- Resource management
- Production-ready deployment

## 📊 API Endpoints

### Health Check
```bash
GET /health
```

### Create JOLT Spec
```bash
POST /create
{
  "input_json": {...},
  "expected_output": {...}
}
```

### Validate JOLT Spec
```bash
POST /validate
{
  "jolt_spec": [...],
  "input_json": {...},
  "expected_output": {...}
}
```

### Complete Workflow
```bash
POST /workflow
{
  "input_json": {...},
  "expected_output": {...},
  "execution_mode": "a2a",  # or "traditional"
  "async_mode": false
}
```

### Check Job Status
```bash
GET /workflow/{job_id}
```

## 🔐 Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Google Gemini API key (required)
- `GEMINI_MODEL`: Model name (default: `gemini-2.0-flash`)
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka servers (for A2A mode)
- `KAFKA_GROUP_ID`: Consumer group ID

### Kubernetes ConfigMap

Edit `k8s/config.yaml`:
```yaml
data:
  GEMINI_MODEL: "gemini-2.0-flash"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
  KAFKA_GROUP_ID: "jolt-group"
```

## 📖 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and design
- **[DEPLOYMENT.md](DEPLOYMENT.md)**: Deployment guide
- **[USER_GUIDE.md](USER_GUIDE.md)**: User guide and examples

## 🧪 Testing

```bash
# Run quickstart example
python quickstart.py

# Run example workflow
python example_workflow.py

# Test API endpoints
curl http://localhost:8000/health
```

## 🛠️ Development

### Adding a New Agent

1. Create agent class in `agents/`
2. Create wrapper in `jolt_platform/agent_wrappers.py`
3. Register in `unified_platform.py`
4. Create Dockerfile and K8s manifest

### Modifying Message Flow

1. Update message types in `messaging.py`
2. Update agent wrappers to handle new messages
3. Update topic creation in `k8s/kafka-topics-job.yaml`

## 🌐 Deployment Options

### Local (Development)
```bash
python quickstart.py
```

### Docker Compose (Testing)
```bash
docker-compose up
```

### Kubernetes (Production)
```bash
./deploy_k8s.sh YOUR_API_KEY
```

## 📈 Scaling

### Scale Agents Horizontally
```bash
kubectl scale deployment jolt-creator --replicas=3
kubectl scale deployment jolt-validator --replicas=2
```

### Kafka Partitions
Edit `k8s/kafka-topics-job.yaml` to increase partitions for higher throughput.

## 🧹 Cleanup

```bash
# Kubernetes
./cleanup_k8s.sh

# Docker Compose
docker-compose down -v
```

## 🤝 Contributing

This is a research/demo project for exploring multi-agent architectures with Kubernetes and Kafka.

## 📝 License

MIT License

## 🙏 Acknowledgments

- CrewAI for multi-agent collaboration
- LangChain for AI orchestration
- Kafka for distributed messaging
- Google Gemini for AI capabilities
- Model Context Protocol for standardized tool integration

---

**Status**: ✅ Production Ready | **Version**: 1.0.0 | **Last Updated**: 2025-11-25
