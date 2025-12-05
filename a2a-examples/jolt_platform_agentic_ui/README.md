# Multi-Agent Jolt Specification System

A sophisticated multi-agent system demonstrating heterogeneous agent architectures, MCP integration, and Agent-to-Agent (A2A) collaborative debugging protocol.

## 🎯 Overview

This project showcases:
- **Multi-Agent Architecture**: CrewAI (Generation) + LangGraph (Validation)
- **MCP Servers**: 
  - **Google Drive MCP**: Secure file access via Model Context Protocol
  - **Jolt MCP Server**: JOLT transformation engine for the Validator agent
- **A2A Protocol**: Agent-to-Agent Collaborative Debugging
- **Full Stack**: FastAPI (Backend) + Streamlit (Frontend)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│          (Auth · Workflow · Visualization)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Orchestrator (Port 8088)               │
│    • Authentication & Authorization                         │
│    • Task Management                                        │
│    • Agent Coordination                                     │
└──────┬──────────────────────┬──────────────────────┬────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ GDrive MCP  │      │  Agent 1    │      │   Agent 2       │
│   Server    │◄─────│  (CrewAI)   │◄────►│  (LangGraph)    │
│ (File I/O)  │      │  Generator  │ A2A  │   Validator     │
└─────────────┘      └─────────────┘      └────────┬────────┘
                                                    │
                                                    ▼
                                          ┌─────────────────┐
                                          │  Jolt MCP       │
                                          │  Server         │
                                          │ (Transform)     │
                                          └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

1. **Create Virtual Environment**
```bash
python3 -m venv venv
```

2. **Activate Virtual Environment**
```bash
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API Key**
```bash
# Get your Gemini API key from https://aistudio.google.com/app/apikey
export GOOGLE_API_KEY="your-api-key-here"

# Optional: Configure Gemini model (defaults to gemini-2.0-flash-exp)
export GEMINI_MODEL="gemini/gemini-2.0-flash-exp"

# Or create a .env file
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY and GEMINI_MODEL
```

5. **Start the Orchestrator**

   **Important**: To ensure you are using the packages installed in the virtual environment, you must run the application using the Python executable from the `venv` directory.

```bash
# Use this command to run the orchestrator
venv/bin/python3 -m orchestrator.main
```

3. **Start the Frontend (in a new terminal)**
```bash
cd frontend
streamlit run app.py
```

4. **Access the Application**
- Frontend: http://localhost:8501
- API Docs: http://localhost:8088/docs

## 📂 Project Structure

```
jolt_platform_agentic_ui/
├── orchestrator/         # FastAPI Backend
│   ├── main.py           # API endpoints
│   ├── models/           # Pydantic schemas
│   └── core/             # MCP client
├── agents/               # Agent implementations
│   ├── generator/        # CrewAI Agent
│   │   ├── crew_agent.py
│   │   └── tools.py
│   └── validator/        # LangGraph Agent
│       ├── langgraph_agent.py
│       ├── a2a_protocol.py
│       └── jolt_utils.py
├── frontend/            # Streamlit UI
│   ├── app.py           # Main page
│   └── pages/           # Sub-pages
├── mcp_server/          # MCP Server
│   ├── server.py        # Server implementation
│   └── storage/         # Mock file storage
└── requirements.txt
```

## 🔌 API Endpoints

### Orchestrator Service (Port 8088)

#### Core Endpoints
- `GET /health` - Health check

#### Workflow Endpoints
- `POST /workflow/generate-and-validate` - Complete workflow with A2A
- `GET /workflow/status/{task_id}` - Get workflow status
- `WS /ws/status/{task_id}` - WebSocket for real-time updates

#### Agent Proxy Endpoints
- `POST /generate` - Proxy to Generator for spec generation
- `POST /validate` - Proxy to Validator for validation
- `POST /refine-with-prompt` - HITL: AI-assisted refinement

#### Task Management
- `POST /tasks` - Create a new task
- `GET /tasks/{task_id}` - Get task status

#### File Management (GDrive)
- `GET /files/list` - List files in storage
- `GET /files/read` - Read file content
- `POST /files/write` - Write/upload file
- `DELETE /files/delete` - Delete file

#### Debug
- `GET /test-mcp` - Test MCP server connectivity

### Generator Service (Port 8081)
- `GET /health` - Health check
- `POST /generate` - Generate Jolt spec from input/output files
- `POST /refine` - Refine Jolt spec based on validation errors (A2A endpoint)
- `POST /refine-with-prompt` - HITL: Refine spec based on natural language feedback

### Validator Service (Port 8080)
- `GET /health` - Health check
- `POST /validate` - Single validation attempt
- `POST /validate-with-a2a` - Validate with A2A collaborative debugging loop

## 🔄 Workflow Execution Flow

The system implements true **Agent-to-Agent (A2A) Collaborative Debugging** with **Human-in-the-Loop (HITL)** fallback:

### Phase 1: Automatic A2A Flow

```
┌─────────────┐
│ Orchestrator│
└──────┬──────┘
       │
       ├─1─→ Generator: POST /generate (initial spec)
       │     └─returns→ jolt_spec
       │
       ├─2─→ Validator: POST /validate-with-a2a (spec + max_retries=3)
       │     │
       │     └─→ [Internal A2A Loop]:
       │         ┌─→ Validate spec
       │         ├─→ If fails: Send ERROR_REPORT to Generator
       │         ├─→ Generator: POST /refine
       │         ├─→ Receives PATCH_PROPOSAL (refined spec)
       │         ├─→ Loop until success or max retries
       │         └─→ Returns final result
       │
       └─3─→ Receives final validated spec & result
```

### Phase 2: Human-in-the-Loop (HITL) Flow

When automatic A2A debugging exhausts retries, the user can intervene:

```
┌─────────────────────────────────────────────────────────────────┐
│                     HITL Interface (Frontend)                   │
├─────────────────────────────────────────────────────────────────┤
│  A2A Failed after 3 retries                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Validation Errors:                                      │    │
│  │  • Path: output.product_name                            │    │
│  │    Expected: "Widget Pro"                               │    │
│  │    Actual: null                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ 💬 AI-Assisted   │    │ ✏️ Manual Edit   │                   │
│  └────────┬─────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌─────────────────┐    ┌─────────────────────────────────┐     │
│  │ User Feedback:  │    │ Direct JSON Editor:              │     │
│  │ "Map product.   │    │ [{"operation": "shift",          │     │
│  │  name to        │    │   "spec": {                      │     │
│  │  product_name"  │    │     "product.name": "product_..." │     │
│  └────────┬────────┘    └────────┬────────────────────────┘     │
│           │                       │                              │
└───────────┼───────────────────────┼──────────────────────────────┘
            │                       │
            ▼                       ▼
     ┌─────────────┐         ┌─────────────┐
     │  Generator  │         │  Validator  │
     │ POST /refine│         │ POST /      │
     │ -with-prompt│         │  validate   │
     └──────┬──────┘         └──────┬──────┘
            │                       │
            └───────────┬───────────┘
                        ▼
              ┌─────────────────┐
              │ Re-validation   │
              │ with new spec   │
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌────────┐   ┌────────┐   ┌────────┐
     │Success!│   │ Retry  │   │ Manual │
     │   ✅   │   │  HITL  │   │  Edit  │
     └────────┘   └────────┘   └────────┘
```

### Complete Flow Summary

| Phase | Actor | Action | Fallback |
|-------|-------|--------|----------|
| 1 | Orchestrator | Initiates generation | - |
| 2 | Generator (CrewAI) | Creates initial spec | - |
| 3 | Validator (LangGraph) | Validates spec | A2A refinement |
| 4 | A2A Loop | Auto-refinement (3 retries) | HITL |
| 5 | User (HITL) | AI-assisted or manual fix | Continue/abandon |
| 6 | System | Re-validation | Loop to HITL |

**Key Points:**
- ✅ **Orchestrator** delegates to agents but doesn't manage refinement
- ✅ **Validator** directly communicates with Generator (A2A)
- ✅ **Generator** refines specs based on validation feedback
- ✅ **No orchestrator involvement** in the refinement loop
- ✅ **HITL** provides fallback when A2A exhausts retries
- ✅ **Two HITL modes**: AI-assisted (natural language) and manual editing

## 📁 File Placement (Mock Storage)

Since this is a demo with mock Google Drive, place your JSON files in:

```
mcp_server/storage/
```

**Sample files provided:**
- `input.json` - Sample input JSON
- `output.json` - Expected output JSON

**To use your own files:**
1. Replace or add JSON files to `mcp_server/storage/`
2. In the Workflow page, enter the filename (e.g., `myinput.json`)
3. The system will read from the storage directory

> **Note:** In production, this would connect to real Google Drive and users would select files from their Drive folders.

## 🤖 Agent Details

### Agent 1: Generator (CrewAI)
- **Role**: Jolt Specification Expert
- **Task**: Analyze input/output JSON and generate transformation spec
- **Tools**: Google Drive MCP Server (File Reader)
- **Port**: 8081 (when running as service)

### Agent 2: Validator (LangGraph)
- **Role**: Validation & Quality Assurance
- **Task**: Execute Jolt transformation and validate results
- **Tools**: 
  - **Jolt MCP Server**: Performs JOLT transformations via MCP protocol
  - **DeepDiff**: Compares actual vs expected output
- **Features**: 
  - A2A Collaborative Debugging Protocol
  - Direct communication with Generator for spec refinement
- **Port**: 8080 (when running as service)

### Jolt MCP Server
- **Role**: JOLT Transformation Engine
- **Technology**: Go-based MCP server with SSE transport
- **Capabilities**:
  - `transform` tool: Applies JOLT specifications to JSON data
  - Supports operations: shift, default, remove, sort, cardinality
- **Integration**: Used exclusively by the Validator agent for transformations
- **Port**: 8081 (MCP SSE endpoint)
- **Deployment Modes**:
  - `mcp-sse`: HTTP-based MCP with Server-Sent Events (Kubernetes)
  - `mcp`: stdio-based MCP (local development)
  - `server`: Plain HTTP API mode

## 🔐 Authentication

**For MCP (Mock Storage):** Use the default token `valid_token`

**For Gemini API:** Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

In production, this would integrate with Google OAuth for real Google Drive access.

## 📝 A2A Protocol

The Agent-to-Agent protocol supports:
- `ERROR_REPORT`: Validator → Generator (issues found)
- `PATCH_PROPOSAL`: Generator → Validator (proposed fixes)
- `VERIFICATION_RESULT`: Validator → Generator (final result)

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Agents**: CrewAI, LangChain, LangGraph
- **Frontend**: Streamlit
- **Protocol**: MCP (Model Context Protocol)
- **MCP Servers**:
  - Google Drive MCP (Python) - File access
  - Jolt MCP Server (Go) - JSON transformations
- **Data**: JSON, DeepDiff
- **Deployment**: Docker, Kubernetes (Kind)
- **LLM**: Google Gemini (gemini-2.5-flash)

## 📖 Usage Example

1. Navigate to the **Auth** page and configure your token
2. Go to the **Workflow** page
3. Set input files (default: `input.json`, `output.json`)
4. Click "Run Complete Workflow"
5. View the A2A messages in the results tabs

## 🐳 Kubernetes Deployment

The system can be deployed to Kubernetes using the provided manifests:

### Prerequisites
- Docker
- Kind (Kubernetes in Docker)
- kubectl

### Quick Deploy

```bash
# Build and load Docker images
make build-all
make load-all

# Deploy to Kubernetes
make deploy

# Check status
make status

# View logs
make logs-validator
make logs-generator
make logs-orchestrator
```

### Architecture Components

The Kubernetes deployment includes:

1. **Namespace**: `jolt-platform`
2. **ConfigMap**: Environment variables and service URLs
3. **Secrets**: API keys (GOOGLE_API_KEY)
4. **Services**:
   - `jolt-orchestrator-service` (Port 80)
   - `jolt-validator-service` (Port 80)
   - `jolt-generator-service` (Port 80)
   - `jolt-mcp-service` (Port 8081) - Jolt MCP Server
   - `jolt-frontend-service` (Port 8501)
5. **Deployments**: All services with health checks and resource limits

### Environment Variables

Configured in `k8s/manifests/01-configmap.yaml`:
- `MCP_SERVICE_URL`: Jolt MCP Server endpoint
- `ORCHESTRATOR_URL`: Orchestrator service
- `VALIDATOR_URL`: Validator service
- `GENERATOR_URL`: Generator service
- `GEMINI_MODEL`: LLM model selection

## 📊 Observability & Tracing

The system supports comprehensive tracing through **LangSmith** for both CrewAI and LangGraph agents.

### LangSmith Integration

Both agents send traces to LangSmith for monitoring, debugging, and analysis:

| Agent | Framework | Tracing Method |
|-------|-----------|----------------|
| Generator | CrewAI | OpenInference Instrumentation |
| Validator | LangGraph | Native LangChain Tracing |

### Enabling Tracing

1. **Get LangSmith API Key**: Sign up at [smith.langchain.com](https://smith.langchain.com/) and create an API key

2. **Configure Environment Variables** (in `k8s/manifests/01-configmap.yaml`):
```yaml
# LangSmith Tracing Configuration
LANGCHAIN_TRACING_V2: "true"
LANGCHAIN_PROJECT: "jolt-platform"
LANGCHAIN_ENDPOINT: "https://api.smith.langchain.com"
```

3. **Add API Key** (in `k8s/manifests/07-secrets.yaml`):
```yaml
# Base64 encode your key: echo -n "your-api-key" | base64
LANGCHAIN_API_KEY: <base64-encoded-api-key>
```

4. **Apply and Restart**:
```bash
kubectl apply -f k8s/manifests/
kubectl rollout restart deployment -n jolt-platform
```

### Trace Data Captured

**CrewAI Generator Traces** (via OpenInference):
- Agent task execution
- Tool calls (MCP file reads)
- LLM calls and responses
- Task completion status

**LangGraph Validator Traces** (native):
- State graph traversal
- Node execution
- A2A protocol messages
- Validation results

### Disabling Tracing

Set `LANGCHAIN_TRACING_V2: "false"` in the configmap to disable tracing.

## 🧑‍💻 Human-in-the-Loop (HITL)

The system provides robust Human-in-the-Loop capabilities for debugging and refining Jolt specifications when automatic validation fails.

### HITL Features

#### 1. AI-Assisted Refinement (Prompt-Based)
When validation fails, users can provide **natural language feedback** to guide the AI in fixing the Jolt spec:

```
Example feedback:
"The baseeventid should come from class_uid, not category_uid"
"Map the product.name field to product_name in the output"
"The metadata section is missing the version field"
```

**How it works:**
1. Validation fails and shows errors
2. User provides feedback in natural language
3. AI refines the spec based on feedback + context
4. Refined spec is automatically re-validated
5. Process repeats until successful or user switches to manual editing

#### 2. Manual JSON Editing
For precise control, users can directly edit:
- **Jolt Specification**: Modify the transformation rules
- **Expected Output**: Adjust the expected result for re-validation

### HITL UI Tabs

| Tab | Description |
|-----|-------------|
| 💬 AI-Assisted | Natural language feedback for AI-guided refinement |
| ✏️ Manual Edit | Direct JSON editing of spec and expected output |

### API Endpoints for HITL

```
POST /refine-with-prompt
```
**Request Body:**
```json
{
  "current_spec": [...],
  "user_feedback": "Fix the product name mapping",
  "input_json": {...},
  "expected_output": {...},
  "validation_errors": [...]
}
```

```
POST /validate
```
**Request Body:**
```json
{
  "jolt_spec": [...],
  "expected_output": {...}
}
```

## 📁 GDrive File Browser

The system includes a file browser for managing JSON files in the mock GDrive storage.

### Features

- **Browse Files**: View all files in storage with size and type info
- **Upload Files**: Upload JSON via file picker or paste content directly
- **Preview Content**: View file contents as formatted JSON
- **Select for Workflow**: Choose files as input/output for transformations
- **Delete Files**: Remove files from storage

### Accessing the File Browser

1. Navigate to **📁 File Browser** in the sidebar
2. Authenticate if prompted
3. Use tabs to browse, upload, or view selected files

### Integration with Workflow

Selected files are automatically used as defaults in the Workflow page:
- Input JSON → Used as transformation source
- Output JSON → Used as expected result for validation


## 🎓 Learning Resources

See `project_brief.md` for detailed architecture and design decisions.

## 📄 License

MIT
