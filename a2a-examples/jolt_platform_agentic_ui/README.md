# Multi-Agent Jolt Specification System

A sophisticated multi-agent system demonstrating heterogeneous agent architectures, MCP integration, and Agent-to-Agent (A2A) collaborative debugging protocol.

## 🎯 Overview

This project showcases:
- **Multi-Agent Architecture**: CrewAI (Generation) + LangGraph (Validation)
- **MCP Server**: Secure file access via Model Context Protocol
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
│              FastAPI Orchestrator (Port 8000)               │
│    • Authentication & Authorization                         │
│    • Task Management                                        │
│    • Agent Coordination                                     │
└──────┬──────────────────────┬──────────────────────┬────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ MCP Server  │      │  Agent 1    │      │   Agent 2       │
│  (GDrive)   │      │  (CrewAI)   │◄────►│  (LangGraph)    │
│             │      │  Generator  │ A2A  │   Validator     │
└─────────────┘      └─────────────┘      └─────────────────┘
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
azure-nebula/
├── orchestrator/          # FastAPI Backend
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
├── frontend/             # Streamlit UI
│   ├── app.py           # Main page
│   └── pages/           # Sub-pages
├── mcp_server/          # MCP Server
│   ├── server.py        # Server implementation
│   └── storage/         # Mock file storage
└── requirements.txt
```

## 🔌 API Endpoints

### Orchestrator Service (Port 8088)
- `GET /health` - Health check
- `POST /workflow/generate-and-validate` - Complete workflow with A2A
- `GET /workflow/status/{task_id}` - Get workflow status
- `WS /ws/status/{task_id}` - WebSocket for real-time updates

### Generator Service (Port 8081)
- `GET /health` - Health check
- `POST /generate` - Generate Jolt spec from input/output files
- `POST /refine` - Refine Jolt spec based on validation errors (A2A endpoint)

### Validator Service (Port 8080)
- `GET /health` - Health check
- `POST /validate` - Single validation attempt
- `POST /validate-with-a2a` - Validate with A2A collaborative debugging loop

## 🔄 Workflow Execution Flow

The system implements true **Agent-to-Agent (A2A) Collaborative Debugging** as per the project brief:

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

**Key Points:**
- ✅ **Orchestrator** delegates to agents but doesn't manage refinement
- ✅ **Validator** directly communicates with Generator (A2A)
- ✅ **Generator** refines specs based on validation feedback
- ✅ **No orchestrator involvement** in the refinement loop

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
- **Tools**: MCP File Reader

### Agent 2: Validator (LangGraph)
- **Role**: Validation & Quality Assurance
- **Task**: Execute Jolt transformation and validate results
- **Features**: A2A Collaborative Debugging Protocol

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
- **Data**: JSON, DeepDiff

## 📖 Usage Example

1. Navigate to the **Auth** page and configure your token
2. Go to the **Workflow** page
3. Set input files (default: `input.json`, `output.json`)
4. Click "Run Complete Workflow"
5. View the A2A messages in the results tabs

## 🎓 Learning Resources

See `project_brief.md` for detailed architecture and design decisions.

## 📄 License

MIT
