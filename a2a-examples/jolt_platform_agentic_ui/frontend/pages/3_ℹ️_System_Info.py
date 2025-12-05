import streamlit as st
import requests
import os

st.set_page_config(page_title="System Info", page_icon="ℹ️", layout="wide")

st.title("ℹ️ System Information")

st.markdown("""
View the system architecture and component status.
""")

# Initialize session state
if 'orchestrator_url' not in st.session_state:
    st.session_state.orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8088')

# Architecture Diagram
st.subheader("🏗️ System Architecture")
st.markdown("""
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
""")

# Component Status
st.subheader("📡 Component Status")

# Get service URLs from environment
orchestrator_url = st.session_state.orchestrator_url
generator_url = os.getenv('GENERATOR_URL', 'http://localhost:8081')
validator_url = os.getenv('VALIDATOR_URL', 'http://localhost:8080')
mcp_service_url = os.getenv('MCP_SERVICE_URL', 'http://localhost:8081')

col1, col2 = st.columns(2)

with col1:
    st.metric("Frontend", "Online", "✅")
    st.caption("Streamlit App")

with col2:
    try:
        response = requests.get(f"{orchestrator_url}/health", timeout=2)
        if response.status_code == 200:
            st.metric("Orchestrator", "Online", "✅")
        else:
            st.metric("Orchestrator", "Error", "❌")
    except:
        st.metric("Orchestrator", "Offline", "⚠️")
    st.caption(f"FastAPI Backend - {orchestrator_url}")

col3, col4 = st.columns(2)

with col3:
    try:
        response = requests.get(f"{generator_url}/health", timeout=2)
        if response.status_code == 200:
            st.metric("Generator Agent", "Online", "✅")
        else:
            st.metric("Generator Agent", "Error", "❌")
    except:
        st.metric("Generator Agent", "Offline", "⚠️")
    st.caption(f"CrewAI Service - {generator_url}")

with col4:
    try:
        response = requests.get(f"{validator_url}/health", timeout=2)
        if response.status_code == 200:
            st.metric("Validator Agent", "Online", "✅")
        else:
            st.metric("Validator Agent", "Error", "❌")
    except:
        st.metric("Validator Agent", "Offline", "⚠️")
    st.caption(f"LangGraph Service - {validator_url}")

col5, col6 = st.columns(2)

with col5:
    try:
        # Try to connect to Jolt MCP Server SSE endpoint
        response = requests.get(f"{mcp_service_url}/sse", timeout=2, stream=True)
        if response.status_code == 200:
            st.metric("Jolt MCP Server", "Online", "✅")
        else:
            st.metric("Jolt MCP Server", "Error", "❌")
    except:
        st.metric("Jolt MCP Server", "Offline", "⚠️")
    st.caption(f"JOLT Transform Engine - {mcp_service_url}")

with col6:
    try:
        response = requests.get(f"{orchestrator_url}/test-mcp?path=input.json&token=valid_token", timeout=2)
        if response.status_code == 200:
            st.metric("GDrive MCP Server", "Online", "✅")
        else:
            st.metric("GDrive MCP Server", "Error", "❌")
    except:
        st.metric("GDrive MCP Server", "Unknown", "⚠️")
    st.caption("File Access Layer")

st.divider()

# Endpoints
st.subheader("🔌 Available Endpoints")

st.markdown("### Orchestrator Service (Port 8088)")
orchestrator_endpoints = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {"method": "GET", "path": "/test-mcp", "description": "Test MCP file reading"},
    {"method": "POST", "path": "/workflow/generate-and-validate", "description": "Complete workflow with A2A"},
    {"method": "GET", "path": "/workflow/status/{task_id}", "description": "Get workflow status"},
    {"method": "WS", "path": "/ws/status/{task_id}", "description": "WebSocket for real-time updates"},
]

for endpoint in orchestrator_endpoints:
    st.markdown(f"**{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}")

st.markdown("### Generator Service (Port 8081)")
generator_endpoints = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {"method": "POST", "path": "/generate", "description": "Generate Jolt spec from input/output files"},
    {"method": "POST", "path": "/refine", "description": "Refine Jolt spec based on validation errors (A2A endpoint)"},
]

for endpoint in generator_endpoints:
    st.markdown(f"**{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}")

st.markdown("### Validator Service (Port 8080)")
validator_endpoints = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {"method": "POST", "path": "/validate", "description": "Single validation attempt"},
    {"method": "POST", "path": "/validate-with-a2a", "description": "Validate with A2A collaborative debugging loop"},
]

for endpoint in validator_endpoints:
    st.markdown(f"**{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}")

st.divider()

# Technology Stack
st.subheader("🛠️ Technology Stack")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Backend & Orchestration**")
    st.markdown("- FastAPI (Orchestrator)")
    st.markdown("- CrewAI (Generator Agent)")
    st.markdown("- LangGraph (Validator Agent)")
    st.markdown("- LangChain (Agent Framework)")

with col2:
    st.markdown("**MCP Servers & Tools**")
    st.markdown("- Google Drive MCP (Python)")
    st.markdown("- Jolt MCP Server (Go)")
    st.markdown("- MCP Protocol (SSE Transport)")
    st.markdown("- DeepDiff (Validation)")

with col3:
    st.markdown("**Frontend & Deployment**")
    st.markdown("- Streamlit (UI)")
    st.markdown("- Docker (Containerization)")
    st.markdown("- Kubernetes/Kind (Orchestration)")
    st.markdown("- Google Gemini (LLM)")

st.divider()

# A2A Protocol
st.subheader("🔄 A2A Collaborative Debugging Protocol")
st.markdown("""
The system implements **Agent-to-Agent (A2A) Collaborative Debugging** where the Validator 
and Generator agents directly communicate to refine JOLT specifications:

1. **ERROR_REPORT**: Validator → Generator (validation issues found)
2. **PATCH_PROPOSAL**: Generator → Validator (proposed spec fixes)
3. **VERIFICATION_RESULT**: Validator → Generator (final validation result)

This enables autonomous spec refinement without orchestrator intervention.
""")
