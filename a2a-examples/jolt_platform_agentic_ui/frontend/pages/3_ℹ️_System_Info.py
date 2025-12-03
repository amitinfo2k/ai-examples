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
│              FastAPI Orchestrator (Port 8000)               │
│    • Authentication & Authorization                         │
│    • Task Management                                        │
│    • Agent Coordination                                     │
└──────┬──────────────────────┬──────────────────────┬────────┘
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ MCP Server  │      │  Agent 1    │      │   Agent 2       │
│  (GDrive)   │      │  (CrewAI)   │      │  (LangGraph)    │
│             │      │  Generator  │◄────►│   Validator     │
│ Mock Files  │      │             │ A2A  │                 │
└─────────────┘      └─────────────┘      └─────────────────┘
```
""")

# Component Status
st.subheader("📡 Component Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Frontend", "Online", "✅")
    st.caption("Streamlit App")

with col2:
    try:
        response = requests.get(f"{st.session_state.orchestrator_url}/health", timeout=2)
        if response.status_code == 200:
            st.metric("Orchestrator", "Online", "✅")
        else:
            st.metric("Orchestrator", "Error", "❌")
    except:
        st.metric("Orchestrator", "Offline", "⚠️")
    st.caption("FastAPI Backend")

with col3:
    try:
        response = requests.get(f"{st.session_state.orchestrator_url}/test-mcp?path=input.json&token=valid_token", timeout=2)
        if response.status_code == 200:
            st.metric("MCP Server", "Online", "✅")
        else:
            st.metric("MCP Server", "Error", "❌")
    except:
        st.metric("MCP Server", "Unknown", "⚠️")
    st.caption("File Access Layer")

st.divider()

# Endpoints
st.subheader("🔌 Available Endpoints")
endpoints = [
    {"method": "GET", "path": "/health", "description": "Health check"},
    {"method": "GET", "path": "/test-mcp", "description": "Test MCP file reading"},
    {"method": "POST", "path": "/generate", "description": "Generate Jolt spec (CrewAI)"},
    {"method": "POST", "path": "/validate", "description": "Validate Jolt spec (LangGraph)"},
    {"method": "POST", "path": "/workflow/generate-and-validate", "description": "Complete workflow with A2A"},
]

for endpoint in endpoints:
    st.markdown(f"**{endpoint['method']}** `{endpoint['path']}` - {endpoint['description']}")

st.divider()

# Technology Stack
st.subheader("🛠️ Technology Stack")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Backend**")
    st.markdown("- FastAPI (Orchestrator)")
    st.markdown("- CrewAI (Generator Agent)")
    st.markdown("- LangGraph (Validator Agent)")
    st.markdown("- MCP (File Access Protocol)")

with col2:
    st.markdown("**Frontend**")
    st.markdown("- Streamlit (UI)")
    st.markdown("- Requests (HTTP Client)")
    st.markdown("- JSON (Data Format)")
