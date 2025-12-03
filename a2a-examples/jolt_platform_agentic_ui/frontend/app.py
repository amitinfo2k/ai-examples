import streamlit as st
import requests
import json
import os

st.set_page_config(
    page_title="Multi-Agent Jolt System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Jolt Specification System")

st.markdown("""
### Welcome to the Multi-Agent Jolt System!

This system demonstrates:
- **Multi-Agent Architecture**: CrewAI (Generator) + LangGraph (Validator)
- **MCP Integration**: Secure Google Drive file access
- **A2A Protocol**: Agent-to-Agent Collaborative Debugging
- **Orchestrated Workflow**: Authentication → Generation → Validation

#### Getting Started
1. **Configure Authentication** - Set up your Google Drive access token
2. **Run Workflow** - Trigger the Jolt spec generation and validation
3. **View Results** - See the A2A collaboration in action

Use the sidebar to navigate between pages.
""")

# Initialize session state
if 'auth_token' not in st.session_state:
    st.session_state.auth_token = 'valid_token'  # Default for demo
if 'orchestrator_url' not in st.session_state:
    st.session_state.orchestrator_url = os.getenv('ORCHESTRATOR_URL', 'http://localhost:8088')

# Sidebar - Global Config
st.sidebar.title("⚙️ Configuration")
st.session_state.orchestrator_url = st.sidebar.text_input(
    "Orchestrator URL",
    value=st.session_state.orchestrator_url
)

# Health check
try:
    response = requests.get(f"{st.session_state.orchestrator_url}/health", timeout=2)
    if response.status_code == 200:
        st.sidebar.success("✅ Orchestrator Online")
    else:
        st.sidebar.error(f"❌ Orchestrator Error ({response.status_code})")
except requests.exceptions.ConnectionError:
    st.sidebar.error("❌ Cannot connect to Orchestrator")
    st.sidebar.caption(f"Make sure it's running on {st.session_state.orchestrator_url}")
except requests.exceptions.Timeout:
    st.sidebar.warning("⚠️ Orchestrator Timeout")
except Exception as e:
    st.sidebar.warning(f"⚠️ Orchestrator: {str(e)}")

st.info("👈 Use the sidebar to navigate to different pages")
