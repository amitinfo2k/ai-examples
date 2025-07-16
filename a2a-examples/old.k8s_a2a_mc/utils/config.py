"""
Configuration utilities for the Kubernetes CrashLoopBackOff diagnosis system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MCP server configuration
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))

# Agent configuration
K8S_AGENT_HOST = os.getenv("K8S_AGENT_HOST", "0.0.0.0")
K8S_AGENT_PORT = int(os.getenv("K8S_AGENT_PORT", "8001"))

LOG_AGENT_HOST = os.getenv("LOG_AGENT_HOST", "0.0.0.0")
LOG_AGENT_PORT = int(os.getenv("LOG_AGENT_PORT", "8002"))

ORCHESTRATOR_AGENT_HOST = os.getenv("ORCHESTRATOR_AGENT_HOST", "0.0.0.0")
ORCHESTRATOR_AGENT_PORT = int(os.getenv("ORCHESTRATOR_AGENT_PORT", "8003"))

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# URLs
K8S_MCP_SERVER_URL = os.getenv("K8S_MCP_SERVER_URL", f"http://localhost:{MCP_SERVER_PORT}")
K8S_AGENT_URL = os.getenv("K8S_AGENT_URL", f"http://localhost:{K8S_AGENT_PORT}")
LOG_AGENT_URL = os.getenv("LOG_AGENT_URL", f"http://localhost:{LOG_AGENT_PORT}")
ORCHESTRATOR_AGENT_URL = os.getenv("ORCHESTRATOR_AGENT_URL", f"http://localhost:{ORCHESTRATOR_AGENT_PORT}")
