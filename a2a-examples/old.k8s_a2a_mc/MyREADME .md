
# Run MCP Server 

cd k8s_a2a_mc
uv venv
source .venv/bin/activate
uv sync
uv run kubernetes_mcp_server/server.py